#!/usr/bin/env python3
"""Run RKNN conversion with persistent logs and a host-memory guard."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--min-available-mib", type=int, default=1024)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.log_dir.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "status.json"
    command = [sys.executable, "-u", str(Path(__file__).with_name("convert_model_rknn.py")),
               "--input", str(args.input.resolve()), "--output", str(args.output.resolve())]
    status = {"state": "running", "command": command, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "input": str(args.input.resolve()), "output": str(args.output.resolve())}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    pressure_started = None
    peak_tree_rss = 0
    reason = None
    with (args.log_dir / "conversion.log").open("w") as log, (args.log_dir / "memory.csv").open("w", newline="") as memory:
        writer = csv.writer(memory)
        writer.writerow(["elapsed_s", "process_tree_rss_mib", "host_available_mib", "swap_free_mib"])
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status["pid"] = process.pid
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        try:
            while process.poll() is None:
                available_mib = psutil.virtual_memory().available / 1048576
                swap_free_mib = psutil.swap_memory().free / 1048576
                try:
                    parent = psutil.Process(process.pid)
                    processes = [parent] + parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    processes = []
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_tree_rss = max(peak_tree_rss, rss)
                writer.writerow([round(time.monotonic() - started, 2), round(rss / 1048576, 2),
                                 round(available_mib, 2), round(swap_free_mib, 2)])
                memory.flush()
                if available_mib < args.min_available_mib:
                    pressure_started = pressure_started or time.monotonic()
                    if time.monotonic() - pressure_started >= 15:
                        reason = "host_memory_pressure"
                        stop_process(process)
                        break
                else:
                    pressure_started = None
                time.sleep(2)
        except BaseException:
            stop_process(process)
            status.update(state="interrupted", elapsed_s=round(time.monotonic() - started, 2))
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise
        returncode = process.wait()
    success = returncode == 0 and args.output.is_file() and args.output.stat().st_size > 0
    status.update(state="success" if success else "failed", returncode=returncode,
                  elapsed_s=round(time.monotonic() - started, 2),
                  sampled_peak_process_tree_rss_mib=round(peak_tree_rss / 1048576, 2), reason=reason)
    if success:
        digest = hashlib.sha256()
        with args.output.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1048576), b""):
                digest.update(chunk)
        status.update(output_bytes=args.output.stat().st_size, output_sha256=digest.hexdigest())
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
