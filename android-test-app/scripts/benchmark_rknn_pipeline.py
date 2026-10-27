#!/usr/bin/env python3
"""Run the experimental test app and retain raw images, timing and memory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'normal': ROOT/'samples/obama.jpg',
    'close-up': ROOT/'image-enhancer-comparison/original/close-up.png',
    'elderly': ROOT/'image-enhancer-comparison/original/elderly.jpeg',
    'low-light': ROOT/'image-enhancer-comparison/original/low-light.png',
}
PACKAGE = 'com.enhancer.test'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', default='192.168.2.42:37361')
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--backend', choices=['cpu', 'rknn'], required=True)
    p.add_argument('--collect-only', action='store_true')
    args = p.parse_args()
    image = CASES[args.case]
    output = ROOT/'android-test-app/rknn-experiment/artifacts/pipeline'/args.case/args.backend
    output.mkdir(parents=True, exist_ok=True)
    runs = 1 if args.backend == 'cpu' else 4

    def adb(*parts, binary=False):
        r = subprocess.run(['adb', '-s', args.serial, *parts], capture_output=True, timeout=45)
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors='replace'))
        return r.stdout if binary else r.stdout.decode(errors='replace')

    if not args.collect_only:
        adb('shell', 'am', 'force-stop', PACKAGE)
        adb('push', str(image), '/data/local/tmp/benchmark_input.jpg')
        adb('shell', 'run-as', PACKAGE, 'cp', '/data/local/tmp/benchmark_input.jpg', 'files/benchmark_input.jpg')
        adb('logcat', '-c')
        adb('shell', 'am', 'start', '-n', PACKAGE+'/.MainActivity',
            '--es', 'benchmark_image', f'/data/user/0/{PACKAGE}/files/benchmark_input.jpg',
            '--es', 'benchmark_backend', args.backend, '--ei', 'benchmark_runs', str(runs),
            '--ez', 'benchmark_export', 'true', '--ez', 'benchmark_full_pipeline', 'true')
    deadline = time.monotonic()+150
    logs = ''
    while time.monotonic() < deadline:
        logs = adb('logcat', '-d', '-s', 'EnhancerBench:I', 'EnhancerPipeline:I',
                   'ExperimentalRknn:I', 'AiFilterEnhancer:I', 'EnhancerFaceDiag:I',
                   'EnhancerTest:E', 'AndroidRuntime:E')
        records = re.findall(r'EnhancerBench: backend=(\w+) init_ms=(\d+) total_ms=(\d+) image=(\d+)x(\d+)', logs)
        records = [v for v in records if v[0] == args.backend]
        if len(records) >= runs: break
        if 'FATAL EXCEPTION' in logs:
            (output/'logcat.txt').write_text(logs)
            raise RuntimeError('Application crashed; see retained logcat')
        time.sleep(2)
    else:
        (output/'logcat.txt').write_text(logs)
        raise RuntimeError('Benchmark did not complete; see retained logcat')
    # Timers end before PNG export. Let the last export complete before reading it.
    time.sleep(3)
    (output/'logcat.txt').write_text(logs)
    pipelines = re.findall(r'EnhancerPipeline: run=(\d+) decode_detect_enhance_ms=(\d+)', logs)
    for name in ['source.png', 'source.json']+[f'run-{i}.png' for i in range(1, runs+1)]:
        export_deadline = time.monotonic()+60
        while True:
            data = adb('exec-out', 'run-as', PACKAGE, 'cat', 'files/benchmark-output/'+name, binary=True)
            if not name.endswith('.png') or (data.startswith(b'\x89PNG') and data[-8:-4] == b'IEND'):
                break
            if time.monotonic() >= export_deadline:
                raise RuntimeError(f'Incomplete PNG export: {name}')
            time.sleep(1)
        (output/name).write_bytes(data)
    pid = adb('shell', 'pidof', PACKAGE).strip()
    memory = adb('shell', 'run-as', PACKAGE, 'cat', f'/proc/{pid}/status')
    (output/'process-status.txt').write_text(memory)
    summary = dict(case=args.case, backend=args.backend, input=str(image.relative_to(ROOT)),
                   input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                   pipeline_ms=[int(v[1]) for v in pipelines],
                   sdk_call_ms=[int(v[2]) for v in records], init_ms=[int(v[1]) for v in records])
    for key in ['VmRSS', 'VmHWM']:
        match = re.search(rf'^{key}:\s+(\d+) kB', memory, re.M)
        if match: summary[key+'_kib'] = int(match.group(1))
    if len(summary['pipeline_ms']) != runs:
        raise RuntimeError('Missing complete pipeline timing')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
