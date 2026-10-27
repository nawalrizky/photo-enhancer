#!/usr/bin/env python3
"""Compare raw RKNN float outputs against the same input on ONNX CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-mae", type=float, default=0.02)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx_model), options, providers=["CPUExecutionProvider"])
    source = session.get_inputs()[0]
    shape = [1 if i == 0 and not isinstance(d, int) else d for i, d in enumerate(source.shape)]
    if not all(isinstance(d, int) and d > 0 for d in shape):
        raise RuntimeError(f"Unsupported dynamic input shape: {shape}")
    tensor = np.fromfile(args.input, dtype=np.float32).reshape(shape)
    reference = np.asarray(session.run(None, {source.name: tensor})[0], dtype=np.float32)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    reference.tofile(args.report.with_suffix(".reference.bin"))
    records = []
    for path in args.outputs:
        actual = np.fromfile(path, dtype=np.float32).reshape(reference.shape)
        finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
        row = {"file": str(path), "finite": finite}
        if finite:
            difference = actual.astype(np.float64) - reference.astype(np.float64)
            absolute = np.abs(difference)
            row.update(mae=float(absolute.mean()), rmse=float(np.sqrt(np.mean(difference ** 2))),
                       max_abs_error=float(absolute.max()), p99_abs_error=float(np.quantile(absolute, 0.99)),
                       minimum=float(actual.min()), maximum=float(actual.max()))
            row["pass"] = row["mae"] <= args.max_mae
        else:
            row["pass"] = False
        records.append(row)
    report = {"onnx_model": str(args.onnx_model), "input": str(args.input), "shape": list(reference.shape),
              "input_minimum": float(tensor.min()), "input_maximum": float(tensor.max()),
              "reference_minimum": float(reference.min()), "reference_maximum": float(reference.max()),
              "max_mae": args.max_mae, "outputs": records, "pass": all(row["pass"] for row in records)}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
