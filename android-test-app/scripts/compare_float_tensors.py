#!/usr/bin/env python3
"""Compare little-endian float32 tensors without third-party dependencies."""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def read(path):
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(f'Invalid float32 file size: {path}')
    values = array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'Nonfinite tensor: {path}')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--max-mae', type=float, default=0.02)
    args = parser.parse_args()
    reference, actual = read(args.reference), read(args.actual)
    if len(reference) != len(actual):
        raise ValueError('Tensor element counts differ')
    difference = sorted(abs(a-b) for a, b in zip(actual, reference))
    mae = math.fsum(difference) / len(difference)
    report = dict(reference=str(args.reference), actual=str(args.actual),
                  elements=len(actual), finite=True, mae=mae,
                  rmse=math.sqrt(math.fsum(v*v for v in difference)/len(difference)),
                  max_abs_error=difference[-1],
                  p99_abs_error=difference[int((len(difference)-1)*0.99)],
                  max_mae=args.max_mae)
    report['pass'] = mae <= args.max_mae
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
