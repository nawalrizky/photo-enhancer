#!/usr/bin/env python3
"""Convert GFPGAN weights and internal tensors to FP16 with FP32 model I/O."""

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "models/GFPGANv1.4.onnx")
    parser.add_argument("--output", type=Path, default=root / "models/GFPGANv1.4_fp16.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input model not found: {args.input}")
    model = onnx.load(str(args.input))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(args.output))

    original_size = args.input.stat().st_size
    converted_size = args.output.stat().st_size
    print(f"Original: {original_size / 1024 / 1024:.1f} MiB")
    print(f"FP16: {converted_size / 1024 / 1024:.1f} MiB")
    print(f"Reduction: {(1 - converted_size / original_size) * 100:.1f}%")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
