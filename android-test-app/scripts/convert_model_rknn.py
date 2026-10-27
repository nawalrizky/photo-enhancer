#!/usr/bin/env python3
"""Attempt non-quantized GFPGAN conversion for the RK3576 NPU on Linux."""

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "models/GFPGANv1.4_fp16.onnx")
    parser.add_argument("--output", type=Path, default=root / "models/GFPGANv1.4_fp16_rk3576.rknn")
    parser.add_argument("--verbose", action="store_true", help="Show RKNN graph optimizer details")
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3,
                        help="RKNN compiler optimization level (default: 3)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Model not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from rknn.api import RKNN
    except ImportError as error:
        parser.error(f"RKNN-Toolkit2 is required on a Linux host: {error}")

    rknn = RKNN(verbose=args.verbose)
    try:
        if rknn.config(target_platform="rk3576", optimization_level=args.optimization_level) != 0:
            raise RuntimeError("RKNN configuration failed")
        if rknn.load_onnx(model=str(args.input)) != 0:
            raise RuntimeError("GFPGAN ONNX graph could not be loaded by RKNN")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("GFPGAN graph could not be compiled for RK3576")
        if rknn.export_rknn(str(args.output)) != 0:
            raise RuntimeError("RKNN model export failed")
    finally:
        rknn.release()

    print(f"RKNN model: {args.output}")
    print(f"Size: {args.output.stat().st_size / (1024 * 1024):.1f} MiB")
    print("Conversion alone does not establish accuracy or sub-5-second device latency.")


if __name__ == "__main__":
    main()
