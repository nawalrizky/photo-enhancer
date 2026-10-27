#!/usr/bin/env python3
"""Create an 8-bit weight-quantized GFPGAN model."""

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "models/GFPGANv1.4.onnx")
    parser.add_argument("--output", type=Path, default=root / "models/GFPGANv1.4_int8.onnx")
    parser.add_argument(
        "--weight-type",
        choices=("quint8", "qint8"),
        default="quint8",
        help="Use quint8 for ARM CPU compatibility; qint8 ConvInteger is not universally supported.",
    )
    return parser.parse_args()


def mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input model not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    weight_type = QuantType.QUInt8 if args.weight_type == "quint8" else QuantType.QInt8
    print(f"Quantizing {args.input} with {args.weight_type} weights...")
    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=weight_type,
    )

    model = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(model)
    operators = Counter(node.op_type for node in model.graph.node)
    input_size = mib(args.input)
    output_size = mib(args.output)
    reduction = (1 - output_size / input_size) * 100
    print(f"Original:  {input_size:.1f} MiB")
    print(f"Quantized: {output_size:.1f} MiB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Quantized operators: ConvInteger={operators['ConvInteger']}, MatMulInteger={operators['MatMulInteger']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
