#!/usr/bin/env python3
"""Validate model structure, runtime compatibility, output error, and latency."""

import argparse
import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=root / "models/GFPGANv1.4.onnx")
    parser.add_argument("--quantized", type=Path, default=root / "models/GFPGANv1.4_int8.onnx")
    parser.add_argument("--image", type=Path, help="Optional aligned or tightly cropped face image.")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Crop the input image before resizing it to 512x512.",
    )
    parser.add_argument("--output-dir", type=Path, help="Write FP32 and INT8 output previews here.")
    parser.add_argument("--max-mae", type=float, default=0.05)
    return parser.parse_args()


def make_input(image_path: Optional[Path], crop: Optional[list[int]]) -> np.ndarray:
    if image_path is None:
        rng = np.random.default_rng(7)
        return np.clip(rng.normal(0, 0.35, (1, 3, 512, 512)), -1, 1).astype(np.float32)
    image = Image.open(image_path).convert("RGB")
    if crop:
        image = image.crop(tuple(crop))
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(values, (2, 0, 1))[None]


def run_model(path: Path, values: np.ndarray) -> tuple[np.ndarray, float]:
    onnx.checker.check_model(onnx.load(str(path), load_external_data=False))
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.shape != [1, 3, 512, 512] or model_output.shape != [1, 3, 512, 512]:
        raise RuntimeError(f"Unexpected model tensors: input={model_input.shape}, output={model_output.shape}")
    output = session.run(None, {model_input.name: values})[0]
    elapsed = time.perf_counter() - started
    if output.dtype != np.float32 or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid output: dtype={output.dtype}, finite={np.isfinite(output).all()}")
    del session
    gc.collect()
    return output, elapsed


def save_preview(output: np.ndarray, path: Path) -> None:
    pixels = np.clip((np.transpose(output[0], (1, 2, 0)) + 1) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main() -> None:
    args = parse_args()
    values = make_input(args.image, args.crop)
    original, original_time = run_model(args.original, values)
    quantized, quantized_time = run_model(args.quantized, values)

    difference = np.abs(original - quantized)
    mae = float(difference.mean())
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    print(f"FP32 runtime: {original_time:.3f}s")
    print(f"Candidate runtime: {quantized_time:.3f}s")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"P99 absolute error: {np.percentile(difference, 99):.6f}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_preview(original, args.output_dir / "gfpgan_fp32.png")
        save_preview(quantized, args.output_dir / "gfpgan_int8.png")

    if mae > args.max_mae:
        raise SystemExit(f"FAILED: MAE {mae:.6f} exceeds threshold {args.max_mae:.6f}")
    print("PASSED")


if __name__ == "__main__":
    main()
