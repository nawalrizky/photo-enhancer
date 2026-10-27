#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/models"
MODEL_FILE="$MODEL_DIR/GFPGANv1.4.onnx"
MODEL_URL="https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GFPGANv1.4.onnx"
VARIANT="${1:-fp32}"

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_FILE" ]]; then
  curl -L --fail --continue-at - --output "$MODEL_FILE.part" "$MODEL_URL"
  mv "$MODEL_FILE.part" "$MODEL_FILE"
  echo "Model downloaded: $MODEL_FILE"
fi

case "$VARIANT" in
  fp32) ;;
  int8) python3 "$ROOT_DIR/scripts/quantize_model.py" ;;
  *) echo "Usage: $0 [fp32|int8]" >&2; exit 2 ;;
esac
