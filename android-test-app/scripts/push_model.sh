#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-fp16}"
case "$VARIANT" in
  int8) MODEL_NAME="GFPGANv1.4_int8.onnx" ;;
  fp16) MODEL_NAME="GFPGANv1.4_fp16.onnx" ;;
  fp32) MODEL_NAME="GFPGANv1.4.onnx" ;;
  *) echo "Usage: $0 [fp16|int8|fp32]" >&2; exit 2 ;;
esac
MODEL_FILE="$ROOT_DIR/models/$MODEL_NAME"
PACKAGE="com.enhancer.test"
TEMP_MODEL="/data/local/tmp/$MODEL_NAME"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Model missing: $MODEL_FILE" >&2
  exit 1
fi

adb push "$MODEL_FILE" "$TEMP_MODEL"
adb shell run-as "$PACKAGE" mkdir -p files/models
adb shell run-as "$PACKAGE" cp "$TEMP_MODEL" "files/models/$MODEL_NAME"
adb shell rm -f "$TEMP_MODEL"
adb shell am force-stop "$PACKAGE"
adb shell monkey -p "$PACKAGE" 1
