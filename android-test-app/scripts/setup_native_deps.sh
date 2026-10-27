#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_DIR="$ROOT_DIR/app/src/main/cpp/onnxruntime"
JNI_DIR="$ROOT_DIR/app/src/main/jniLibs/arm64-v8a"
TEMP_DIR="${TMPDIR:-/tmp}/photo-enhancer-native-deps"
ORT_VERSION="1.18.0"
ORT_AAR="$TEMP_DIR/onnxruntime-android-$ORT_VERSION.aar"

mkdir -p "$CPP_DIR/include" "$JNI_DIR" "$TEMP_DIR"

if [[ ! -f "$ORT_AAR" ]]; then
  curl -L --fail --output "$ORT_AAR" \
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar"
fi

unzip -jo "$ORT_AAR" "jni/arm64-v8a/libonnxruntime.so" -d "$JNI_DIR"

BASE_URL="https://raw.githubusercontent.com/microsoft/onnxruntime/v$ORT_VERSION/include/onnxruntime/core/session"
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h; do
  curl -L --fail --output "$CPP_DIR/include/$header" "$BASE_URL/$header"
done

curl -L --fail --output "$CPP_DIR/include/nnapi_provider_factory.h" \
  "https://raw.githubusercontent.com/microsoft/onnxruntime/v$ORT_VERSION/include/onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"

echo "ONNX Runtime $ORT_VERSION prepared for arm64-v8a."
