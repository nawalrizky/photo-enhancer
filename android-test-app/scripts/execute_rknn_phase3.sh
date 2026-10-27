#!/usr/bin/env bash
# Run from a staged Linux project containing models/, scripts/, and .venv/.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
STATUS_FILE="$PROJECT_DIR/logs/phase3-$ATTEMPT.status"
mkdir -p logs
trap 'result=$?; if (( result != 0 )); then printf "FAILED exit=%s utc=%s\n" "$result" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"; fi' EXIT

wait_for_model() {
    local filename="$1" expected_bytes="$2" expected_sha="$3"
    printf 'Waiting for complete model: %s\n' "$filename"
    printf 'WAITING %s\n' "$filename" >> "$STATUS_FILE"
    while [[ ! -f "models/$filename" ]] || [[ "$(stat -c %s "models/$filename")" != "$expected_bytes" ]]; do
        sleep 20
    done
    printf '%s  models/%s\n' "$expected_sha" "$filename" | sha256sum --check
}

printf 'START utc=%s\n' "$(date -u +%FT%TZ)" > "$STATUS_FILE"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "logs/environment-$ATTEMPT.lock.txt"

wait_for_model GFPGANv1.4.onnx 340256690 6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73
printf 'CONVERTING fp32\n' >> "$STATUS_FILE"
"$PYTHON" scripts/run_rknn_conversion.py \
    --input models/GFPGANv1.4.onnx \
    --output models/GFPGANv1.4_rk3576.rknn \
    --log-dir "logs/fp32-$ATTEMPT"

wait_for_model GFPGANv1.4_fp16.onnx 170260352 d40cedbd48b2f5829ca6e109efa5b55ad7178a40c8f68ae35f1cb9d540cf0d2f
printf 'REPAIRING fp16\n' >> "$STATUS_FILE"
"$PYTHON" scripts/repair_fp16_resize_roi.py \
    --input models/GFPGANv1.4_fp16.onnx \
    --output models/GFPGANv1.4_fp16_repaired.onnx
printf 'CONVERTING fp16\n' >> "$STATUS_FILE"
"$PYTHON" scripts/run_rknn_conversion.py \
    --input models/GFPGANv1.4_fp16_repaired.onnx \
    --output models/GFPGANv1.4_fp16_rk3576.rknn \
    --log-dir "logs/fp16-$ATTEMPT"
printf 'MILESTONE_1_EXPORTED utc=%s\n' "$(date -u +%FT%TZ)" >> "$STATUS_FILE"
printf 'RKNN models exported. Tensor, quality, runtime, and latency gates are still pending.\n'
