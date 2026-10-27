#!/usr/bin/env bash
set -euo pipefail

SERIAL="${ADB_SERIAL:-192.168.2.42:37361}"
RUNS="${BENCHMARK_RUNS:-3}"
IMAGE="${BENCHMARK_IMAGE:-$(cd "$(dirname "$0")/../.." && pwd)/samples/obama.jpg}"
PACKAGE="com.enhancer.test"
DEVICE_IMAGE="/data/user/0/$PACKAGE/files/benchmark_input.jpg"

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]] || (( RUNS > 10 )); then
    echo "BENCHMARK_RUNS must be 1 to 10" >&2
    exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "Image not found: $IMAGE" >&2
    exit 2
fi

adb -s "$SERIAL" shell run-as "$PACKAGE" test -f files/models/GFPGANv1.4_fp16.onnx
adb -s "$SERIAL" push "$IMAGE" /data/local/tmp/benchmark_input.jpg
adb -s "$SERIAL" shell run-as "$PACKAGE" cp /data/local/tmp/benchmark_input.jpg files/benchmark_input.jpg
adb -s "$SERIAL" shell am force-stop "$PACKAGE"
adb -s "$SERIAL" logcat -c
adb -s "$SERIAL" shell am start -n "$PACKAGE/.MainActivity" \
    --es benchmark_image "$DEVICE_IMAGE" --ei benchmark_runs "$RUNS"

deadline=$((SECONDS + RUNS * 45 + 20))
while (( SECONDS < deadline )); do
    logs="$(adb -s "$SERIAL" logcat -d -s EnhancerBench:I AiFilterEnhancer:I AndroidRuntime:E)"
    completed="$(printf '%s\n' "$logs" | rg -c 'EnhancerBench.*total_ms=' || true)"
    if (( ${completed:-0} >= RUNS )); then
        printf '%s\n' "$logs"
        exit 0
    fi
    sleep 2
done

printf '%s\n' "$logs"
echo "Benchmark timed out after $RUNS requested run(s)" >&2
exit 1
