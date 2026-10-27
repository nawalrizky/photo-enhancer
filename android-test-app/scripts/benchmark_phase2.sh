#!/usr/bin/env bash
set -euo pipefail

PACKAGE="com.enhancer.test"
OUTPUT_DIR="${1:-phase2-benchmark}"
WAIT_SECONDS="${WAIT_SECONDS:-35}"

mkdir -p "$OUTPUT_DIR"
adb shell am start -n "$PACKAGE/.MainActivity" >/dev/null
echo "Prepare a detected source portrait in the app, then press Enter."
read -r

adb logcat -c
# Coordinates target Enhance Full on the 1920x1080 RK3576 reference display.
adb shell input tap 705 945
sleep "$WAIT_SECONDS"
adb logcat -d -s "EnhancerMem:I" "PhotoEnhancerNative:*" > "$OUTPUT_DIR/native-memory.log"
adb shell dumpsys meminfo "$PACKAGE" > "$OUTPUT_DIR/meminfo.txt"
adb exec-out screencap -p > "$OUTPUT_DIR/result.png"

echo "Benchmark written to $OUTPUT_DIR"
