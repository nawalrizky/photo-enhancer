# Photo Enhancer Android Test App

Offline Android test harness for `libphoto_enhancer.so`. The app accepts a full portrait, detects one face and five landmarks with bundled ML Kit, aligns the face to 512x512, runs GFPGAN through ONNX Runtime, and displays either the aligned crop or a paste-back result.

## Requirements

- Android SDK 34, NDK `27.2.12479018`, CMake `3.22.1`
- JDK 17 or newer
- An `arm64-v8a` Android device connected through ADB
- Approximately 400 MB free for FP32, or 200 MB for the recommended FP16 model

## Setup

```bash
./scripts/setup_native_deps.sh
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/download_model.sh
python3 -m venv .venv-quantize
source .venv-quantize/bin/activate
pip install -r scripts/requirements-quantization.txt
python scripts/convert_model_fp16.py
./scripts/push_model.sh fp16
```

The model is intentionally not embedded in the APK. It is pushed into the app's private storage using `run-as`:

```text
/data/user/0/com.enhancer.test/files/models/GFPGANv1.4_fp16.onnx
```

This keeps APK rebuilds fast and still provides fully offline inference after setup.

## Recommended FP16 Model

The app selects FP16 first, then FP32, then the experimental INT8 model. FP16 keeps float32 model input/output tensors while converting internal weights and tensors. It reduces the model from 324.5 MiB to 162.4 MiB and passed the close-face numerical gate with MAE `0.000272` against FP32.

The accepted RK3576 CPU configuration uses four intra-op threads, sequential execution, disabled CPU arena and memory pattern, and direct 8-bit paste-back blending. Leave **Try NNAPI** off: the tested driver delegated only a small graph subset, increased latency to `27.594s`, and increased RSS HWM to `887,796 KB`.

## Experimental 8-bit Model

Create the quantization environment and model with:

```bash
python3 -m venv .venv-quantize
source .venv-quantize/bin/activate
pip install -r scripts/requirements-quantization.txt
python scripts/quantize_model.py
python scripts/validate_quantized.py
./scripts/push_model.sh int8
```

The app uses `GFPGANv1.4_int8.onnx` only when FP16 and FP32 are absent, and identifies the active variant in the model badge. Use `QUInt8` weights, the script default, for ARM CPU compatibility. Signed `QInt8` generated `ConvInteger` nodes that were unsupported by the tested ONNX Runtime CPU build.

The current 8-bit candidate is deployable but experimental. It is 150.9 MiB rather than the initially projected 85 MiB. A close full-face capture measured MAE `0.054501`, RMSE `0.066192`, and P99 absolute error `0.167266`, so it still exceeds the `0.05` acceptance threshold. The INT8 preview also showed visible grain and tone differences. Keep FP16 available until a better quantization method passes representative visual review.

Build the R8/resource-shrunk APK with:

```bash
./gradlew :app:assembleRelease
```

## Test Flow

1. Open **Enhancer Lab**.
2. Confirm the model badge says `READY`.
3. Select a gallery image or use **Take Photo** for a full-resolution camera capture. Images larger than 2048 px are downscaled before processing.
4. Wait for face detection.
5. Use **Enhance Crop** for the aligned 512x512 face or **Enhance Full** for paste-back.
6. Leave **Try NNAPI** off for the first correctness test. GFPGAN operators may fall back or fail on some NNAPI drivers.
7. Use **Save Result** to write the output to `Pictures/EnhancerLab`.

For repeatable device capture, run `scripts/benchmark_phase2.sh`. The script expects a source portrait with completed face detection in the app and uses the 1920x1080 RK3576 reference coordinates; adjust the tap coordinates for another display layout.

## Current Scope

- One face per operation; the largest detected face is selected.
- ML Kit face detection and all enhancement inference are offline.
- GFPGAN only; RealESRGAN background upscaling is not included.
- The paste-back mask is feathered but is not equivalent to GFPGAN face-parsing masks.
- The native wrapper expects a float32 ONNX input/output tensor shaped `[1,3,512,512]`.

## Observed Device Result

Tested over ADB on an Android 14 RK3576 `arm64-v8a` device with 8 GB RAM:

- Debug APK: approximately 33.5 MB, excluding the model
- R8/resource-shrunk release APK: 23,276,840 bytes test-signed
- GFPGAN FP32 model: 340,256,690 bytes (324.5 MiB)
- Recommended FP16 model: 170,260,352 bytes (162.4 MiB, 50.0% reduction)
- Experimental QUInt8 model: 158,272,369 bytes (150.9 MiB, 53.5% reduction)
- Final FP16 full-image paste-back on a 256x256 input: 18.053 seconds on CPU
- Final FP16 native RSS HWM: 723,836 KB; sampled peak PSS: 540,122 KB
- Final FP16 post-run RSS: 417,644 KB; post-run PSS: 293,050 KB
- Minified release verification: 18.506 seconds; native RSS HWM 708,420 KB
- QUInt8 crop: 17.432 seconds; sampled RSS during inference: 828,636 KB
- QUInt8 full paste-back: 15.161 seconds; sampled RSS during inference: 1,237,084 KB

See `PHASE1_RESULTS.md` and `PHASE2_RESULTS.md` for complete measured outcomes.

The ongoing sub-5-second investigation and repeatable latency benchmark are documented in `PHASE3_RESULTS.md`. The current source code uses eight intra-op threads for a measured warm-call latency of about 13.7 seconds on a separate RK3576 unit; the delivered 1.0.0 AAR/ZIP has not been replaced.
