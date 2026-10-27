# Experimental RKNN backend

This module is used by the Android test app. It does not replace the delivered
1.0.0 AAR or ZIP. See [execution results](EXECUTION_RESULTS.md) for gates and evidence.

## Candidate model

The initial FP32 and repaired-FP16 RKNN exports lose RGB skip bias contributions
and fail numerical validation. Use the candidate built from
`GFPGANv1.4_rgb_bias_consolidated.onnx` after validating the bias rewrite on CPU:

```sh
python scripts/consolidate_gfpgan_rgb_bias.py \
  --input models/GFPGANv1.4.onnx \
  --output models/GFPGANv1.4_rgb_bias_consolidated.onnx
python scripts/convert_model_rknn.py \
  --input models/GFPGANv1.4_rgb_bias_consolidated.onnx \
  --output models/GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn
```

These commands run on the Linux conversion host, with RKNN Toolkit2 2.3.2.
Original failed candidate copies on the Mac are compressed losslessly as `.rknn.gz`;
uncompressed copies remain on the conversion host and test device.

## Runtime and build

`vendor/librknnrt.so` is the Android arm64 runtime from official RKNN Toolkit2
v2.3.2, commit `42aa1d426c0a9e0869b6374edba009f7208a1926`.
Runtime SHA256: `07a8398be5caed21a1998ffa313e0425bb9e76b154a8dbbc667d71722c009351`.
Place its matching `rknn_api.h` in the same directory. Large binaries are ignored.

Build from `android-test-app` with `./gradlew :app:assembleDebug`. The experimental
CMake target links the runtime and OpenCV 4.10.0. The test app defaults to RKNN;
`benchmark_backend=cpu` explicitly selects the CPU reference.
The experimental engine retains the CPU engine's five-point affine alignment,
RGB normalization, output conversion, inverse warp and blending mask. Its input
adapter packs native FP16 NHWC and its output is logical float32 NCHW.

Push the candidate to the test app's `files/models/` directory using `adb shell
run-as com.enhancer.test`. Select RKNN through the launch intent extra
`benchmark_backend=rknn`, or open the test app normally. The model filename expected by the test app is
`GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn`.

## Reproduce measurements

From the repository root:

```sh
python3 android-test-app/scripts/benchmark_rknn_pipeline.py --case normal --backend cpu
python3 android-test-app/scripts/benchmark_rknn_pipeline.py --case normal --backend rknn
```

Other cases: `close-up`, `elderly`, `low-light`; `--serial` overrides the device.
CPU uses one first call for the reference; RKNN uses a first call and three warm
calls. Every call decodes the image and detects landmarks again. PNG export and
ADB transfer occur after the processing timer; user image selection and app
startup are not included. Output files are retained in `artifacts/pipeline/`.

`scripts/compare_float_tensors.py` compares little-endian float32 raw tensors.
`scripts/compare_png_outputs.py` compares RGB pixels in noninterlaced 8-bit RGB/RGBA
PNG exports and can create a side-by-side image (CPU left, RKNN right).

The standalone `rknn_probe.cpp` isolates runtime inference from the photo pipeline.
Its profiling flag must be disabled for latency measurements. Profiling results
and inference-only timings must not be presented as full application latency.
