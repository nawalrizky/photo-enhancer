# Phase 1 Results

## Status

Phase 1 tooling, release shrinking, and RK3576 compatibility testing are complete. The 8-bit model is retained as an experimental candidate because it does not meet the original size or numerical quality targets.

## Model Quantization

| Metric | FP32 | QUInt8 | Change |
| --- | ---: | ---: | ---: |
| Model size | 340,256,690 bytes | 158,272,369 bytes | -53.5% |
| Model size (MiB) | 324.5 | 150.9 | -53.5% |
| Close full-face output MAE | - | 0.054501 | Fails 0.05 threshold |
| Close full-face output RMSE | - | 0.066192 | - |
| Close full-face P99 absolute error | - | 0.167266 | - |
| Deterministic synthetic MAE | - | 0.085452 | Fails 0.05 threshold |

Dynamic quantization converted 95 `Conv` nodes to `ConvInteger` and 24 `Gemm` nodes to `MatMulInteger`. The proposed signed `QInt8` format could not initialize on the tested ARM ONNX Runtime CPU build. Unsigned `QUInt8` initialized and ran on both Apple ARM and RK3576.

The projected 85 MB result was not achievable with ONNX Runtime dynamic quantization because substantial constants and unsupported/non-quantized graph data remain in floating-point form.

Quality was rechecked with the latest close portrait rather than relying on the earlier distant subject. A `900x900` full-face crop was resized to the model's 512x512 input and evaluated through the same FP32 and QUInt8 graphs. The MAE improved to `0.054501` but remained above the required `0.05`; side-by-side outputs showed additional grain, edge texture, and tone changes in the QUInt8 result. The minimum quality criterion therefore remains unmet.

## Android Validation

Device: Android 14, RK3576, `arm64-v8a`, 8 GB RAM.

| Test | Result |
| --- | ---: |
| INT8 model badge | `MODEL: INT8 / 150 MB / READY` |
| Crop enhancement | 17.432 seconds, 512x512 |
| Full enhancement | 15.161 seconds, 2048x1536 |
| Minified release full enhancement | 14.506 seconds, 2048x1536 |
| Crop sampled RSS | 828,636 KB |
| Full sampled RSS | 1,237,084 KB |
| Native/runtime crashes | None |

Crop speed is effectively unchanged from the 17.7-second FP32 baseline. Full enhancement is also within normal run-to-run variance. The lower crop RSS is useful, but full-image memory remains above 1 GB because the model, source bitmap, result bitmap, and paste-back buffers coexist.

## APK Shrinking

| Artifact | Size |
| --- | ---: |
| Debug APK | 33,490,158 bytes |
| R8 release APK, unsigned | 23,241,133 bytes |
| R8 release APK, test-signed | 23,276,840 bytes |

The signed release is 30.5% smaller than the debug APK. The model is stored outside the APK, so model quantization does not affect APK size. Most remaining APK bytes are ONNX Runtime, OpenCV, and ML Kit native libraries and bundled detector models.

## Decision

- R8 and resource shrinking are accepted.
- QUInt8 runtime compatibility is accepted.
- QUInt8 quality and model-size success criteria are not met.
- Keep `GFPGANv1.4.onnx` for rollback and conduct representative side-by-side portrait review before making INT8 the production default.
- Do not claim a 105 MB total footprint. The measured signed APK plus QUInt8 model is 181,549,209 bytes, excluding installed extraction and app data.
