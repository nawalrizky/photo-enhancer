# AI Filter Enhancer SDK 1.0.0

Offline Android library for enhancing one detected face and pasting the enhanced face back into a full image.

## Delivery Contents

- `ai-filter-enhancer-1.0.0.aar`: Kotlin/JNI API and all required `arm64-v8a` native libraries
- `GFPGANv1.4_fp16.onnx`: FP16 GFPGAN model with float32 input/output
- `INTEGRATION.md`: API contract and integration example
- `consumer-rules.pro`: R8 rules, also embedded in the AAR
- `THIRD_PARTY_NOTICES.md`: dependency and redistribution notes
- `SHA256SUMS`: integrity hashes for the AAR and model

## Runtime Contract

- Android API 28 or newer
- `arm64-v8a` only
- Fully offline after the model is installed on the device
- No Android permissions are declared by the AAR
- Input: oriented `ARGB_8888` full-image bitmap and five facial landmarks
- Aspect ratio: unrestricted
- Maximum image edge: 2048 px
- Output: new `ARGB_8888` bitmap with the same width, height, and aspect ratio as the input
- One face and one enhancement at a time per SDK instance

## Validated Device Result

Validated on Android 14, RK3576, 8 GB RAM:

- AAR API full enhancement: 18.031 seconds on a 256x256 test image
- Minified/R8 host integration: 18.182 seconds on the same image
- FP16 versus FP32 close-face MAE: 0.000272
- Minified release reference HWM: 708,420 KB
- Native crashes: none
- Network connection during runtime smoke test: none

The model is deliberately separate from the AAR. The host app decides whether to bundle it as an install-time asset or provision it through its existing model delivery system.
