# Porting This Photo Enhancer to Native Android ARM (.so) — Feasibility Report

*Generated: 2026-08-12 | Deep research via Exa AI (2 rounds, 20+ queries, 30+ sources) + codebase analysis | Confidence: High (resource availability), Medium (perf numbers vary by device)*

---

## Executive Summary

This project's enhancement logic is a **Python/PyTorch pipeline** (`GFPGAN` face restoration + `RealESRGAN` background upscaling, orchestrated by `GFPGANer` from `facexlib`). Shipping it as a native C++ `.so` for Android ARM means: **the pipeline logic itself is trivial to port (it's ~100 lines), but the model runtime and model conversion are the real work.**

**Verdict: Real-ESRGAN-only `.so` = EASY (1–2 weeks). Full GFPGAN face-enhancement `.so` = MEDIUM-HARD (3–6+ weeks), with significant performance trade-offs.** The single biggest blocker is not C++ code — it is converting GFPGAN's PyTorch graph (which contains runtime-generated StyleGAN noise and control flow) into a mobile-friendly format, and then accepting that inference takes **seconds per photo**, not milliseconds.

Key facts driving the assessment:

| Fact | Source |
|---|---|
| Ready-made, official NCNN port of Real-ESRGAN exists with Android builds | [xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) |
| Multiple complete open-source Android apps already ship Real-ESRGAN via `.so` (JNI) | [tumuyan/RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android), [HF-CYGG/LumaSR](https://github.com/HF-CYGG/LumaSR), [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile) |
| GFPGAN NCNN ports exist but are **WIP/incomplete**, none production-ready on Android | [onuralpszr/GFPGAN-ncnn-vulkan](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan) (open issue: doesn't work on Android), [FeiGeChuanShu/GFPGAN-ncnn](https://github.com/FeiGeChuanShu/GFPGAN-ncnn) |
| GFPGAN → ONNX conversion works but needs hacks (pre-generated noise, linearized forward pass, fixed 512×512 input) | [xuanandsix/GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo) |
| Real-ESRGAN on a phone takes **~1.9 s per 224×224 image** (NCNN) / **~4.2 s** (ExecuTorch) — the codebase's own warning ("slow on CPU") carries over to mobile | [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile), `enhancer/enhancer.py:13-16` |
| GFPGAN model is **~330 MB fp32**; Real-ESRGAN x4plus is **~64 MB fp32** (16.7 MB int8) | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/real_esrgan_x4plus), [TencentARC/GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases) |
| ncnn Android libs: ~13–26 MB (prebuilt zip, all ABIs); MNN core `.so` ~800 KB; ONNX Runtime custom-build `.so` ~4 MB; PyTorch Mobile `libpytorch_jni` ~11 MB | [Tencent/ncnn releases](https://github.com/Tencent/ncnn/releases), [alibaba/MNN](https://github.com/alibaba/MNN), [onnxruntime.ai mobile docs](https://onnxruntime.ai/docs/tutorials/mobile/), [pytorch/pytorch#59865](https://github.com/pytorch/pytorch/issues/59865) |

---

## 1. What the Codebase Actually Does

### 1.1 Pipeline (from `enhancer/enhancer.py`)

```
input image (RGB)
  │
  ├─ cv2 RGB→BGR
  ├─ check_image_dimensions (>2048px → skip enhancement entirely)
  │
  └─ GFPGANer.enhance(...)          ← the whole heavy lifting (facexlib)
       ├─ [1] RetinaFace detection   (retinaface_resnet50, from facexlib)
       ├─ [2] 5-point landmark alignment + warp crop → 512×512
       ├─ [3] GFPGANv1.4 inference   (StyleGAN2-based prior network, channel_multiplier=2, arch='clean')
       ├─ [4] RealESRGAN background upscale (RRDBNet 23 blocks, x2 or x4, tile=400)
       └─ [5] inverse affine + paste restored faces back
```

### 1.2 What must be shipped in a `.so`

| Component | Python today | Native C++ equivalent | Difficulty |
|---|---|---|---|
| Image I/O + color space | OpenCV | OpenCV for Android (prebuilt) | Trivial |
| Face detection + landmarks | facexlib RetinaFace | ncnn RetinaFace (models + code already in ncnn examples) | Easy (ready-made) |
| Affine align / warp / paste-back | cv2.warpAffine | OpenCV or Eigen — ~50 lines | Trivial |
| GFPGAN v1.4 inference | PyTorch `GFPGANv1Clean` | **Needs model conversion** — the hard part | **Hard** |
| RealESRGAN inference | PyTorch `RRDBNet` | ncnn `.param/.bin` models already public | Easy |
| JNI bridge | — | CMake + JNI wrapper, `System.loadLibrary("enhancer")` | Easy (well-documented pattern) |

**Important:** ~90% of the pipeline (detection, warp, paste, RealESRGAN) already has battle-tested open-source C++/Android code. Only GFPGAN inference itself needs genuine engineering work.

---

## 2. What "Creating a .so on Android ARM" Actually Involves

1. Build native libs for ABI `arm64-v8a` (optionally `armeabi-v7a`) with Android NDK + CMake (`ANDROID_STL=c++_shared`).
2. Link an inference engine (ncnn / ONNX Runtime / libtorch / MNN / ExecuTorch) — you do **not** write the neural net ops by hand.
3. Convert the PyTorch `.pth` weights into the engine's format (the real work).
4. Expose a `extern "C"` JNI function (`Java_com_..._Enhancer_enhance`) so Kotlin/Java can call it.
5. Models are usually shipped as **assets** (`.param/.bin` or `.onnx`/`.pte` files) next to the `.so`, not baked into it.

Reference implementation of exactly this pattern (JNI + CMake + ncnn + OpenCV): [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile) and [ArchieMeng/ncnn-android-waifu2x-demo](https://github.com/ArchieMeng/ncnn-android-waifu2x-demo) (cited by CodeFormer maintainers as the working approach: [sczhou/CodeFormer#401](https://github.com/sczhou/CodeFormer/issues/401)).

---

## 3. Inference Engine Options (Compared)

### 3.1 NCNN (Tencent) — **Recommended**

- Purpose-built for mobile ARM: NEON-optimized CPU kernels, Vulkan GPU backend, zero third-party runtime deps; used in WeChat/QQ ([Tencent/ncnn](https://github.com/Tencent/ncnn)).
- Prebuilt Android libs: `ncnn-20260526-android-vulkan.zip` ~26 MB, plain ~14.7 MB (all 4 ABIs) ([releases](https://github.com/Tencent/ncnn/releases/tag/20250503)).
- Conversion: `pth → onnx → pnnx → .param/.bin`. GFPGAN & CodeFormer already converted and hosted: [Baiyuetribe/ncnn-models](https://github.com/Baiyuetribe/ncnn-models) (GFPGAN 214 MB, CodeFormer 212.5 MB — converted successfully).
- **Caveat:** fp16 path had landmark-precision bugs on ARM for RetinaFace (fixed by disabling `use_fp16_storage/arithmetic`); budget time for such quirks ([ncnn#2885](https://github.com/Tencent/ncnn/issues/2885)).
- Production precedent: the codebase's own ecosystem (Real-ESRGAN) has an official ncnn port: [xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).

### 3.2 PyTorch Mobile / libtorch (JIT)

- Path of least conversion resistance: `torch.jit.trace/script` → `.pt` / lite-interpreter `.ptl`, load with libtorch C++ API in your `.so` ([pytorch/android README](https://github.com/pytorch/pytorch/blob/main/android/README.md)).
- **Sizes hurt:** `libtorch_cpu.so` ~75–89 MB per ABI without op-selection; `libpytorch_jni` ~11 MB; custom builds with `SELECTED_OP_LIST` reduce but not dramatically for big models ([pytorch#59865](https://github.com/pytorch/pytorch/issues/59865), [discuss.pytorch.org](https://discuss.pytorch.org/t/is-it-possible-to-build-a-minimal-libpytorch-so-for-c-inference-on-mobile-device/97798)).
- Slower on ARM than ncnn in most benchmarks; PyTorch Mobile is being superseded by ExecuTorch.

### 3.3 ExecuTorch (PyTorch 2.x)

- AOT-compiles via `torch.export` → `.pte`, XNNPACK/Vulkan/QNN backends, `executorch.aar` on Maven Central, stable since v1.0 (Oct 2025) ([pytorch/executorch](https://github.com/pytorch/executorch), [blog](https://pytorch.org/blog/introducing-executorch-1-0/)).
- **Caveat:** in the one public Real-ESRGAN-on-Android benchmark it was ~2.3× slower than ncnn and showed **corrupted-output instability** on complex GAN graphs ([ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile)). GANs are its weak spot today.

### 3.4 ONNX Runtime Mobile

- Convert `pth → onnx` (GFPGAN conversion recipe exists: [xuanandsix/GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo)), run via `onnxruntime-android` AAR; NNAPI + XNNPACK EPs.
- Custom (op-minimized) build: `libonnxruntime.so` **~4 MB** vs 16 MB stock ([onnxruntime docs](https://onnxruntime.ai/docs/tutorials/mobile/)).
- Upside: model is portable across Android/iOS/desktop with one artifact. Downside: NNAPI support for GAN-style ops is uneven (PyTorch Mobile had the same problem: [pytorch#48411](https://github.com/pytorch/pytorch/issues/48411)).

### 3.5 MNN (Alibaba)

- Extremely small core (`libmnncore.so` ~800 KB), strong ARM assembly kernels + Vulkan, used in 30+ Alibaba apps ([alibaba/MNN](https://github.com/alibaba/MNN)).
- Supports TorchScript/ONNX input; GFPGAN/CodeFormer conversion reports exist in the wild but **no maintained public GFPGAN demo** — you'd do the conversion yourself.

### 3.6 Summary table

| Engine | GFPGAN conv. effort | RealESRGAN resources | `.so` size (arm64) | Android perf track record | Overall |
|---|---|---|---|---|---|
| **NCNN** | Medium (pnnx; models partially public) | **Official port + models + apps** | ~1–3 MB lib + assets | Best (NEON/Vulkan, mass production) | **Recommended** |
| ONNX Runtime | Medium (recipe exists) | Some ports | ~4–16 MB | Good (XNNPACK), NNAPI spotty | Good fallback |
| ExecuTorch | High (GAN graph instability reported) | One demo, slow | AAR-based | Promising but immature for GANs | Risky today |
| PyTorch Mobile | Low (native format) | Slow, huge lib | 11–89 MB | Weakest | Avoid |
| MNN | Medium-High (no public demo) | No public demo | ~1 MB | Good | Solid but more DIY |

---

## 4. Performance Reality Check (Critical)

- The codebase **itself** disables RealESRGAN on CPU: *"The unoptimized RealESRGAN is slow on CPU... If you really want to use it, please modify the corresponding codes"* (`enhancer/enhancer.py:13-16`). Mobile CPUs are slower than server CPUs — the warning scales down.
- Measured on Android (Real-ESRGAN x4plus, 224×224 input, mid-range phone):
  - NCNN: **~1.85 s** per image
  - ExecuTorch: **~4.2 s** per image
  - (vs desktop GPU PyTorch: 3.3 ms) ([ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile))
- Real-ESRGAN author's own warning on mobile: *"it will run your mobile device's graphics card at full capacity for several minutes... cannot be used in production [as-is]"* ([xinntao/Real-ESRGAN#86 discussion](https://github.com/xinntao/Real-ESRGAN/issues/86)).
- GFPGAN inference per face: 512×512 StyleGAN2-style decoder — expect **1–4 s per face** on ARM (CPU or Vulkan) based on public mobile experiments; on a photo with N faces multiply accordingly. For reference, the GPU path in this repo is ~15 ms/face (HyperLips timings) — desktop GPU is ~100–200× faster than mobile.
- Your repo's 2048px cap (`check_image_dimensions`) actually helps: it bounds the worst-case RealESRGAN cost, but a 2048×2048 image at x4 still means a ~8192×8192 output through a 23-block RRDBNet — budget **tens of seconds** at full resolution, and use the `tile=400` tiling strategy that the Python code already uses.

**Implication:** "Run locally on Android" is feasible for **single-photo, offline, seconds-per-photo** use cases (an "enhance this photo" button with a progress spinner). It is **not** feasible as a real-time/video pipeline with these models. If realtime is needed, the answer is a smaller model (e.g., lighter GAN, or face-only enhancement with a mobile-optimized detector), not a bigger build system.

---

## 5. Model Sizes & Quantization (APK Budget)

| Model | fp32 | fp16 (~2×) | int8 (~4×) | Notes |
|---|---|---|---|---|
| GFPGANv1.4 `.pth` | **~330 MB** | ~165 MB | ~85 MB | StyleGAN2-based decoder dominates |
| Real-ESRGAN x4plus | **~64 MB** | ~32 MB | **16.7 MB** (w8a8, verified) | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/real_esrgan_x4plus) |
| Real-ESRGAN x2plus | ~68 MB | ~34 MB | ~17 MB | |
| RetinaFace (mnet/ResNet50) | 1.7–43 MB | — | — | MobileNet-based mnet recommended for ARM |
| ncnn engine libs | — | — | — | ~1–3 MB per ABI linked into your `.so` |

Practical APK budget for the full pipeline (fp16): **~250 MB**. Face-only GFPGAN (no background upsampler): **~200 MB**. RealESRGAN-only: **~30 MB**. If APK size matters, ship models on first-run download (Play Asset Delivery / in-app download) — a common pattern in these open-source apps.

---

## 6. GFPGAN Conversion — The Real Hard Part

GFPGAN cannot be naively exported. Known blockers and solutions (from [xuanandsix/GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo) + DeepWiki analysis):

1. **Dynamic noise in StyleGAN layers** — `out.new_empty(...).normal_()` has no ONNX equivalent → must pre-generate noise tensors and pass them as static inputs.
2. **Control flow / branching in forward()** — must rewrite forward into a single linear pass.
3. **Fixed input resolution** — GFPGAN is 512×512 input; the pipeline guarantees this via detection+align (so it's fine, just non-negotiable).
4. **`sft_half=True`, channel_multiplier=2, arch='clean'** — these codebase-specific settings must be reproduced exactly in whatever conversion script you use (the codebase uses GFPGANv1.4 = clean arch).
5. **Face parsing** — the Python path uses face-parsing masks for paste-back (`use_parse=True`); a faithful port needs the parsing net (another model) or an acceptable mask-free paste (visually softer blend).

Then, for ncnn: `onnx → pnnx → .param/.bin`, or reuse already-converted community files ([Baiyuetribe/ncnn-models](https://github.com/Baiyuetribe/ncnn-models) — GFPGAN 214 MB and CodeFormer 212.5 MB already converted ✅). Existing GFPGAN ncnn repos are either naive CPU-only ([FeiGeChuanShu/GFPGAN-ncnn](https://github.com/FeiGeChuanShu/GFPGAN-ncnn)) or WIP-Vulkan with an open "doesn't work on Android" issue ([onuralpszr/GFPGAN-ncnn-vulkan#19](https://github.com/onuralpszr/gfpgan-ncnn-vulkan/issues/19)) — **do not expect to copy-paste a finished Android integration; expect to finish one yourself.**

---

## 7. Recommended Roadmap (with Effort Estimates)

Assumes 1 experienced Android/NDK engineer with C++ and some PyTorch familiarity.

| Phase | Work | Effort |
|---|---|---|
| **P0 — Proof of concept (RealESRGAN only)** | Take [tumuyan/RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) or [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile), wire your own JNI `.so` (`libenhancer.so`), verify x2/x4 on arm64 device | **3–5 days** |
| **P1 — Face pipeline on device** | RetinaFace (ncnn example models) + align/warp + paste-back in C++; run only on cropped faces with the Python GPU output as ground truth | **1–2 weeks** |
| **P2 — GFPGAN conversion** | pth → ONNX (apply noise + linearization fixes), validate PSNR/SSIM vs Python reference; then pnnx → ncnn; or adopt ONNX Runtime path if ncnn conversion fights back | **1–2 weeks** (riskiest phase) |
| **P3 — Integration & optimization** | Tile-based RealESRGAN (tile=400 like Python), fp16/int8 quantization of models, Vulkan backend for supported devices, threading, progress reporting | **1 week** |
| **P4 — Polish** | APK size strategy, device matrix testing (Mali vs Adreno), thermal/ANR handling (10+ s operations → WorkManager/foreground service) | **1 week** |

**Total: ~4–7 engineering-weeks for a complete, shippable full-pipeline `.so`; ~1–2 weeks for RealESRGAN-only.** Realistic risk profile: conversion & performance (medium-high risk) > memory (medium) > JNI plumbing (low).

---

## 8. Risks & Gotchas

1. **Performance ceiling** — seconds per photo is the reality; set expectations early (see §4).
2. **fp16 ARM precision bugs** — ncnn fp16 caused wrong RetinaFace landmarks on ARM; pin ncnn version and test (ncnn#2885, ncnn#2678).
3. **Github model URLs hardcoded** (`enhancer.py:24,42,58...`) — irrelevant for mobile (models ship as assets), but note the pipeline currently auto-downloads at init; the native port must bundle weights.
4. **Face parsing model** — `use_parse=True` adds a model; simplest faithful port includes it, pragmatic port fakes the mask.
5. **GPU (Vulkan) fragmentation** — Vulkan is fast on Adreno/Mali but must gracefully fall back to CPU; ncnn handles this via `use_vulkan_compute` flag.
6. **Thermal/ANR** — 10–60 s operations must run off the main thread; Android may kill background work — use foreground service or user-initiated jobs.
7. **Licensing** — GFPGAN (Apache-2.0), Real-ESRGAN (BSD-3), ncnn (BSD-3) are permissive; the repo itself is MIT. No blocking license issues for commercial use, but keep attributions.
8. **Output size** — upscale=2/4 on a 2048px input produces huge bitmaps (8192² × 3 bytes ≈ 200 MB RAM) — decode/encode JPEG streaming and tile processing are mandatory.

---

## 9. Final Verdict

| Scenario | Difficulty | Effort | Recommendation |
|---|---|---|---|
| `.so` with **Real-ESRGAN only** (background upscale) | **Easy** | 1–2 weeks | Copy ncnn-vulkan port, ship `.param/.bin` |
| `.so` with **full pipeline** (GFPGAN faces + RealESRGAN bg) | **Medium-Hard** | 4–7 weeks | ncnn + own GFPGAN conversion; validate quality vs Python |
| Real-time enhancement on device | **Not feasible** with these models | — | Needs a different (smaller) model family |

**Bottom line:** The codebase is unusually well-positioned for a native port because Real-ESRGAN already has an official ncnn Android path, and the whole pre/post-processing (detection→align→paste) is standard OpenCV work with public ncnn examples. GFPGAN itself is the one genuinely hard component — plan the conversion as a spike before committing to the full build. All required pieces are open source; none of them is a finished, copy-paste Android GFPGAN integration as of this writing.

---

## 10. Ready-Made Open Source — What You Can Adopt Today (Focus: Face Enhancement)

*Follow-up deep research (2026-08-12). Bottom line: **yes, ready open source exists** — but not as one finished app that does exactly this repo's GFPGAN+RealESRGAN pipeline. Every component exists; some are drop-in, some need assembly.*

### 10.1 Drop-in: Pre-converted face-restoration models (biggest unlock)

**[harisreedhar/Face-Upscalers-ONNX](https://github.com/harisreedhar/Face-Upscalers-ONNX)** — official-style ONNX conversions of *all* major face upscalers, published as GitHub release assets ([release page](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/tag/Models)):

| Model | fp32 size | fp16 size (HF mirror) | Notes |
|---|---|---|---|
| GFPGANv1.4.onnx | 324.5 MB | ~170 MB | **Same model this repo uses** ([netrunner-exe HF mirror](https://huggingface.co/netrunner-exe/Face-Upscalers-onnx) has v1.2/1.3/1.4 fp16) |
| GPEN-BFR-256.onnx | 72.2 MB | **38 MB** | Most mobile-friendly face enhancer |
| GPEN-BFR-512.onnx | 271.1 MB | — | Better quality than 256 |
| codeformer.onnx | 359.3 MB | — | — |
| restoreformer.onnx | 278.5 MB | — | — |

These run directly on Android via the official **onnxruntime-android** AAR (Java + C++ API) — no conversion work at all. They are already battle-tested in production-grade open source: **Deep-Live-Cam** consumes `GPEN-BFR-256.onnx` / `GPEN-BFR-512.onnx` straight from these releases ([source](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_enhancer_gpen256.py)).

### 10.2 Drop-in: Complete Android apps (open source, released APKs)

| App | Stack | Face enhancement? | License |
|---|---|---|---|
| **[Lucchetto/SuperImage](https://github.com/Lucchetto/SuperImage)** | MNN + Real-ESRGAN, GPU (Vulkan/OpenCL), tiled processing, on Play Store | ❌ (background only) | GPL-3.0 |
| **[egeoz/Upscaler](https://github.com/egeoz/Upscaler)** | Fork of SuperImage, same MNN+RealESRGAN | ❌ | GPL-3.0 |
| **[tumuyan/RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android)** | ncnn + Vulkan, multiple SR models | ❌ | — |
| **[HF-CYGG/LumaSR](https://github.com/HF-CYGG/LumaSR)** | ncnn + Vulkan, Kotlin/Compose, model manifest | ❌ | — |
| **[Yang-013/Stable-diffusion-Android-termux](https://github.com/Yang-013/Stable-diffusion-Android-termux)** | Termux + GFPGAN (Python on-device) | ✅ GFPGAN **face only** — works, but crashes with RealESRGAN bg (RAM) | — |

**No finished open-source Android app ships the GFPGAN face-restoration path in native code yet** — confirmed by [GFPGAN discussion #376](https://github.com/TencentARC/GFPGAN/discussions/376) (still just "has anyone tried?") and [onuralpszr/GFPGAN-ncnn-vulkan#19](https://github.com/onuralpszr/gfpgan-ncnn-vulkan/issues/19) (the only GFPGAN ncnn Vulkan port doesn't run on Android, author says "I wasn't targeting android"). The Termux route proves GFPGAN *works* on ARM devices, but is a hack, not a shippable app.

### 10.3 Ready components (models + engines) you assemble yourself

| Need | Ready open source | Effort to wire |
|---|---|---|
| Face detection + 5 landmarks (ARM) | [ncnn RetinaFace example](https://github.com/Tencent/ncnn/blob/master/examples/retinaface.cpp) (mnet models), [nihui/ncnn-android-scrfd](https://github.com/nihui/ncnn-android-scrfd) (Android project), MNNKit FaceDetection SDK on Maven (`com.alibaba.android.mnnkit:facedetection`) | 1–3 days |
| Face restoration model | Face-Upscalers-ONNX (10.1) via onnxruntime-android; OR ncnn-converted CodeFormer 212.5 MB / GFPGAN 214 MB from [Baiyuetribe/ncnn-models](https://github.com/Baiyuetribe/ncnn-models) | 1–2 days |
| Background upscale | RealESRGAN ncnn `.param/.bin` bundled in [tumuyan assets.zip](https://github.com/tumuyan/RealSR-NCNN-Android/releases/download/1.11.1/assets.zip) or [LumaSR](https://github.com/HF-CYGG/LumaSR) builtin models | 1–2 days |
| Whole JNI + CMake skeleton | [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile) (NCNN C++/JNI + ExecuTorch variants, with on-device PSNR/SSIM benchmarks) | 1 day |
| Align/warp/paste-back | OpenCV (prebuilt for Android) — affine transform + feathering, ~50 lines | 1 day |

### 10.4 Recommended fast path (fewest unknowns)

> **ONNX Runtime Mobile + Face-Upscalers-ONNX models + SCRFD/RetinaFace (ncnn) + RealESRGAN-ncnn for background.**

- Face path: `GFPGANv1.4.onnx` (fp16, ~170 MB) or `GPEN-BFR-256.onnx` (fp16, 38 MB) — no graph surgery needed since conversions already solved the StyleGAN-noise problem ([see repo](https://github.com/harisreedhar/Face-Upscalers-ONNX)).
- Everything runs through the `onnxruntime-android` AAR or its C++ API inside your `.so`; ncnn handles detection + background for maximum speed.
- This path de-risks the single hardest task from §6 (GFPGAN conversion) completely.
- Expected effort drop: **full pipeline 4–7 weeks → ~2–3 weeks**; face-only enhancement even less.

### 10.5 Commercial (not open source, for reference)

- **eyeQ Perfectly Clear SDK** — Android AAR with on-device face beautification, auto-exposure, noise reduction; `arm64-v8a/armeabi-v7a/x86_64` supported ([docs](https://docs.eyeq.photos/docs/mobile-sdk/android)). Paid; useful as a baseline if a ready SDK is preferred over DIY.

### 10.6 What still does NOT exist (verify before committing)

1. A finished, native Android app implementing this repo's exact pipeline (GFPGAN faces + RealESRGAN bg + paste-back).
2. A working Android integration of `GFPGAN-ncnn-vulkan` (WIP, open issue).
3. Mobile-specific GFPGAN distillation/quantization to make 512×512 GAN inference fast (<1 s) on mid-range ARM — expect seconds per face (see §4).

---

## Sources

1. [xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) — official ncnn+Vulkan port of Real-ESRGAN (same model as this repo's bg upsampler)
2. [tumuyan/RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) — production Android app: Real-ESRGAN/RealSR/Waifu2x via ncnn `.so`, with model assets zip
3. [HF-CYGG/LumaSR](https://github.com/HF-CYGG/LumaSR) — Android offline AI super-resolution app (Kotlin + Compose + JNI/CMake + ncnn Vulkan), Real-ESRGAN included
4. [ibrahimghadre/realesrgan-mobile](https://github.com/ibrahimghadre/realesrgan-mobile) — Real-ESRGAN on Android, NCNN vs ExecuTorch benchmark: 1.85 s vs 4.2 s per image; ExecuTorch instability on GAN graphs
5. [onuralpszr/GFPGAN-ncnn-vulkan](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan) — WIP GFPGAN ncnn port; [issue #19](https://github.com/onuralpszr/gfpgan-ncnn-vulkan/issues/19) reports it doesn't run on Android
6. [FeiGeChuanShu/GFPGAN-ncnn](https://github.com/FeiGeChuanShu/GFPGAN-ncnn) — naive CPU-only GFPGAN ncnn port (GFPGANCleanv1-NoCE-C2)
7. [xuanandsix/GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo) — working pth→ONNX recipe for GFPGANv1.3, with documented noise/linearization fixes; [DeepWiki analysis](https://deepwiki.com/xuanandsix/GFPGAN-onnxruntime-demo)
8. [Baiyuetribe/ncnn-models](https://github.com/Baiyuetribe/ncnn-models) — already-converted ncnn models: GFPGAN (214 MB ✅), CodeFormer (212.5 MB ✅)
9. [sczhou/CodeFormer#401](https://github.com/sczhou/CodeFormer/issues/401) — community report: RealESRGAN on Android works well via ncnn; CodeFormer conversion "much more complicated"
10. [xinntao/Real-ESRGAN#86](https://github.com/xinntao/Real-ESRGAN/issues/86) — author comments on mobile: minutes-long full-GPU runs, not production-ready as-is
11. [Tencent/ncnn](https://github.com/Tencent/ncnn) + [releases](https://github.com/Tencent/ncnn/releases/tag/20250503) — Android prebuilt libs (12.6–26.2 MB zips), pnnx conversion, ARM NEON + Vulkan
12. [ncnn#2885](https://github.com/Tencent/ncnn/issues/2885) / [ncnn#2678](https://github.com/Tencent/ncnn/issues/2678) — fp16 arithmetic bug breaking RetinaFace landmarks on ARM
13. [Tencent/ncnn retinaface example](https://github.com/Tencent/ncnn/blob/master/examples/retinaface.cpp) — ready-made C++ RetinaFace detection used by the official example
14. [pytorch/pytorch#59865](https://github.com/pytorch/pytorch/issues/59865), [PyTorch Mobile size discussion](https://discuss.pytorch.org/t/is-it-possible-to-build-a-minimal-libpytorch-so-for-c-inference-on-mobile-device/97798) — libtorch Android size reality (75–89 MB libtorch_cpu)
15. [pytorch/android README](https://github.com/pytorch/pytorch/blob/main/android/README.md) — linking libtorch into your own CMake/JNI `.so`
16. [pytorch/executorch](https://github.com/pytorch/executorch) + [ExecuTorch 1.0 announcement](https://pytorch.org/blog/introducing-executorch-1-0/) + [Android docs](https://docs.pytorch.org/executorch/stable/using-executorch-android.html) — `.pte` AOT compilation, XNNPACK/Vulkan/QNN backends
17. [onnxruntime.ai mobile docs](https://onnxruntime.ai/docs/tutorials/mobile/) — ORT mobile packages; custom build shrinks `libonnxruntime.so` 16.3 MB → 4.0 MB (arm64)
18. [alibaba/MNN](https://github.com/alibaba/MNN) — ~800 KB core `.so`, TorchScript/ONNX input, Vulkan support
19. [Qualcomm AI Hub — Real-ESRGAN-x4plus](https://aihub.qualcomm.com/models/real_esrgan_x4plus) — model sizes: 63.9 MB fp32, 16.7 MB w8a8
20. [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) — model zoo (v1.4 = clean arch, channel_multiplier=2 — matches `enhancer.py`), Apache-2.0
21. [GFPGAN utils.py (GFPGANer)](https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/utils.py) — exact pipeline steps the repo delegates to (detect → align 512 → GFPGAN → RealESRGAN bg → paste back)
22. [harisreedhar/Face-Upscalers-ONNX](https://github.com/harisreedhar/Face-Upscalers-ONNX) + [Models release](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/tag/Models) — pre-converted ONNX: GFPGANv1.0–v1.4, GPEN-256/512, CodeFormer, RestoreFormer (72–359 MB fp32)
23. [netrunner-exe/Face-Upscalers-onnx (HF)](https://huggingface.co/netrunner-exe/Face-Upscalers-onnx) — fp16 variants: GFPGANv1.4 ~170 MB, GPEN-BFR-256 38 MB
24. [hacksider/Deep-Live-Cam face enhancers](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_enhancer_gpen256.py) — production consumer of Face-Upscalers-ONNX models via ONNX sessions
25. [Lucchetto/SuperImage](https://github.com/Lucchetto/SuperImage) — complete Android upscaling app, MNN + Real-ESRGAN on GPU, GPL-3.0, released APK
26. [egeoz/Upscaler](https://github.com/egeoz/Upscaler) — SuperImage fork (MNN + Real-ESRGAN) with APK releases
27. [Yang-013/Stable-diffusion-Android-termux](https://github.com/Yang-013/Stable-diffusion-Android-termux) — GFPGAN face restoration running on Android via Termux (crashes on RealESRGAN bg due to RAM)
28. [TencentARC/GFPGAN discussion #376](https://github.com/TencentARC/GFPGAN/discussions/376) — confirms no official/community Android app for GFPGAN exists as of the discussion
29. [nihui/ncnn-android-scrfd](https://github.com/nihui/ncnn-android-scrfd) — ready Android face detection project (SCRFD, ncnn + OpenCV)
30. [alibaba/MNNKit](https://github.com/alibaba/MNNKit) — ready-made face detection SDK for Android on Maven Central (com.alibaba.android.mnnkit)
31. [eyeQ Perfectly Clear Android SDK](https://docs.eyeq.photos/docs/mobile-sdk/android) — commercial on-device enhancement/face beautification AAR
32. [FeiGeChuanShu/CodeFormer-ncnn](https://github.com/FeiGeChuanShu/CodeFormer-ncnn) — ncnn port of CodeFormer (desktop-oriented reference for conversion)
33. [onnxruntime super-res Android tutorial](https://onnxruntime.ai/docs/tutorials/mobile/superres.html) — official ORT Mobile Android app pattern (Java + custom ops)

## Methodology (round 2)
Searched 9 additional Exa queries focused on ready-made resources: complete Android apps, pre-converted model files, SDKs/AARs, GPEN/CodeFormer mobile ports, and community confirmations of what works vs what doesn't. Cross-referenced GitHub releases (verified asset sizes), Hugging Face mirrors, and maintainer/community issue threads. Model sizes verified from release metadata; app maturity claims from repo status + issue threads.

## Methodology

Searched 11 queries across Exa web search covering: GFPGAN-on-Android resources, RealESRGAN ncnn/Vulkan Android, PyTorch Mobile/ONNX/ExecuTorch size & performance data, RetinaFace-on-ncnn, model conversion pitfalls, and pre-converted model repositories. Cross-referenced 20+ sources; performance figures are from a single published on-device benchmark and device-dependent — flagged as medium confidence. Codebase claims verified against `enhancer/enhancer.py`, `requirements.txt`, and the GFPGAN upstream source.
