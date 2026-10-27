# Phase 3: sub-5-second latency investigation

## Update — 2026-10-07 Linux/RKNN execution

The Linux conversion, runtime smoke test, experimental Android backend and four-image validation are complete. All 12 warm full-pipeline calls (including repeated decode and face detection) are 1.573–2.672 s. Raw CPU/RKNN image comparisons have a maximum RGB channel difference of 1–2/255. First elderly processing remains 5.980 s; application peak memory with diagnostic export reaches about 1.03 GiB.

The initial RKNN exports failed quality because RGB bias contributions were lost. A mathematically equivalent RGB bias consolidation fixed the discrepancy. The decision is ACCEPT as a candidate for production integration under the plan's warm-call gate; the delivered 1.0.0 AAR/ZIP remain unchanged. See [current execution results](../PHASE3_RKNN_EXECUTION_RESULTS.md). The sections below record earlier CPU and Docker experiments.

## Device and measurement

- Device: YF_021B, Android/RK3576, 8 CPU cores, 8 GB RAM.
- Image: `samples/obama.jpg`, decoded to 256x256 pixels.
- Model: `GFPGANv1.4_fp16.onnx`; one aligned 512x512 face per run.
- Timing: `System.nanoTime()` around the SDK call and `steady_clock` around native stages. The first run includes session initialization; later runs reuse the session.
- Benchmark: `scripts/benchmark_latency.sh` starts the test app with `benchmark_image` and `benchmark_runs` intent extras, then reads `EnhancerBench` and `AiFilterEnhancer` logcat tags.

| CPU configuration | First full call | Warm full calls | Inference portion | VmHWM |
| --- | ---: | ---: | ---: | ---: |
| 4 threads, arena/pattern off | 16.579s | 15.306s, 15.227s | 15.196–15.292s | Not sampled |
| 8 threads, arena/pattern off | 14.951s | 13.925s, 13.759s | 13.616–13.895s | 740,128 KB in repeat verification |
| 8 threads, arena/pattern on | 14.661s | 13.300s, 13.067s | 13.037–13.292s | 1,193,188 KB |

The 8-thread setting with arena and memory pattern disabled is the current source configuration. It improves warm latency by about 10% versus 4 threads on this unit. Enabling the arena/pattern saved only about 0.7s while adding roughly 453 MB to the process high-water mark, so that experiment was reverted.

The existing ONNX Runtime 1.18 Android library contains XNNPACK. A test build explicitly enabled it with an 8-thread XNNPACK pool while retaining the 8-thread CPU fallback. On the same unit and image, full-call times were 24.916s first and 22.322s/22.378s warm, substantially worse than the 13.7s CPU baseline. This experiment was reverted in source; it does not offer a path to sub-5-second latency for this GFPGAN graph.

The native stage logs show that inference consumes more than 99% of warm call time. Bitmap conversion, alignment, preprocessing, output conversion, and paste-back together are below 50 ms for this 256x256 image. Reducing image size or rewriting paste-back cannot bridge the remaining gap to 5 seconds because GFPGAN still processes a fixed 512x512 tensor.

## NPU readiness

The board exposes `/dev/dri/renderD129` with `DRIVER=RKNPU` and compatible string `rockchip,rk3576-rknpu`. It does not currently expose `librknnrt.so` in `/vendor`, `/system`, or `/odm`. The existing ONNX model cannot run on this NPU directly: Rockchip's flow first converts it to RKNN format with RKNN-Toolkit2, then invokes the RKNN Runtime API. This path has not yet been converted, integrated, or benchmarked, so sub-5-second latency is **not verified**.

Docker Desktop initially failed because an old backend process was stuck. After the host cache cleanup and a targeted backend restart, Docker client/server 28.4.0 became available. RKNN-Toolkit2 2.3.2 and its ARM64 dependencies were installed in a temporary Python 3.10 Linux container.

The FP16 ONNX model loaded, but compilation failed during constant folding with `The shape of '/conv_body_down.0/Resize_input_cast_1' is (0,), it is abnormal!`. The original FP32 ONNX model passed that stage and reached `I rknn building ...`; conversion was stopped before export because host free space fell to 945 MiB and the converter container was using 2.56 GiB of its 3 GiB memory limit. No `.rknn` file was produced. The temporary container and Python image were removed; unrelated Docker containers, images, and volumes were retained. Do not replace the delivered 1.0.0 AAR or ZIP with this experimental build until output quality and device latency are verified.

After removing only rebuildable Gradle, npm, and Deno caches, host free space rose to about 12 GiB. A second FP32 conversion reached `I rknn building ...` again, but the process was OOM-killed at the container's 3 GiB RAM limit (exit 137, `OOMKilled=true`) while 8.6 GiB of host disk remained. This confirms that disk space alone is not the remaining conversion blocker; the conversion environment needs more memory. No `.rknn` file was produced on the retry.

A third FP32 attempt removed the container-specific memory cap, allowing up to the Docker VM's 3.83 GiB, but was again killed at `I rknn building ...` (exit 137). Inspection of the FP16 graph found 42 `Resize` nodes with empty ROI constants cast to float32. `scripts/repair_fp16_resize_roi.py` replaces only these empty constants with the equivalent explicit ROI for `half_pixel` mode and validates the resulting ONNX graph. This repaired FP16 graph passed RKNN's former constant-folding failure, but its compilation was likewise killed at `I rknn building ...` under the 3.83 GiB Docker VM limit. No `.rknn` file was produced. The four existing application containers were left running.

After the Docker VM allocation was raised from 4 GiB to 5.5 GiB on the 8 GiB Mac, two more repaired-FP16 attempts reached `I rknn building ...` and were OOM-killed (exit 137) even with RKNN's `optimization_level=0`; the default level 3 failed identically. The Docker VM exposed 5.295 GiB to containers. No RKNN model was exported. Further whole-model conversion on this Mac is not advisable; use a Linux host with at least 16 GiB RAM and adequate disk, or redesign/split the graph before compilation. The running four application containers were restored after Docker's settings restart.

## Next validation gate

1. Retry conversion on a Linux host with substantially more RAM than this Docker VM. FP32: `python scripts/convert_model_rknn.py --input models/GFPGANv1.4.onnx --output models/GFPGANv1.4_rk3576.rknn`. FP16: first run `python scripts/repair_fp16_resize_roi.py --input models/GFPGANv1.4_fp16.onnx --output models/GFPGANv1.4_fp16_repaired.onnx`, then convert the repaired ONNX model. Validate FP16 output equivalence before shipping.
2. Inspect conversion warnings, operator coverage, model input/output contract, and resulting model size. Reject conversion if graph sections fall back to CPU or output changes materially.
3. Bundle a compatible `librknnrt.so` with a test build, run the model on this board, and measure 3 warm full calls plus peak memory and visual/numerical quality against the current FP16 ONNX output.
4. Accept the NPU build only when all three warm full calls are below 5 seconds on representative images and the quality gate is met.

Official references: [RKNN-Toolkit2 platform support](https://github.com/airockchip/rknn-toolkit2), [Android RKNN runtime integration](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/examples/librknn_api_android_demo/README.md).
