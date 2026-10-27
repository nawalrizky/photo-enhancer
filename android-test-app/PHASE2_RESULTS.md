# Phase 2 Results

## Status

Phase 2 memory optimization and RK3576 profiling are complete. FP16 CPU inference is the accepted configuration. It meets the numerical quality and latency criteria and substantially reduces memory, but the measured true process peak remains above the 500 MB target.

## Final Configuration

- Model: `GFPGANv1.4_fp16.onnx`, with float32 input/output tensors
- Execution provider: ONNX Runtime CPU
- Four intra-op threads with sequential execution
- CPU memory arena disabled
- Memory pattern disabled
- Direct per-pixel `CV_8UC3` paste-back blending
- ML Kit face detector closed after landmark extraction
- Previous result bitmap recycled before another enhancement

The app selects FP16 before FP32 and INT8. FP32 remains a rollback model, while INT8 remains an experimental fallback because it failed the Phase 1 quality gate.

## Model Quality

| Metric | FP16 versus FP32 | Acceptance |
| --- | ---: | ---: |
| Model size | 170,260,352 bytes (162.4 MiB) | -50.0% |
| Close-face MAE | 0.000272 | Pass, maximum 0.02 |
| Close-face RMSE | 0.000377 | Pass |
| P99 absolute error | 0.000876 | Pass |

The conversion keeps model I/O as float32, so the Android preprocessing and postprocessing contract is unchanged.

## Memory Progression

The main profiling sequence used the same full enhancement path and native `/proc/self/status` RSS/VmHWM instrumentation.

| Configuration | RSS HWM | Full elapsed | Decision |
| --- | ---: | ---: | --- |
| FP32, float full-frame blend | 1,410,164 KB | Baseline | Reject |
| FP32, direct 8-bit blend | 1,133,452 KB | 17.941s | Keep blend |
| FP32, direct blend, arena/pattern off | 955,704 KB | 18.430s | Keep allocator settings |
| FP16, four threads | 796,172 KB | approximately 18.2s | Keep FP16 |
| FP16, two threads | 778,856 KB | 26.882s | Reject: too slow |
| FP16, NNAPI | 887,796 KB | 27.594s | Reject: slower and larger |
| Final FP16 debug, detector released | 723,836 KB | 18.053s | Accept |
| Final FP16 minified release | 708,420 KB | 18.506s | Accept |

Direct blending removed the full-frame float source, restored image, inverse mask, and expression temporaries from the peak paste-back path. Disabling the CPU arena and memory pattern allows ONNX Runtime workspace pages to return after inference instead of remaining retained by the session allocator.

The final detector-lifecycle benchmark used a 256x256 source portrait. Before model initialization, `dumpsys meminfo` reported PSS `104,763 KB` and RSS `229,320 KB`. During inference, sampled values reached PSS `540,122 KB` and RSS `664,712 KB`; native VmHWM recorded `723,836 KB`. After completion, memory settled to PSS `293,050 KB` and RSS `417,644 KB`.

The test-signed minified release was then installed over the debug build and completed the same full path in 18.506 seconds with native VmHWM `708,420 KB`. This verifies the R8/JNI release path without changing the peak-memory decision.

VmHWM is the authoritative true RSS high-water mark. Sampled `dumpsys` values can miss short-lived workspace peaks and should not be substituted for HWM when evaluating the 500 MB criterion.

## Rejected Approaches

### Full-frame tiling

The pipeline runs GFPGAN once on one aligned 512x512 face. The full image is used only for warp and paste-back. Tiling the source would not reduce the fixed 512x512 model workspace and would change the pipeline semantics or invoke GFPGAN repeatedly, so it was not implemented.

### Two CPU threads

Two threads saved only 17,316 KB of HWM versus the initial FP16 four-thread run while increasing full enhancement to 26.882 seconds, beyond the latency criterion.

### NNAPI

The RK3576 NNAPI driver supported 42 of 645 nodes in one graph partitioning pass and 79 of 719 in another. CPU fallback dominated execution. NNAPI increased both HWM and latency, so the UI toggle remains diagnostic and defaults off.

### XNNPACK

The XNNPACK candidate did not complete inference within 40 seconds and was rejected.

### Prepacking and pruning

Disabling session prepacking produced no useful memory reduction and was reverted. Structural pruning was not attempted because no training/fine-tuning pipeline or source PyTorch checkpoint was available; graph-level weight removal without retraining would create an unacceptable quality risk.

## Acceptance Decision

| Criterion | Result |
| --- | --- |
| Quality at least 98% of FP32 / MAE at most 0.02 | Pass, MAE 0.000272 |
| Full enhancement at most 20-22 seconds | Pass, 18.506 seconds in release verification |
| No native crash | Pass |
| Post-run RSS at most 500 MB | Pass, 417,644 KB |
| True peak RSS at most 500 MB | Fail, release VmHWM 708,420 KB |

Reaching a true peak below 500 MB requires reducing ONNX Runtime activation/workspace memory, not further full-image paste-back tuning. A different graph architecture, operator implementation, or hardware provider with broad graph coverage is the next technically credible route.

## Reproduction

1. Convert and deploy the model with `python scripts/convert_model_fp16.py` and `scripts/push_model.sh fp16`.
2. Build and install the debug APK so `run-as` model deployment remains available.
3. Select a portrait and wait for landmark detection.
4. Run `scripts/benchmark_phase2.sh phase2-benchmark` with NNAPI disabled.
5. Inspect `native-memory.log`, `meminfo.txt`, and `result.png` in the output directory.
