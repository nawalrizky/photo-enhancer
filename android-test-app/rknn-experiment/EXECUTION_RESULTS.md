# RKNN execution — 2026-10-07

Status: all four phase milestones complete; ACCEPT as a candidate for production integration under the plan's warm-call gate. See [final results](../../PHASE3_RKNN_EXECUTION_RESULTS.md) for full application timing, quality, memory and limits. Delivered AAR 1.0.0 and ZIP have not been replaced. The sections below retain the investigation chronology.

## Conversion

Host: `ubuntu@10.147.20.16`, project directory `/home/ubuntu/Documents/work/photo-enhancer-rknn`. Python 3.10, RKNN Toolkit2 2.3.2, target RK3576, unquantized compilation.

| Source | Export bytes | Conversion time | Sampled process-tree peak RSS |
| --- | ---: | ---: | ---: |
| Original FP32 ONNX | 458,742,006 | 75.66 s | 5,730.56 MiB |
| FP16 ONNX with 42 empty Resize ROIs repaired | 458,747,382 | 70.21 s | 5,763.11 MiB |

SHA256 of exported FP32 candidate: `2a5c268c7c926393bd08a542237dedf01c7ef7c8eb0ecfbeba8cf11ecb697d22`.

SHA256 of exported FP16 candidate: `d12fe059f1e424cf9c86f08bb634f628ec7bd788c13506450af8470cb80b0805`.

Conversion logs contain `Unkown op target: 0`; successful export alone does not establish operator coverage. Logs are retained in `logs/conversion-host`.

The initial GFPGAN device profile (`artifacts/gfpgan-profile.perf.txt`) shows eight `Mul` operators on CPU, plus input/output operators. Total CPU operator time is 3,854 us out of 1,071,075 us (~0.36%). The expensive convolution and resize operations are on NPU. The final candidate profile (`artifacts/consolidated-profile.perf.txt`) retains these small CPU operations and records about 1.09% CPU operator time. Mixed execution is accepted for this candidate after measured quality, full latency and short stability gates pass; full-NPU execution is not claimed. Profiling timings are separate from the unprofiled benchmark.

## Device and input contract

Device: YF_021B, RK3576, Android 14. Official Android runtime from RKNN Toolkit2 tag v2.3.2, commit `42aa1d426c0a9e0869b6374edba009f7208a1926`; runtime API 2.3.2, NPU driver 0.9.8.

The standalone `rknn_probe.cpp` supports native input packing. For these tests, input must be packed as FP16 NHWC and submitted with `pass_through=1`. The initial FLOAT32 NCHW submission produced incorrect smoke output. Native submission fixed the discrepancy: smoke MAE against ONNX CPU was 0.00002583954, all four runs finite and matching. The smoke profile identifies ConvRelu on NPU; that profile does not establish GFPGAN coverage.

## GFPGAN FP16 candidate, synthetic tensor

| Measurement | Result |
| --- | ---: |
| Runtime initialization | 1,369.55 ms |
| First inference call | 1,266.48 ms |
| Warm inference calls | 1,342.05 / 1,403.01 / 1,331.51 ms |
| Process VmHWM | 720,716 KiB (~703.8 MiB) |
| Nonfinite output elements | 0 in each run |

Profiling was disabled for these timings. Inference call time includes input submission, run, and output retrieval. It excludes image decoding, face detection, alignment, native input packing, paste-back and file I/O. This is **not an end-to-end sub-5-second result**. Process VmHWM does not account for every possible driver allocation.

Raw input, output and summary are in `artifacts/gfpgan-fp16-native.*`; runtime log is `logs/device-gfpgan-fp16-native.log`. Numerical GFPGAN equivalence **failed** on the synthetic input: MAE 0.624818, RMSE 0.629570, maximum absolute error 0.715196, versus the provisional MAE limit 0.02. See `artifacts/gfpgan-synthetic-numerical-validation.json` and the raw CPU reference. Input upload succeeded; output upload failed, so the CPU reference was downloaded and compared locally using Python's float array reader.

The FP32-derived candidate also failed against the same original FP16 CPU reference (MAE 0.624598). Its warm inference calls were 1,353.16 / 1,445.49 / 1,359.14 ms. Standard FLOAT32/NCHW input submission on the FP16 candidate gave MAE 0.666704. The failure is therefore not isolated to the repaired FP16 candidate or one submission mode. Native input query confirms FP16 NHWC, matching the probe's packing. The ONNX graph contains no Random operators. Cause remains under investigation; these candidates cannot be accepted based on speed.

## RGB bias investigation

On a resized photograph, FP16 RKNN versus original FP16 CPU MAE is 0.624363. The RGB mean differences are +0.515610 / +0.670399 / +0.687080 with within-channel standard deviation below 0.00044. These closely match the negative sum of the first seven RGB skip biases: +0.515803 / +0.670537 / +0.686912. This is evidence for loss of bias contributions in compilation; exact compiler internals have not been confirmed. The side-by-side raw output visualization is `artifacts/photo-cpu-vs-rknn.png` (CPU left, RKNN right).

`scripts/consolidate_gfpgan_rgb_bias.py` moves all eight RGB biases through the linear RGB skip path into one final Add. It checks that intervening consumers are Add or linear Resize. This preserves the mathematical model; it does not fit a color correction to test outputs. Original FP32 ONNX versus rewritten FP32 ONNX on CPU:

| Input | MAE | Maximum absolute error |
| --- | ---: | ---: |
| Resized photograph | 3.3210e-8 | 2.3842e-7 |
| Synthetic tensor | 3.7001e-8 | 2.9802e-7 |

Rewritten RKNN export succeeded in 60.86 s; sampled peak RSS 5,749.02 MiB. Export size 458,741,110 bytes, SHA256 `ac5cb93f894599e26d4a3ea71f0b546073678c86016f15ca21b1efcadc4b7a1c`, verified after download.

The rewritten candidate passes the provisional tensor MAE gate versus original FP16 CPU:

| Input | MAE | RMSE | Maximum absolute error | P99 absolute error |
| --- | ---: | ---: | ---: | ---: |
| Resized photograph | 0.000379822 | 0.000520338 | 0.0149191 | 0.00160182 |
| Synthetic tensor | 0.00139073 | 0.00218200 | 0.0343132 | 0.00816596 |

All output elements are finite. Unprofiled photo-tensor warm calls are 1,445.27 / 1,429.58 / 1,363.37 ms; initialization 1,274.55 ms; process VmHWM 683,848 KiB (~667.8 MiB). This remains an inference probe measurement, not the full application pipeline.

Experimental Android backend built successfully under `rknn-experiment/android`. It initially required `benchmark_backend=rknn`; after a manual test still selected CPU, the app default was changed to RKNN. `benchmark_backend=cpu` remains available for reference tests. Production SDK source and delivered artifacts remain separate. The test app supports `benchmark_export=true` (raw PNG and landmark JSON) and `benchmark_full_pipeline=true` (repeat decode and face detection on every run). Full application validation completed on four images: all 12 warm calls below 5 s; raw RGB error at most 2/255; 16 successful RKNN calls, with identical per-image outputs. First elderly call is 5.980 s. Full timing and memory details are in the final results report.

## Acceptance gate completion

1. Tensor quality: completed; initial candidates failed, consolidated-bias candidate passed.
2. Candidate comparison: completed; FP32 initial export had the same RGB bias failure.
3. Experimental application backend: completed and built; baseline detection/alignment/paste-back retained.
4. Raw photo quality: completed for normal, close-up, elderly and available low-light; missing Low Light Cropped excluded.
5. First and three warm full-pipeline calls: completed for all four images. Memory and short repeated-call stability passed on the tested 8 GB board. Final profile has eight small CPU Mul operators plus I/O, about 1.09% CPU operator time; heavy operators run on NPU.

Production release packaging and longer endurance testing are subsequent work, not completed by this experiment.
