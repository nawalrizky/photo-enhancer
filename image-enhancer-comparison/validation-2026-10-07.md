# Follow-up validation on RK3576 (2026-10-07)

The original comparison remains a valid record of its own end-to-end test: REST API succeeded on 6/6 images and the offline SDK on 3/6. Its 2.5x latency ratio mixes different image sets. On the three shared successes (Low Light, Low Resolution, Low Resolution Cropped), REST API averaged `(6.24 + 6.21 + 5.70) / 3 = 6.05` seconds and SDK averaged `(18.77 + 18.27 + 18.16) / 3 = 18.40` seconds, so the SDK was `18.40 / 6.05 = 3.04x` slower in that test.

## Reproduction on the available device

Device `192.168.2.42:37361`, model `GFPGANv1.4_fp16.onnx`, current debug test app with eight ONNX Runtime CPU threads. The benchmark loads the files in `original/`, runs ML Kit face/landmark detection, and measures SDK initialization plus full enhancement. This is **not** the delivered 1.0.0 AAR's four-thread configuration, so the times below must not be substituted into the original comparison.

| Original file | Decoded size | Faces | Five landmarks | Result | Init + full enhancement |
| --- | ---: | ---: | --- | --- | ---: |
| `close-up.png` | 401×498 | 1 | All present | Success | 15.013 s |
| `elderly.jpeg` | 1519×2048 after app resize | 1 | All present | Success | 14.868 s |
| `low-light.png` | 365×547 | 1 | All present | Success | 14.837 s |

These results do **not** reproduce the original Close Up and Elderly failures. The original host app's detector, orientation/resize handling, landmark coordinates, exact input versions, release AAR integration, and underlying exception are not available in this folder. The generic message “Gagal enhance. Pastikan wajah terdeteksi dan model tersedia” does not distinguish detection from SDK initialization or inference failure. `Low Light Cropped` has no source image in `original/`, so that failure cannot be reproduced yet. The test app now logs `EnhancerFaceDiag` with decoded dimensions, face count, bounds, and landmark availability for future reproduction.

## Quality comparison scope

The server uses `GFPGANer` with `upscale=2`, and may enable Real-ESRGAN background enhancement when CUDA is available. The offline SDK enhances one aligned 512×512 face and pastes it back into an output with the **same dimensions as its input**; it does not enhance the background. The repository's Dockerfile is CUDA-based, but that alone does not prove a GPU was active during the comparison. The saved files in `sdk-nawal-ai-enhance-image/` and `rest-api-ai-enhanced-image/` are 1080×1920 or 1920×1080 screen captures, not matched raw enhancement outputs. Therefore the low-light smoothness observation is useful visually, but pixel-level quality or attribution to a specific model stage cannot yet be verified from these artifacts.

## Required for a definitive fix

1. Exact original `Low Light Cropped` input and, if they differ, the original `Close Up` and `Elderly` inputs used by the host app.
2. Host app logcat/exception or integration code around face detection, bitmap resize/orientation, `initialize`, and `enhanceFull` for each failure. Record the AAR checksum and model checksum.
3. Raw API and SDK output files from the same input, plus server GPU/background-enhancement configuration. Compare both at the same output dimensions before drawing a quantitative quality conclusion.

Do not replace the delivered 1.0.0 ZIP based on the debug app reproduction alone.
