# Third-party components

Retain the upstream notice files provided in `licenses/` when redistributing this package.

| Component | Included version/provenance | Notice |
| --- | --- | --- |
| RKNN Android runtime/header | Toolkit2 v2.3.2, commit `42aa1d426c0a9e0869b6374edba009f7208a1926` | `licenses/RKNN-LICENSE.txt`, upstream Rockchip copyright statement |
| GFPGAN model | GFPGAN v1.4; ONNX from harisreedhar/Face-Upscalers-ONNX; validated RGB bias rewrite | `licenses/GFPGAN-LICENSE.txt`, upstream project license |
| OpenCV native runtime | OpenCV 4.10.0 Android AAR | `licenses/OpenCV-LICENSE.txt` |
| LLVM/libc++ and Android native runtime notices | Android NDK 27.2.12479018 | `licenses/NDK-NOTICE.txt` and `licenses/LLVM-NOTICE.txt` |

The SDK package does not include ONNX Runtime. The demo host declares Kotlin
stdlib 1.9.24 as its own Gradle dependency.

Upstream sources:

- https://github.com/airockchip/rknn-toolkit2/tree/42aa1d426c0a9e0869b6374edba009f7208a1926
- https://github.com/TencentARC/GFPGAN
- https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/tag/Models
- https://github.com/opencv/opencv/tree/4.10.0
- https://github.com/llvm/llvm-project
