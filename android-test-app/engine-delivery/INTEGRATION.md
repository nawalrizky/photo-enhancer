# Integration Guide

## Add The AAR

Place `ai-filter-enhancer-1.0.0.aar` in the app module's `libs` directory:

```kotlin
dependencies {
    implementation(files("libs/ai-filter-enhancer-1.0.0.aar"))
}

android {
    defaultConfig {
        ndk {
            abiFilters += "arm64-v8a"
        }
    }
}
```

The AAR contains `libai_filter_enhancer.so`, ONNX Runtime, OpenCV, and `libc++_shared.so` for `arm64-v8a`. Do not add duplicate copies of these native libraries. If the host app already ships OpenCV, ONNX Runtime, or `libc++_shared.so`, align versions and resolve packaging at the application level before release.

## Install The Model

`initialize` requires a normal filesystem path. Copy or provision `GFPGANv1.4_fp16.onnx` into app-private storage before initialization:

```text
/data/user/0/<host.package>/files/models/GFPGANv1.4_fp16.onnx
```

For enhancement to work offline on first use, ship the model through an install-time asset mechanism. The SDK never downloads or updates the model and does not declare `INTERNET` permission.

Bundling the model inside the base APK or an install-time asset pack increases installed size by approximately 162.4 MiB. If the host app downloads the model on demand, enhancement remains offline after that one-time provisioning step, but first use is not fully offline.

## API Usage

Create and initialize one long-lived instance. Initialization loads a large model and should not be repeated for each image.

```kotlin
import android.graphics.Bitmap
import com.enhancer.sdk.AiFilterEnhancer
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AiFilterEngine(
    private val modelFile: File
) : AutoCloseable {
    private val enhancer = AiFilterEnhancer()

    suspend fun initialize() = withContext(Dispatchers.Default) {
        enhancer.initialize(modelFile)
    }

    suspend fun enhance(
        source: Bitmap,
        landmarks: FloatArray
    ): Bitmap = withContext(Dispatchers.Default) {
        enhancer.enhanceFull(source, landmarks)
    }

    override fun close() {
        enhancer.close()
    }
}
```

`enhanceFull` is synchronous and CPU-bound. Never call it from the main thread. Calls on the same instance are serialized. Coroutine cancellation does not interrupt an inference that has already entered native code.

## Landmark Contract

Pass exactly ten floating-point values in this order:

```text
[
  leftEyeX, leftEyeY,
  rightEyeX, rightEyeY,
  noseX, noseY,
  mouthLeftX, mouthLeftY,
  mouthRightX, mouthRightY
]
```

Coordinates must:

- Be finite pixel coordinates
- Use the final oriented bitmap coordinate system
- Match any resize applied before enhancement
- Stay inside the bitmap bounds

The host app owns face detection and landmark selection. The SDK does not include ML Kit or another detector.

## Image Contract

- Any portrait, landscape, or square aspect ratio is supported.
- Input must be an unrecycled `Bitmap.Config.ARGB_8888` bitmap.
- EXIF rotation must already be applied to the bitmap and landmarks.
- The longest edge must not exceed 2048 px.
- Output dimensions are identical to input dimensions.
- The input bitmap is not modified or recycled by the SDK.
- The caller owns and should recycle the returned bitmap when appropriate.

For larger images, resize without cropping and scale all landmarks by the same factor:

```kotlin
val scale = minOf(1f, 2048f / maxOf(source.width, source.height))
val resized = if (scale < 1f) {
    Bitmap.createScaledBitmap(
        source,
        (source.width * scale).toInt(),
        (source.height * scale).toInt(),
        true
    )
} else {
    source
}
val resizedLandmarks = landmarks.map { it * scale }.toFloatArray()
```

## Production Configuration

The native runtime is fixed to the validated configuration:

- FP16 GFPGAN model
- ONNX Runtime CPU execution provider
- Four intra-op threads
- Sequential execution
- CPU memory arena disabled
- Memory pattern disabled
- Full-image direct 8-bit blending
- NNAPI disabled

Do not create multiple initialized SDK instances. Each instance owns a complete ONNX Runtime session and would multiply model and workspace memory.

## Error Handling

The public API throws:

- `IllegalArgumentException` for a missing model, invalid bitmap, dimensions, or landmarks
- `IllegalStateException` when initialization fails or enhancement is called before initialization
- `RuntimeException` for native inference failures

Always close the SDK when its owning engine scope is destroyed.

## Offline Verification

1. Install the host app and model.
2. Disable Wi-Fi and mobile data, or enable airplane mode.
3. Start the process from a stopped state.
4. Initialize the SDK and run enhancement.
5. Confirm the output completes and no network permission is introduced by manifest merging.

The delivered AAR manifest contains only `minSdkVersion=28` and declares no permissions, providers, services, activities, or receivers.
