package com.enhancer.sdk

import android.graphics.Bitmap
import java.io.File
import kotlin.math.max

/** Offline full-image GFPGAN enhancer. One instance may process one image at a time. */
class AiFilterEnhancer : AutoCloseable {
    companion object {
        const val MAX_IMAGE_EDGE = 2048

        init {
            System.loadLibrary("onnxruntime")
            System.loadLibrary("opencv_java4")
            System.loadLibrary("ai_filter_enhancer")
        }
    }

    private var nativeHandle = 0L
    private var initializedModel: String? = null

    @Synchronized
    fun initialize(modelFile: File) {
        require(modelFile.isFile) { "Model not found: ${modelFile.absolutePath}" }
        if (nativeHandle != 0L && initializedModel == modelFile.absolutePath) return

        close()
        nativeHandle = initializeNative(modelFile.absolutePath)
        check(nativeHandle != 0L) { "Could not initialize the FP16 enhancement model" }
        initializedModel = modelFile.absolutePath
    }

    /**
     * Enhances one face and pastes it into a new bitmap with the same dimensions as [source].
     * Landmark order: left eye, right eye, nose, left mouth, right mouth as x/y pairs.
     */
    @Synchronized
    fun enhanceFull(source: Bitmap, landmarks: FloatArray): Bitmap {
        check(nativeHandle != 0L) { "Enhancer is not initialized" }
        require(!source.isRecycled) { "Source bitmap is recycled" }
        require(source.config == Bitmap.Config.ARGB_8888) { "Source must use ARGB_8888" }
        require(max(source.width, source.height) <= MAX_IMAGE_EDGE) {
            "Longest image edge must not exceed $MAX_IMAGE_EDGE px"
        }
        require(landmarks.size == 10) { "Exactly five landmark x/y pairs are required" }
        landmarks.forEachIndexed { index, value ->
            require(value.isFinite()) { "Landmark values must be finite" }
            val limit = if (index % 2 == 0) source.width.toFloat() else source.height.toFloat()
            require(value in 0f..limit) { "Landmark lies outside the source bitmap" }
        }
        return requireNotNull(enhanceFullNative(nativeHandle, source, landmarks)) {
            "Native full-image enhancement failed"
        }
    }

    @Synchronized
    override fun close() {
        if (nativeHandle != 0L) releaseNative(nativeHandle)
        nativeHandle = 0L
        initializedModel = null
    }

    private external fun initializeNative(modelPath: String): Long
    private external fun enhanceFullNative(
        handle: Long,
        source: Bitmap,
        landmarks: FloatArray
    ): Bitmap?
    private external fun releaseNative(handle: Long)
}
