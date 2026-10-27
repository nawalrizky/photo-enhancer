package com.enhancer

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream

class PhotoEnhancer(private val context: Context) {

    companion object {
        init {
            System.loadLibrary("photo_enhancer")
        }
    }

    private var isInitialized = false

    /**
     * Inisialisasi model ONNX dari folder assets Android ke internal storage.
     * @param modelAssetName Nama file model di folder assets (misal: "GFPGANv1.4.onnx" atau "GPEN-BFR-256.onnx")
     * @param useGPU Gunakan akselerasi hardware NNAPI/GPU (default: true)
     */
    fun init(modelAssetName: String = "GFPGANv1.4.onnx", useGPU: Boolean = true): Boolean {
        val modelFile = File(context.filesDir, modelAssetName)
        if (!modelFile.exists()) {
            copyAssetToFile(modelAssetName, modelFile)
        }
        isInitialized = initNative(modelFile.absolutePath, useGPU)
        return isInitialized
    }

    /**
     * Memperjelas area wajah dan memotongnya (cropped) sesuai dimensi target FR engine Anda (misal 112x112 / 256x256).
     * @param inputBitmap Bitmap foto asli dari kamera / galeri.
     * @param landmarks Array Float 5 titik koordinat [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4] (Mata Kiri, Mata Kanan, Hidung, Mulut Kiri, Mulut Kanan).
     * @param targetWidth Ukuran lebar hasil crop untuk input FR (default: 112).
     * @param targetHeight Ukuran tinggi hasil crop untuk input FR (default: 112).
     * @return Bitmap potongan wajah yang sudah diperjelas dan siap dikirim ke Engine Face Recognition in-house Anda.
     */
    fun enhanceCroppedFace(
        inputBitmap: Bitmap,
        landmarks: FloatArray,
        targetWidth: Int = 112,
        targetHeight: Int = 112
    ): Bitmap? {
        check(isInitialized) { "PhotoEnhancer belum diinisialisasi. Panggil init() terlebih dahulu." }
        require(landmarks.size >= 10) { "Landmarks harus berisi minimal 10 float (5 pasang koordinat [x, y])." }

        return enhanceCroppedFaceNative(inputBitmap, landmarks, targetWidth, targetHeight)
    }

    /**
     * Memperjelas wajah dan menempelkan kembali ke full resolution image (untuk kebutuhan penampilan UI).
     */
    fun enhanceFullImage(inputBitmap: Bitmap, landmarks: FloatArray): Bitmap? {
        check(isInitialized) { "PhotoEnhancer belum diinisialisasi. Panggil init() terlebih dahulu." }
        return enhanceFullImageNative(inputBitmap, landmarks)
    }

    private fun copyAssetToFile(assetName: String, outFile: File) {
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
    }

    // Native JNI method declarations
    private native fun initNative(modelPath: String, useGPU: Boolean): Boolean
    private native fun enhanceCroppedFaceNative(
        inputBitmap: Bitmap,
        landmarks: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap?

    private native fun enhanceFullImageNative(
        inputBitmap: Bitmap,
        landmarks: FloatArray
    ): Bitmap?
}
