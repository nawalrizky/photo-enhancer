package com.enhancer.integration

import android.content.Context
import java.io.File
import java.security.MessageDigest

/** Invoke once during provisioning/startup on a worker thread. */
object ModelInstaller {
    const val NAME = "GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn"
    private const val BYTES = 458741110L
    private const val SHA256 = "ac5cb93f894599e26d4a3ea71f0b546073678c86016f15ca21b1efcadc4b7a1c"

    fun installFromAssets(context: Context): File {
        val directory = File(context.filesDir, "models")
        check(directory.isDirectory || directory.mkdirs()) { "Could not create model directory" }
        val target = File(directory, NAME)
        if (target.isFile && valid(target)) return target
        val temporary = File(directory, "$NAME.installing")
        try {
            context.assets.open(NAME).use { input ->
                temporary.outputStream().use { output -> input.copyTo(output, 1024 * 1024) }
            }
            check(valid(temporary)) { "RKNN model size/checksum mismatch" }
            check(temporary.renameTo(target)) { "Could not install verified RKNN model" }
            return target
        } finally {
            if (temporary.exists()) temporary.delete()
        }
    }

    private fun valid(file: File): Boolean {
        if (file.length() != BYTES) return false
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count == -1) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) } == SHA256
    }
}
