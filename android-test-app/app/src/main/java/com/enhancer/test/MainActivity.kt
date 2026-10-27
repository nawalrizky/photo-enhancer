package com.enhancer.test

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import com.enhancer.sdk.AiFilterEnhancer
import com.enhancer.rknn.RknnEnhancer
import com.enhancer.test.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private lateinit var binding: ActivityMainBinding
    private val enhancer = AiFilterEnhancer()
    private val rknnEnhancer by lazy { RknnEnhancer() }
    private val useRknn get() = (intent.getStringExtra("benchmark_backend") ?: "rknn") == "rknn"
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var detector: FaceDetector? = null
    private fun getDetector(): FaceDetector {
        return detector ?: FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setMinFaceSize(0.12f)
                .build()
        ).also { detector = it }
    }

    private var sourceBitmap: Bitmap? = null
    private var resultBitmap: Bitmap? = null
    private var landmarks: FloatArray? = null
    private var modelFile: File? = null
    private var cameraUri: Uri? = null
    private var busyStartedAt = 0L
    private var busyMessage = ""
    private var benchmarkRunsRemaining = 0
    private var benchmarkRunIndex = 0
    private var photoStartedAt = 0L
    private var benchmarkUri: Uri? = null
    private val progressTicker = object : Runnable {
        override fun run() {
            if (binding.busyOverlay.visibility != View.VISIBLE) return
            val elapsedSeconds = (SystemClock.elapsedRealtime() - busyStartedAt) / 1000
            binding.progressElapsed.text = "${elapsedSeconds}s elapsed"
            mainHandler.postDelayed(this, 250)
        }
    }

    private val photoPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let(::loadPhoto)
    }

    private val cameraCapture = registerForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
        val uri = cameraUri
        if (saved && uri != null) {
            loadPhoto(uri)
        } else {
            setBusy(false, "Camera capture cancelled.")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelDirectory = File(filesDir, "models")
        modelFile = File(modelDirectory, "GFPGANv1.4_fp16.onnx").takeIf(File::isFile)
            ?: File(modelDirectory, "GFPGANv1.4.onnx").takeIf(File::isFile)
            ?: File(modelDirectory, "GFPGANv1.4_int8.onnx")
        if (useRknn) modelFile = File(modelDirectory, "GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn")
        updateModelState()

        binding.selectButton.setOnClickListener { photoPicker.launch("image/*") }
        binding.cameraButton.setOnClickListener { capturePhoto() }
        binding.cropButton.visibility = View.GONE
        binding.fullButton.setOnClickListener { enhance() }
        binding.saveButton.setOnClickListener { saveResult() }
        binding.nnapiSwitch.visibility = View.GONE

        intent.getStringExtra("benchmark_image")?.let { path ->
            val image = File(path)
            if (image.isFile && image.canonicalFile.parentFile == filesDir.canonicalFile) {
                benchmarkRunsRemaining = intent.getIntExtra("benchmark_runs", 3).coerceIn(1, 10)
                benchmarkUri = Uri.fromFile(image)
                loadPhoto(requireNotNull(benchmarkUri))
            }
        }
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(progressTicker)
        detector?.close()
        enhancer.close()
        if (useRknn) rknnEnhancer.close()
        inferenceExecutor.shutdown()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        if (::binding.isInitialized && modelFile != null) {
            updateModelState()
            updateButtons()
        }
    }

    private fun updateModelState() {
        val file = modelFile
        if (file?.isFile == true) {
            val variant = when {
                file.extension == "rknn" -> "RKNN / NPU"
                file.name.contains("fp16") -> "FP16"
                file.name.contains("int8") -> "INT8"
                else -> "FP32"
            }
            binding.modelState.text = "MODEL: $variant / ${(file.length() / 1_048_576)} MB / READY"
            binding.modelState.setTextColor(getColor(R.color.acid))
        } else {
            binding.modelState.text = "MODEL: MISSING"
            binding.modelState.setTextColor(0xFFFF8A80.toInt())
            binding.statusText.text = "Push GFPGANv1.4.onnx to ${file?.absolutePath ?: "app external storage"}."
        }
    }

    private fun capturePhoto() {
        val cameraDir = File(cacheDir, "camera").apply { mkdirs() }
        val photoFile = File(cameraDir, "portrait_${System.currentTimeMillis()}.jpg")
        val captureUri = FileProvider.getUriForFile(this, "$packageName.files", photoFile)
        cameraUri = captureUri
        setBusy(true, "Opening camera...")
        cameraCapture.launch(captureUri)
    }

    private fun loadPhoto(uri: Uri) {
        photoStartedAt = System.nanoTime()
        setBusy(true, "Decoding photo...")
        inferenceExecutor.execute {
            runCatching {
                decodeOrientedBitmap(uri, uri == cameraUri)
            }.onSuccess { bitmap ->
                runOnUiThread {
                    val previousSource = sourceBitmap
                    resultBitmap?.takeIf { it !== previousSource }?.recycle()
                    previousSource?.recycle()
                    sourceBitmap = bitmap
                    resultBitmap = null
                    landmarks = null
                    binding.sourceImage.setImageBitmap(bitmap)
                    binding.resultImage.setImageDrawable(null)
                    binding.saveButton.isEnabled = false
                    detectFace(bitmap)
                }
            }.onFailure { showError("Could not decode photo", it) }
        }
    }

    private fun decodeOrientedBitmap(uri: Uri, fromCamera: Boolean): Bitmap {
        val orientation = if (fromCamera) {
            // This device's camera writes upright pixels with stale rotation metadata.
            ExifInterface.ORIENTATION_NORMAL
        } else {
            contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Could not open image" }
                ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            }
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Could not open image" }
            BitmapFactory.decodeStream(input, null, bounds)
        }
        check(bounds.outWidth > 0 && bounds.outHeight > 0) { "Unsupported image format" }

        var sampleSize = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sampleSize * 2) >= MAX_IMAGE_EDGE) {
            sampleSize *= 2
        }
        val decoded = contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Could not open image" }
            BitmapFactory.decodeStream(
                input,
                null,
                BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inSampleSize = sampleSize
                }
            )
        } ?: error("Could not decode image")

        val matrix = Matrix().apply {
            when (orientation) {
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
                ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
                ExifInterface.ORIENTATION_TRANSPOSE -> {
                    setRotate(90f)
                    postScale(-1f, 1f)
                }
                ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
                ExifInterface.ORIENTATION_TRANSVERSE -> {
                    setRotate(-90f)
                    postScale(-1f, 1f)
                }
                ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(-90f)
            }
        }
        val oriented = if (matrix.isIdentity) {
            decoded
        } else {
            Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true).also {
                if (it !== decoded) decoded.recycle()
            }
        }
        val longest = max(oriented.width, oriented.height)
        if (longest <= MAX_IMAGE_EDGE) return oriented

        val scale = MAX_IMAGE_EDGE.toFloat() / longest
        return Bitmap.createScaledBitmap(
            oriented,
            (oriented.width * scale).roundToInt(),
            (oriented.height * scale).roundToInt(),
            true
        ).also {
            if (it !== oriented) oriented.recycle()
        }
    }

    private fun detectFace(bitmap: Bitmap) {
        setBusy(true, "Detecting face and five landmarks offline...")
        val activeDetector = getDetector()
        activeDetector.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { faces ->
                Log.i("EnhancerFaceDiag", "image=${bitmap.width}x${bitmap.height} faces=${faces.size}")
                faces.forEachIndexed { index, candidate ->
                    val available = listOf(
                        FaceLandmark.LEFT_EYE,
                        FaceLandmark.RIGHT_EYE,
                        FaceLandmark.NOSE_BASE,
                        FaceLandmark.MOUTH_LEFT,
                        FaceLandmark.MOUTH_RIGHT
                    ).map { candidate.getLandmark(it) != null }
                    Log.i(
                        "EnhancerFaceDiag",
                        "face=$index bounds=${candidate.boundingBox.flattenToString()} " +
                            "landmarks=$available yaw=${candidate.headEulerAngleY} roll=${candidate.headEulerAngleZ}"
                    )
                }
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                val detected = face?.toLandmarkArray()
                if (face == null || detected == null) {
                    landmarks = null
                    setBusy(false, "No face with all five landmarks detected. Try a clear frontal portrait.")
                } else {
                    landmarks = detected
                    setBusy(
                        false,
                        "Face ready at ${face.boundingBox.width()}x${face.boundingBox.height()} px. Select crop or full-image enhancement."
                    )
                }
                activeDetector.close()
                if (detector === activeDetector) detector = null
                updateButtons()
                if (benchmarkRunsRemaining > 0 && landmarks != null) enhance()
            }
            .addOnFailureListener {
                Log.e("EnhancerFaceDiag", "Face detector failed", it)
                activeDetector.close()
                if (detector === activeDetector) detector = null
                showError("Face detection failed", it)
            }
    }

    private fun Face.toLandmarkArray(): FloatArray? {
        val types = intArrayOf(
            FaceLandmark.LEFT_EYE,
            FaceLandmark.RIGHT_EYE,
            FaceLandmark.NOSE_BASE,
            FaceLandmark.MOUTH_LEFT,
            FaceLandmark.MOUTH_RIGHT
        )
        val points = types.map { getLandmark(it)?.position ?: return null }
        return FloatArray(10) { index ->
            val point = points[index / 2]
            if (index % 2 == 0) point.x else point.y
        }
    }

    private fun enhance() {
        val bitmap = sourceBitmap ?: return
        val faceLandmarks = landmarks ?: return
        val model = modelFile
        if (model?.isFile != true) {
            updateModelState()
            return
        }

        resultBitmap?.takeIf { it !== bitmap }?.recycle()
        resultBitmap = null
        binding.resultImage.setImageDrawable(null)
        binding.saveButton.isEnabled = false

        setBusy(true, "Enhancing and pasting face into source...")
        val started = System.nanoTime()
        inferenceExecutor.execute {
            runCatching {
                val initStarted = System.nanoTime()
                if (useRknn) rknnEnhancer.initialize(model) else enhancer.initialize(model)
                val initMs = (System.nanoTime() - initStarted) / 1_000_000
                val output = if (useRknn) rknnEnhancer.enhanceFull(bitmap, faceLandmarks)
                    else enhancer.enhanceFull(bitmap, faceLandmarks)
                val totalMs = (System.nanoTime() - started) / 1_000_000
                Log.i("EnhancerBench", "backend=${if (useRknn) "rknn" else "cpu"} init_ms=$initMs total_ms=$totalMs image=${bitmap.width}x${bitmap.height}")
                if (benchmarkRunsRemaining > 0) {
                    benchmarkRunIndex++
                    if (benchmarkRunIndex == 1 || intent.getBooleanExtra("benchmark_full_pipeline", false)) {
                        val pipelineMs = (System.nanoTime() - photoStartedAt) / 1_000_000
                        Log.i("EnhancerPipeline", "run=$benchmarkRunIndex decode_detect_enhance_ms=$pipelineMs")
                    }
                    if (intent.getBooleanExtra("benchmark_export", false)) {
                        val directory = File(filesDir, "benchmark-output").apply { mkdirs() }
                        File(directory, "run-$benchmarkRunIndex.png").outputStream().use {
                            check(output.compress(Bitmap.CompressFormat.PNG, 100, it))
                        }
                        if (benchmarkRunIndex == 1) {
                            File(directory, "source.png").outputStream().use {
                                check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it))
                            }
                            val record = JSONObject().put("width", bitmap.width).put("height", bitmap.height)
                                .put("landmarks", JSONArray(faceLandmarks.toList()))
                            File(directory, "source.json").writeText(record.toString())
                        }
                    }
                }
                output
            }.onSuccess { result ->
                val elapsedMs = (System.nanoTime() - started) / 1_000_000
                runOnUiThread {
                    resultBitmap = result
                    binding.resultImage.setImageBitmap(result)
                    setBusy(false, "Completed locally in ${elapsedMs} ms. Output: ${result.width}x${result.height}.")
                    updateButtons()
                    updateModelState()
                    if (benchmarkRunsRemaining > 0) {
                        benchmarkRunsRemaining--
                        if (benchmarkRunsRemaining > 0) {
                            if (intent.getBooleanExtra("benchmark_full_pipeline", false)) {
                                loadPhoto(requireNotNull(benchmarkUri))
                            } else enhance()
                        }
                    }
                }
            }.onFailure { showError("Enhancement failed", it) }
        }
    }

    private fun saveResult() {
        val bitmap = resultBitmap ?: return
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "enhanced_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/EnhancerLab")
        }
        runCatching {
            val uri = requireNotNull(contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values))
            contentResolver.openOutputStream(uri).use { output ->
                checkNotNull(output)
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output))
            }
        }.onSuccess {
            binding.statusText.text = "Result saved to Pictures/EnhancerLab."
        }.onFailure { showError("Could not save result", it) }
    }

    private fun updateButtons() {
        val canEnhance = sourceBitmap != null && landmarks != null && modelFile?.isFile == true
        binding.cropButton.isEnabled = canEnhance
        binding.fullButton.isEnabled = canEnhance
        binding.saveButton.isEnabled = resultBitmap != null
    }

    private fun setBusy(busy: Boolean, message: String) {
        if (busy) {
            if (binding.busyOverlay.visibility != View.VISIBLE) {
                busyStartedAt = SystemClock.elapsedRealtime()
            }
            busyMessage = message
            binding.progressMessage.text = busyMessage
            binding.busyOverlay.visibility = View.VISIBLE
            mainHandler.removeCallbacks(progressTicker)
            progressTicker.run()
        } else {
            binding.busyOverlay.visibility = View.GONE
            mainHandler.removeCallbacks(progressTicker)
        }
        binding.selectButton.isEnabled = !busy
        binding.cameraButton.isEnabled = !busy
        binding.cropButton.isEnabled = !busy && sourceBitmap != null && landmarks != null && modelFile?.isFile == true
        binding.fullButton.isEnabled = binding.cropButton.isEnabled
        binding.saveButton.isEnabled = !busy && resultBitmap != null
        binding.statusText.text = message
    }

    private fun showError(prefix: String, error: Throwable) {
        runOnUiThread {
            setBusy(false, "$prefix: ${error.message ?: error.javaClass.simpleName}")
            updateButtons()
        }
    }

    companion object {
        private const val MAX_IMAGE_EDGE = 2048
    }
}
