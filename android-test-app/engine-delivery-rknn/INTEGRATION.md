# Integrasi SDK RKNN ke aplikasi Android sebenarnya

## 1. Tambahkan AAR

Letakkan `ai-filter-enhancer-rknn-1.1.0.aar` di `app/libs/`:

```kotlin
dependencies {
    implementation(files("libs/ai-filter-enhancer-rknn-1.1.0.aar"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.9.24")
}

android {
    defaultConfig {
        minSdk = 28
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
```

Jika aplikasi Kotlin sudah membawa stdlib, selaraskan dengan versi Kotlin aplikasi.
AAR file tidak membawa dependency Maven secara transitif. Untuk backend ini,
tidak perlu menambahkan dependency ONNX Runtime, RKNN Toolkit, ML Kit, atau OpenCV
Java. Jika aplikasi membutuhkan deteksi wajah, dependency detektornya tetap milik aplikasi.

AAR berisi `librknnrt.so`, `libopencv_java4.so`, `libc++_shared.so`, dan library
engine RKNN pada ABI arm64-v8a. Jika aplikasi sudah menyertakan library native
bernama sama, selaraskan versinya sebelum memilih salah satu salinan. Jangan
memilih `pickFirst` secara acak untuk runtime dengan versi berbeda.

Aturan consumer ProGuard/R8 sudah ada di AAR untuk mempertahankan kelas dan metode
JNI. Contoh consumer memakai R8/minification untuk memeriksa aturan tersebut.

## 2. Sediakan model sebagai file

Gunakan **model dari paket ini**:

```text
GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn
SHA256: ac5cb93f894599e26d4a3ea71f0b546073678c86016f15ca21b1efcadc4b7a1c
Bytes: 458741110
```

SDK menerima `File`, bukan nama asset/URI. Model dapat dipasang melalui deployment
perangkat atau disertakan dalam assets lalu disalin ke `filesDir/models/`.
`examples/ModelInstaller.kt` menyediakan contoh copy dengan verifikasi checksum.
Jalankan instalasi model pada worker thread dan sebelum benchmark pemrosesan foto.

Model berasal dari GFPGAN v1.4 ONNX FP32 dengan rewrite bias RGB yang sudah diuji
ekuivalen. Komputasi besar pada NPU memakai FP16. Jangan memakai dua kandidat RKNN
awal yang belum mendapat perbaikan bias warna, maupun mengganti model ini dengan ONNX.

SDK tidak mengunduh model dan tidak mendeklarasikan permission Internet.
Pemrosesan dapat offline sejak penggunaan pertama apabila model sudah disediakan.

## 3. Panggil API pada worker thread

```kotlin
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import com.enhancer.rknn.RknnEnhancer
import java.io.File
import java.util.concurrent.Executors

val worker = Executors.newSingleThreadExecutor()
val main = Handler(Looper.getMainLooper())
val enhancer = RknnEnhancer()

fun enhancePhoto(
    modelFile: File,
    bitmap: Bitmap,
    landmarks: FloatArray,
    onResult: (Bitmap) -> Unit,
    onError: (Throwable) -> Unit
) {
    worker.execute {
        try {
            enhancer.initialize(modelFile) // Sesi yang sama dipakai ulang.
            val result = enhancer.enhanceFull(bitmap, landmarks)
            main.post { onResult(result) }
        } catch (error: Throwable) {
            main.post { onError(error) }
        }
    }
}

// Panggil ketika pemilik engine benar-benar selesai, bukan setelah setiap foto.
// Hentikan pengiriman pekerjaan baru sebelum menjalankan blok ini.
fun releaseEnhancer() {
    worker.execute { enhancer.close() }
    worker.shutdown()
}
```

`initialize` dan `enhanceFull` bersifat synchronous. Jangan menjalankannya di
main thread. Panggilan pada instance yang sama diserialisasi. Pembatalan coroutine
atau Future tidak menghentikan inferensi native yang sedang berjalan.

Sesi model tidak bergantung pada isi foto. Setelah inisialisasi, foto berbeda
dapat menggunakan sesi yang sama. Buat satu engine untuk layanan enhancement;
membuat beberapa instance sekaligus akan menggandakan kebutuhan memori.

## 4. Kontrak bitmap dan landmark

- Bitmap belum di-recycle dan memakai `Bitmap.Config.ARGB_8888`.
- Terapkan orientasi EXIF dan resize terlebih dahulu; sisi terpanjang maksimum 2048 px.
- Deteksi/konversi landmark pada koordinat bitmap final tersebut.
- Array memiliki sepuluh angka finite dan berada di dalam batas bitmap:

```text
[leftEyeX, leftEyeY,
 rightEyeX, rightEyeY,
 noseBaseX, noseBaseY,
 mouthLeftX, mouthLeftY,
 mouthRightX, mouthRightY]
```

Untuk ML Kit, urutannya adalah `LEFT_EYE`, `RIGHT_EYE`, `NOSE_BASE`, `MOUTH_LEFT`,
`MOUTH_RIGHT`, sama dengan aplikasi uji. Pilih satu wajah jika gambar memiliki
beberapa wajah. SDK ini melakukan enhancement satu wajah per panggilan.

Output adalah bitmap baru dengan dimensi yang sama. Source tidak dimodifikasi
atau di-recycle SDK. Pertahankan source hidup sampai panggilan selesai; aplikasi
pemanggil bertanggung jawab mengelola/recycle bitmap hasil.

## 5. Error dan diagnosis

- `IllegalArgumentException`: file tidak tersedia/bukan `.rknn`, bitmap atau landmark tidak valid.
- `IllegalStateException`: belum initialize, atau model/runtime tidak dapat diinisialisasi.
- `RuntimeException`: inferensi native gagal.
- `UnsatisfiedLinkError`: library native tidak terkemas atau ABI/runtime tidak sesuai.

Periksa `logcat` tag `ExperimentalRknn` untuk durasi tahap dan error native.
Nama internal log/library masih berasal dari backend yang diuji; API distribusi
adalah `RknnEnhancer`. Jika aplikasi masih memakai `AiFilterEnhancer` dari AAR CPU
lama, model/backend lama tetap dapat menghasilkan waktu sekitar 14 detik.

## 6. Contoh consumer dengan package berbeda

`consumer-example/` hanya memakai file AAR di atas. Package-nya
`com.enhancer.integrationcheck`; build release memakai minification/R8 dan
signing debug khusus demonstrasi. Jangan gunakan identitas/signing demo untuk
release aplikasi sebenarnya.

Build dari direktori `consumer-example` dengan `./gradlew :app:assembleRelease`.
Konfigurasikan Android SDK lokal melalui `local.properties` atau lingkungan build Anda.

Model demo ditulis ke storage privat melalui provider deployment yang hanya
mengizinkan pemanggil dengan permission sistem `DUMP` (misalnya ADB shell).
Provider ini hanya milik contoh consumer; AAR SDK tidak menambahkan provider.
Untuk aplikasi sebenarnya, gunakan provisioning aplikasi Anda atau `ModelInstaller`.

```sh
adb install -r consumer-example/app/build/outputs/apk/release/app-release.apk
adb push GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn /data/local/tmp/model.rknn
adb shell 'cat /data/local/tmp/model.rknn | content write --uri content://com.enhancer.integrationcheck.provisioning/model'
adb shell am force-stop com.enhancer.integrationcheck
adb shell am start -n com.enhancer.integrationcheck/.MainActivity
adb logcat -s RknnConsumer:I ExperimentalRknn:I AndroidRuntime:E
```

Consumer memakai landmark khusus foto sample yang disertakan dan menjalankan
empat calls memakai satu sesi. Landmark sample tidak boleh dipakai untuk foto lain.
Pada aplikasi sebenarnya, gunakan detektor wajah aplikasi Anda.
