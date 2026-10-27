# Ringkasan Akhir AI Filter Enhancer SDK

## Ringkasan Satu Menit

Library yang dihasilkan adalah **SDK Android offline untuk memperbaiki kualitas satu wajah pada sebuah foto menggunakan GFPGAN v1.4**. SDK menerima bitmap beserta lima titik landmark wajah, menjalankan model melalui native C++ dan ONNX Runtime, lalu menempelkan wajah yang sudah diperbaiki kembali ke foto dengan ukuran asli.

Hasil akhirnya sudah berbentuk artefak integrasi nyata:

- `ai-filter-enhancer-1.0.0.aar` sebagai library Android;
- `GFPGANv1.4_fp16.onnx` sebagai model terpisah;
- API Kotlin `AiFilterEnhancer`;
- native engine C++/JNI untuk alignment, inference, dan paste-back;
- integration guide, R8 rules, third-party notices, dan SHA-256 checksums;

Konfigurasi final menggunakan **GFPGAN FP16, ONNX Runtime CPU, empat thread, dan sequential execution**. Pada perangkat uji Android 14 dengan RK3576 dan RAM 8 GB, proses penuh selesai sekitar **18,5 detik**, tidak mengalami native crash, dan menghasilkan perbedaan numerik yang sangat kecil terhadap model FP32 (`MAE 0,000272`). Keterbatasan utamanya adalah peak memory masih tinggi, sekitar **708 MB**, dan SDK hanya memproses satu wajah tanpa background upscaling.

## Masalah yang Diselesaikan

Pipeline awal hanya berjalan melalui Python/PyTorch dan bergantung pada GFPGAN/facexlib serta optional Real-ESRGAN. Bentuk tersebut tidak ideal untuk langsung dipasang ke aplikasi Android karena:

- runtime PyTorch dan modelnya besar;
- graph GFPGAN sulit dikonversi secara naif;
- proses deteksi, alignment, inference, dan paste-back perlu dijembatani ke Android Bitmap;
- inference harus dijalankan di luar main thread;
- penggunaan memory pada perangkat ARM perlu dikendalikan;
- model harus dapat digunakan secara offline.

Hasil pekerjaan ini memindahkan bagian face enhancement ke SDK Android native dengan kontrak API yang jelas dan artefak AAR yang siap diintegrasikan.

## Ruang Lingkup Hasil Final

### Yang sudah selesai

- Integrasi Android melalui AAR versi `1.0.0`.
- Dukungan Android API 28 ke atas.
- Dukungan ABI `arm64-v8a`.
- Inference GFPGAN v1.4 secara offline setelah model tersedia di device.
- Model FP16 dengan input dan output tensor tetap `float32`.
- Input berupa `Bitmap.Config.ARGB_8888` dan lima facial landmarks.
- Alignment wajah ke resolusi model `512x512`.
- Preprocessing RGB, NCHW, dan normalisasi ke rentang `[-1, 1]`.
- Inference melalui native C++ dan ONNX Runtime.
- Inverse affine transform dan paste-back memakai feathered elliptical mask.
- Output berupa bitmap baru dengan dimensi dan aspect ratio yang sama dengan input.
- Validasi input pada layer Kotlin dan native.
- Pengelolaan lifecycle native menggunakan handle per instance dan `AutoCloseable`.
- Pemanggilan pada satu instance diserialisasi agar tidak terjadi concurrent access ke session yang sama.
- Test app dengan input gallery/camera, koreksi orientasi EXIF, ML Kit landmark detection, dan penyimpanan JPEG.
- R8/resource shrinking dan consumer ProGuard rules.
- Paket delivery lengkap beserta dokumentasi dan integrity hash.

### Yang tidak termasuk dalam hasil final

- Background enhancement atau super-resolution dengan Real-ESRGAN.
- Output x2 atau x4; output tetap berukuran sama dengan input.
- Face detector di dalam SDK; detector menjadi tanggung jawab host app.
- Pemrosesan banyak wajah dalam satu pemanggilan.
- RestoreFormer atau CodeFormer.
- Face-parsing mask yang identik dengan pipeline Python.
- Dukungan ABI selain `arm64-v8a`.
- Real-time atau video enhancement.
- Automatic model download/update.

## Arsitektur Final

```text
Host Android App
    |
    | Bitmap ARGB_8888 + 5 facial landmarks
    v
AiFilterEnhancer (Kotlin API)
    |
    | validasi input, lifecycle, sinkronisasi
    v
JNI Bridge
    |
    | konversi Android Bitmap <-> OpenCV Mat
    v
Native C++ Engine
    |
    |-- affine alignment ke 512x512
    |-- RGB/NCHW/[-1,1] preprocessing
    |-- ONNX Runtime CPU inference
    |-- output tensor ke image
    |-- inverse warp
    `-- feathered paste-back
    |
    v
Bitmap ARGB_8888 baru, ukuran sama dengan input
```

Komponen final berada di:

- SDK Kotlin: `android-test-app/enhancer-sdk/src/main/java/com/enhancer/sdk/AiFilterEnhancer.kt`
- Native engine: `android-test-app/enhancer-sdk/src/main/cpp/ai_filter_enhancer.cpp`
- JNI bridge: `android-test-app/enhancer-sdk/src/main/cpp/jni_bridge.cpp`
- Build native: `android-test-app/enhancer-sdk/src/main/cpp/CMakeLists.txt`
- Artefak delivery: `android-test-app/engine-delivery/`

## Alur Pemrosesan

1. Host app menyediakan model `GFPGANv1.4_fp16.onnx` pada filesystem biasa.
2. Host membuat satu instance `AiFilterEnhancer` dan memanggil `initialize(modelFile)`.
3. Host mendeteksi wajah dan menghasilkan lima landmark: mata kiri, mata kanan, hidung, sudut mulut kiri, dan sudut mulut kanan.
4. Host memanggil `enhanceFull(source, landmarks)` di background thread.
5. JNI membaca bitmap Android dan mengubahnya menjadi OpenCV BGR matrix.
6. Native engine menghitung affine transform dari landmark ke template GFPGAN.
7. Wajah di-warp menjadi input `512x512`.
8. Pixel diubah menjadi tensor RGB planar `[1, 3, 512, 512]`, lalu dinormalisasi ke `[-1, 1]`.
9. ONNX Runtime menjalankan GFPGAN.
10. Tensor output diubah kembali menjadi image 8-bit.
11. Transform dibalik untuk menempatkan wajah ke posisi asal.
12. Wajah hasil enhancement dibaurkan dengan foto menggunakan feathered elliptical mask.
13. SDK mengembalikan bitmap baru dengan ukuran yang sama seperti input.
14. Host memanggil `close()` ketika engine tidak lagi digunakan.

## Public API

```kotlin
val enhancer = AiFilterEnhancer()
enhancer.initialize(modelFile)

val output: Bitmap = enhancer.enhanceFull(
    source = inputBitmap,
    landmarks = floatArrayOf(
        leftEyeX, leftEyeY,
        rightEyeX, rightEyeY,
        noseX, noseY,
        mouthLeftX, mouthLeftY,
        mouthRightX, mouthRightY
    )
)

enhancer.close()
```

Kontrak input:

- bitmap belum di-recycle;
- format `ARGB_8888`;
- orientasi EXIF sudah diterapkan;
- sisi terpanjang maksimal 2048 px;
- tepat sepuluh nilai landmark yang finite dan berada dalam batas bitmap;
- landmark menggunakan sistem koordinat bitmap final.

Karena `enhanceFull` bersifat synchronous dan CPU-bound, fungsi ini wajib dipanggil dari background thread. Membatalkan coroutine tidak menghentikan inference yang sudah masuk ke native code.

## Keputusan Teknis Utama

### 1. ONNX Runtime dipilih sebagai runtime

ONNX Runtime dipakai karena model GFPGAN ONNX yang kompatibel tersedia dan dapat dijalankan dari C++ pada Android. Pendekatan ini mengurangi risiko dibanding menulis operator model sendiri atau membawa seluruh runtime PyTorch ke aplikasi.

### 2. FP16 dipilih sebagai model produksi

Model FP16 mengurangi ukuran model dari sekitar 324,5 MiB menjadi 162,4 MiB. Model tetap menerima dan mengeluarkan tensor `float32`, sehingga kontrak preprocessing dan postprocessing tidak berubah.

Hasil perbandingan FP16 terhadap FP32:

| Metrik | Hasil | Keputusan |
| --- | ---: | --- |
| Ukuran model | 170.260.352 byte | 50% lebih kecil |
| MAE | 0,000272 | Lulus batas 0,02 |
| RMSE | 0,000377 | Lulus |
| P99 absolute error | 0,000876 | Lulus |

### 3. INT8 tidak dijadikan default

QUInt8 berhasil dijalankan pada ARM, tetapi tidak memenuhi target kualitas dan ukuran:

- ukuran masih 150,9 MiB, bukan sekitar 85 MB;
- MAE close-face `0,054501`, melewati batas `0,05`;
- terdapat perbedaan grain, edge texture, dan tone;
- peningkatan kecepatan tidak signifikan.

Karena itu, INT8 dipertahankan sebagai eksperimen, bukan konfigurasi produksi.

### 4. CPU dipilih dibanding NNAPI

NNAPI hanya dapat menangani sebagian kecil graph dan sisanya kembali ke CPU. Pada perangkat uji, hasilnya lebih lambat dan memakai memory lebih besar. XNNPACK juga tidak menyelesaikan inference dalam 40 detik. Konfigurasi CPU empat thread menjadi pilihan paling stabil.

### 5. Direct 8-bit blending dipakai untuk paste-back

Paste-back awal menggunakan beberapa full-frame float buffer. Implementasi final melakukan blending langsung pada `CV_8UC3`, sehingga mengurangi temporary buffer dan peak memory secara signifikan tanpa mengubah model inference.

### 6. Model dipisahkan dari AAR

AAR berisi API dan native libraries, sedangkan model sekitar 162,4 MiB diletakkan terpisah. Host app dapat memilih install-time asset atau mekanisme provisioning sendiri. Setelah model tersedia, inference berjalan offline dan SDK tidak memerlukan permission internet.

## Hasil Pengujian Final

Perangkat validasi:

- Android 14;
- SoC RK3576;
- RAM 8 GB;
- ABI `arm64-v8a`.

| Kriteria | Hasil | Status |
| --- | ---: | --- |
| Full enhancement release | 18,506 detik | Lulus target 20-22 detik |
| FP16 vs FP32 MAE | 0,000272 | Lulus |
| Native crash | Tidak ada | Lulus |
| Post-run RSS | 417.644 KB | Lulus target 500 MB |
| True peak RSS/VmHWM | 708.420 KB | Belum lulus target 500 MB |
| Runtime network access | Tidak ada | Lulus |

`VmHWM` adalah acuan peak memory yang lebih tepat daripada sampling `dumpsys`, karena sampling dapat melewatkan lonjakan singkat selama ONNX Runtime mengalokasikan workspace.

## Pencapaian Utama

1. Pipeline yang sebelumnya bergantung pada Python/PyTorch sudah memiliki delivery Android berupa AAR.
2. Sulitnya model conversion ditangani dengan model ONNX yang kemudian dikonversi dan divalidasi ke FP16.
3. Kontrak API dibuat kecil dan jelas: initialize, enhance, close.
4. Boundary Kotlin/JNI/C++ memiliki validasi dan lifecycle yang lebih aman daripada prototipe awal yang memakai global raw pointer.
5. Ukuran model turun 50% dengan perbedaan output yang sangat kecil.
6. Peak memory turun dari sekitar 1,41 GB pada baseline menjadi sekitar 708 MB.
7. Latency berada di bawah batas penerimaan 20-22 detik pada perangkat target.
8. Artefak release telah diuji melalui host app dengan R8/minification aktif.
9. Library dapat bekerja offline dan tidak menambahkan Android permission.
10. Paket delivery memiliki dokumentasi integrasi, attribution, dan checksum.

## Keterbatasan dan Risiko

### Peak memory

Peak memory 708 MB masih lebih tinggi dari target 500 MB. Optimasi image blending sudah dilakukan; sumber memory terbesar berikutnya adalah activation/workspace graph GFPGAN di ONNX Runtime. Perbaikan lanjutan memerlukan perubahan graph/model, operator runtime, atau hardware execution provider dengan graph coverage yang lebih baik.

### Latency

Sekitar 18 detik cocok untuk fitur user-triggered seperti tombol "Enhance", tetapi tidak cocok untuk preview real-time atau video. UI host harus menampilkan progress/loading dan mencegah pemanggilan di main thread.

### Model dan footprint

Model FP16 berukuran sekitar 162,4 MiB dan tidak berada di dalam AAR. Strategi distribusi model perlu diputuskan oleh host app. Membuat beberapa instance SDK juga akan menggandakan session dan workspace memory.

### Parity dengan Python

SDK Android bukan port satu-banding-satu dari seluruh pipeline Python. Perbedaannya:

- Python dapat memproses banyak wajah; SDK memproses satu wajah.
- Python dapat memakai Real-ESRGAN x2/x4; SDK tidak melakukan background upscaling.
- Python menggunakan detector dan face parsing dari facexlib; SDK menerima landmark dari host dan memakai feathered ellipse.
- Python dapat menawarkan model restorasi lain; SDK final hanya GFPGAN.

Yang sudah divalidasi secara numerik adalah **FP16 ONNX terhadap FP32 ONNX**, bukan seluruh output end-to-end Android terhadap pipeline Python.

### Dukungan perangkat

Artefak final hanya mendukung `arm64-v8a` dan minimum Android API 28. Pengujian performa utama baru terdokumentasi pada RK3576, sehingga device matrix yang lebih luas masih diperlukan sebelum rollout besar.

### Testing

Pengujian saat ini mencakup script validasi model, benchmark perangkat, visual review, dan runtime smoke test. Belum tersedia automated unit/instrumentation regression suite untuk API, JNI, output visual, dan berbagai tipe perangkat.

## Status Komponen dalam Repository

| Komponen | Status | Catatan |
| --- | --- | --- |
| `enhancer-sdk/` | Final/aktif | Sumber SDK yang menghasilkan `libai_filter_enhancer.so` |
| `engine-delivery/` | Final/delivery | AAR, model, guide, notices, dan checksum |
| `app/` | Test host aktif | Menggunakan dependency `:enhancer-sdk` |
| `app/src/main/cpp/` | Historis | Implementasi profiling lama, bukan jalur Gradle aktif |
| root `native/` dan `android/` | Prototipe | Eksperimen awal, bukan library final |
| root `export/export_onnx.py` | Eksperimental | Bukan sumber model ONNX final yang tervalidasi |
| Python `enhancer/` | Referensi awal | Pipeline GFPGAN + optional Real-ESRGAN |

Pembedaan ini penting saat menjelaskan proyek agar implementasi prototipe tidak dianggap sebagai API produksi.

## Cara Menjelaskan kepada Senior

### Versi singkat

> Saya membuat SDK Android offline untuk face enhancement berbasis GFPGAN. SDK dikemas sebagai AAR dan menggunakan Kotlin untuk public API, JNI/C++ untuk image processing, serta ONNX Runtime untuk inference. Host app memberikan bitmap dan lima landmark wajah, lalu SDK melakukan alignment ke 512x512, inference GFPGAN, inverse warp, dan paste-back ke bitmap dengan ukuran asli. Model final memakai FP16 sehingga ukurannya turun 50% dengan MAE 0,000272 terhadap FP32. Pada RK3576 proses selesai sekitar 18,5 detik tanpa native crash. Scope finalnya satu wajah, CPU, arm64, dan tanpa Real-ESRGAN. Risiko yang masih terbuka adalah peak memory sekitar 708 MB dan belum adanya device-matrix serta automated regression test yang luas.

### Penjelasan keputusan desain

> Saya sengaja mempersempit scope dari pipeline Python penuh menjadi face-only SDK. Real-ESRGAN, multi-face, dan detector internal tidak dimasukkan karena akan menaikkan latency, memory, ukuran artefak, dan kompleksitas integrasi. Dengan kontrak input lima landmark, SDK tetap modular: host bebas memilih detector, sedangkan library fokus pada bagian yang paling sulit, yaitu alignment, GFPGAN inference, dan paste-back native.

### Penjelasan kenapa belum real-time

> GFPGAN menggunakan graph besar dengan input wajah tetap 512x512. Bottleneck utamanya adalah inference workspace, bukan ukuran foto penuh. Karena itu tiling foto tidak menyelesaikan masalah dan target yang realistis adalah enhancement foto berdasarkan aksi user, bukan video real-time.

## Pertanyaan yang Mungkin Diajukan

### Apakah ini benar-benar library, bukan hanya demo app?

Ya. Deliverable utamanya adalah `ai-filter-enhancer-1.0.0.aar` dengan public Kotlin API, native libraries, consumer rules, integration guide, dan checksum. Test app hanya berfungsi sebagai host untuk membuktikan integrasi.

### Kenapa detector tidak dimasukkan ke SDK?

Agar SDK tetap modular dan tidak memaksakan dependency ML Kit atau detector tertentu kepada host. SDK hanya membutuhkan lima landmark dalam koordinat bitmap final.

### Kenapa memakai ONNX Runtime?

Karena tersedia model GFPGAN ONNX yang dapat dijalankan melalui C++ Android. Runtime ini memberi jalur implementasi yang lebih rendah risiko dibanding membawa PyTorch penuh atau melakukan port semua operator secara manual.

### Kenapa model FP16 tetapi input/output float32?

Bobot dan internal tensor diperkecil ke FP16 untuk mengurangi ukuran dan memory, sedangkan interface tensor tetap float32 agar preprocessing/postprocessing dan native contract tetap sederhana serta kompatibel.

### Kenapa INT8 tidak dipakai jika lebih kecil?

INT8 memang dapat dijalankan, tetapi model masih 150,9 MiB, kualitas melewati threshold MAE, dan tidak memberi percepatan yang berarti. FP16 memberikan trade-off kualitas dan ukuran yang lebih aman.

### Kenapa tidak memakai NNAPI?

Driver pada perangkat target hanya menangani sebagian kecil node graph. Sebagian besar operasi fallback ke CPU, sehingga latency dan memory justru meningkat.

### Apakah hasilnya sama dengan Python?

Core modelnya sama-sama GFPGAN v1.4, tetapi pipeline end-to-end tidak identik. Android hanya memproses satu wajah, memakai landmark dari host, tidak menjalankan Real-ESRGAN, dan memakai mask paste-back yang lebih sederhana. Klaim parity yang sudah terbukti adalah FP16 ONNX terhadap FP32 ONNX.

### Apa langkah berikutnya yang paling penting?

1. Tambahkan automated API/JNI/instrumentation regression tests.
2. Jalankan benchmark pada beberapa SoC arm64 kelas low, mid, dan high-end.
3. Tentukan strategi distribusi model di host app.
4. Evaluasi graph/model yang lebih ringan jika target peak memory di bawah 500 MB wajib.
5. Tambahkan cancellation/progress contract jika diperlukan oleh UX produksi.
6. Sinkronkan README lama dengan SDK final agar tidak ada dokumentasi yang bertentangan.

## Kesimpulan

Proyek ini sudah melewati tahap riset dan proof of concept. Hasil akhirnya adalah **SDK Android AAR yang nyata, dapat diintegrasikan, berjalan offline, dan telah divalidasi pada perangkat ARM** untuk kontrak yang spesifik: memperbaiki satu wajah dengan GFPGAN dan menempelkannya kembali ke foto tanpa mengubah ukuran gambar.

Keberhasilan utamanya adalah membawa model besar dan pipeline image processing dari ekosistem Python ke boundary Kotlin/JNI/C++ yang terukur, menurunkan ukuran model 50%, menjaga kualitas mendekati FP32, dan mencapai latency penerimaan tanpa native crash. Library belum dapat disebut sebagai port lengkap pipeline Python atau solusi real-time; peak memory, cakupan device, dan regression testing masih menjadi pekerjaan lanjutan yang perlu disampaikan secara transparan.
