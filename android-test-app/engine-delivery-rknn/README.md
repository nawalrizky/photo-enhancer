# SDK Android RKNN 1.1.0

Paket untuk aplikasi Android pada perangkat RK3576 yang sama dengan pengujian.

## Isi paket

| File | Kegunaan |
| --- | --- |
| `ai-filter-enhancer-rknn-1.1.0.aar` | Library Android; sudah berisi RKNN Runtime, OpenCV dan libc++ untuk arm64-v8a |
| `GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn` | Model final yang lolos uji kualitas, sekitar 459 MB |
| `INTEGRATION.md` | Cara menambahkan SDK dan memanggil API |
| `examples/ModelInstaller.kt` | Contoh instalasi model dari assets ke penyimpanan privat dengan verifikasi SHA256 |
| `consumer-example/` | Aplikasi mandiri dengan package berbeda; hanya bergantung pada AAR file, Kotlin stdlib dan Android SDK |
| `MODEL_METADATA.json`, `SHA256SUMS` | Identitas model/runtime dan pemeriksaan berkas |
| `VALIDATION.md` | Hasil pengujian paket AAR ini |
| `licenses/` | Pemberitahuan/lisensi komponen upstream |

## Mulai integrasi

1. Salin AAR ke folder `app/libs/` aplikasi sebenarnya.
2. Tambahkan dependency sesuai `INTEGRATION.md`.
3. Sediakan model `.rknn` sebagai file yang dapat dibaca aplikasi.
4. Gunakan `com.enhancer.rknn.RknnEnhancer` pada worker thread.
5. Pertahankan satu instance/session untuk foto berikutnya; `close()` ketika layanan enhancement selesai digunakan.

Library ini langsung menggunakan RKNN. Tidak ada pemilihan backend melalui intent,
ketergantungan pada `com.enhancer.test`, atau fallback diam-diam ke CPU ONNX.
Beberapa operasi kecil di dalam graph RKNN tetap memakai CPU, sebagaimana profil yang diuji.

## Perbedaan dari SDK CPU 1.0.0

- Import berubah dari `com.enhancer.sdk.AiFilterEnhancer` ke `com.enhancer.rknn.RknnEnhancer`.
- Model berubah dari `.onnx` menjadi `.rknn` yang disertakan dalam paket ini.
- Metode utama tetap `initialize(File)`, `enhanceFull(Bitmap, FloatArray)`, dan `close()`.
- AAR sudah membawa runtime; ONNX Runtime tidak diperlukan untuk backend baru ini.
- Aplikasi pemanggil tetap menyediakan lima landmark wajah pada koordinat bitmap final.

Diuji pada Android 14 / RK3576 / arm64-v8a, driver NPU 0.9.8. Minimum SDK library
28; versi Android lain belum divalidasi di perangkat fisik. Tidak ditujukan untuk
menjanjikan kompatibilitas NPU pada chipset Android lain.

Baseline empat foto mencatat warm pipeline lengkap 1,57–2,67 detik; cold elderly
masih 5,98 detik. Angka itu termasuk deteksi wajah di aplikasi uji sebelumnya.
API AAR sendiri menerima bitmap dan landmark, sehingga waktu deteksi aplikasi
pemanggil harus ditambahkan ketika mengukur pipeline lengkap.
