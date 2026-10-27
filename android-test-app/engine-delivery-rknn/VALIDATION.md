# Validasi paket SDK RKNN 1.1.0 — 7 Oktober 2026

## Hasil

**AAR release berhasil dipakai oleh aplikasi dengan package berbeda**, hanya
melalui dependency file AAR. Consumer tidak bergantung pada source/project
SDK maupun aplikasi uji lama. Tidak ada ONNX Runtime di APK consumer.

| Pemeriksaan | Hasil |
| --- | --- |
| Build AAR release | Berhasil |
| Build consumer release dengan R8/minification | Berhasil |
| Consumer package | `com.enhancer.integrationcheck` |
| Consumer debuggable | Tidak; release APK memakai debug signing khusus demo |
| Permission Internet | Tidak ada pada manifest SDK maupun consumer |
| Runtime native lengkap di AAR/APK | RKNN, OpenCV, engine, libc++ — arm64-v8a |
| Initialize dan enhancement | Berhasil pada RK3576 / Android 14 / driver 0.9.8 |
| Nama JNI setelah R8 | Berfungsi; tidak ada UnsatisfiedLinkError |
| Empat panggilan satu session | Berhasil tanpa crash/OOM |
| Hasil vs raw CPU reference | Rata-rata error RGB 0,026606/255; maksimum 2/255 |

## Waktu consumer

Foto sample 256×256 dengan lima landmark yang sama dengan benchmark sebelumnya:

| Run | Waktu SDK call |
| --- | ---: |
| Pertama, termasuk initialize | 2,701 detik |
| Warm 1 | 1,359 detik |
| Warm 2 | 1,534 detik |
| Warm 3 | 1,384 detik |

Timer consumer meliputi `initialize` dan `enhanceFull`, termasuk bitmap conversion,
alignment, inference dan paste-back. Decode foto dilakukan sebelumnya; consumer
memakai landmark sample sehingga waktu face detection tidak termasuk. Angka ini
tidak disajikan sebagai latency pipeline aplikasi sebenarnya.

Hasil numerik, PNG dan log tersedia di `validation/`. Baseline empat foto untuk
pipeline lengkap aplikasi sebelumnya tetap 1,57–2,67 detik hangat; cold elderly
5,98 detik. Pengujian AAR consumer ini menambah bukti bahwa packaging, package
aplikasi berbeda, release native libraries dan R8 dapat bekerja.

## Deployment model

Percobaan pertama menyalin model ke external storage lewat ADB, tetapi perangkat
memberi file tersebut ownership `shell`, sehingga aplikasi release tidak dapat
membacanya. Contoh consumer sekarang menulis model ke `filesDir/models/` melalui
provider deployment yang dibatasi permission sistem `DUMP` untuk ADB shell.
Provider hanya ada pada APK demonstrasi; AAR tidak menambah provider/permission.
Untuk aplikasi sebenarnya, gunakan installer/provisioning aplikasi dan contoh
`ModelInstaller.kt` yang disertakan.

## Batas validasi

Paket sudah siap diintegrasikan oleh tim aplikasi. Pengujian pada source dan alur
aplikasi pelanggan sendiri belum dilakukan karena project tersebut belum diberikan.
SDK tidak menyertakan face detector; koordinat, lifecycle bitmap, provisioning
model dan lifecycle session tetap dikelola aplikasi pemanggil. Ketahanan jangka
panjang dan perangkat selain unit RK3576 ini belum diuji.
