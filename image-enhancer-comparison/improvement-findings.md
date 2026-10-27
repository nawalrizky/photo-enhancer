# Temuan dan Acuan Improvement Image Enhancer

Dokumen ini merangkum lima temuan dari perbandingan SDK offline dan REST API. Angka eksperimen awal bersumber dari [conclusion.md](conclusion.md); hasil uji ulang dan batas validasinya tercatat di [validation-2026-10-07.md](validation-2026-10-07.md).

## 1. Penyebab kegagalan pada integrasi asli belum diketahui

**Bukti:** Eksperimen awal mencatat SDK gagal pada `Close Up`, `Elderly`, dan `Low Light Cropped`. Namun, uji ulang dengan aplikasi debug berhasil mendeteksi lima landmark dan memproses `Close Up`, `Elderly`, serta `Low Light`. File input asli `Low Light Cropped` dan log aplikasi integrasi yang gagal belum tersedia.

**Implikasi:** Angka keberhasilan 3/6 menggambarkan eksperimen awal, bukan tingkat keberhasilan SDK yang telah terkonfirmasi pada pengujian ulang. Pesan error umum belum cukup untuk mengaitkan kegagalan dengan deteksi wajah, model, atau inferensi.

**Solusi saat bukti tersedia:** Catat tahap yang gagal secara terpisah: decode/orientasi gambar, deteksi wajah, kelengkapan landmark, pemuatan model, dan inferensi. Cocokkan checksum input, model, dan AAR; ulangi pengujian memakai aplikasi integrasi dan versi rilis yang sama. Temuan ini ditunda sampai input `Low Light Cropped` atau log/kode integrasi asli tersedia.

## 2. Materi visual belum cukup untuk mengukur kualitas secara objektif

**Bukti:** File pembanding yang tersedia merupakan tangkapan layar aplikasi, bukan keluaran gambar mentah. Ukuran layar, orientasi, zoom, dan proses penyimpanan dapat memengaruhi tampilan detail.

**Implikasi:** Pengamatan bahwa hasil REST API terlihat lebih halus pada `Low Light` tetap berguna secara visual, tetapi tidak dapat dipakai untuk menghitung selisih kualitas piksel atau memastikan tahap pemrosesan penyebabnya.

**Solusi:** Simpan bitmap hasil SDK langsung dari aplikasi dan decode gambar hasil REST API ke file. Gunakan input yang identik, sejajarkan area wajah, dan tampilkan pada skala yang sama. Nilai wajah dan latar secara terpisah; catat ketajaman, noise, artefak, warna, dan kesesuaian identitas. Simpan pula hasil tanpa kompresi tambahan bila memungkinkan.

## 3. SDK offline lebih lambat pada kasus yang sebanding

**Bukti:** Pada tiga kasus yang sama-sama berhasil dalam eksperimen awal, REST API rata-rata **6,05 detik** dan SDK **18,40 detik**; SDK **3,04× lebih lama**. Uji ulang pada aplikasi debug 8-thread mencatat sekitar **14,8–15,0 detik** untuk `Close Up`, `Elderly`, dan `Low Light`, termasuk inisialisasi. Angka debug tidak dapat menggantikan angka AAR rilis 1.0.0 yang memakai konfigurasi berbeda.

**Implikasi:** Selisih waktu merupakan masalah nyata, tetapi pengukuran awal dan uji ulang memiliki konfigurasi dan kumpulan gambar berbeda.

**Solusi:** Ukur waktu total dari permintaan pengguna hingga hasil tersedia, serta waktu decode, deteksi, inisialisasi, inferensi, dan penyimpanan. Pisahkan panggilan pertama dan panggilan berikutnya. Ulangi tiap kasus beberapa kali pada perangkat dan kondisi yang sama, lalu laporkan median, rentang, dan puncak memori. Bandingkan build debug 8-thread dengan AAR rilis secara terpisah.

## 4. Jalur pemrosesan kualitas SDK dan server belum setara

**Bukti:** SDK offline memakai `GFPGANv1.4_fp16.onnx` untuk satu wajah yang disejajarkan ke input model 512×512, kemudian menempelkannya ke gambar berukuran sama dengan input. Kode REST API memakai `GFPGANv1.4.pth` sebagai model wajah default melalui `GFPGANer` dengan `upscale=2`; Real-ESRGAN `RealESRGAN_x2plus.pth` dapat dipakai untuk latar jika CUDA tersedia. Kode membaca variabel `METHOD`, sedangkan Dockerfile dan Compose menetapkan `ENHANCE_METHOD`; ketidaksesuaian nama itu membuat konfigurasi tersebut tidak mengubah pilihan model, walaupun keduanya saat ini bernilai `gfpgan`. Model dan GPU yang benar-benar aktif saat eksperimen awal belum diverifikasi dari server berjalan.

**Implikasi:** Perbedaan hasil keseluruhan gambar tidak otomatis berarti model wajah SDK lebih buruk. Skala output, pemrosesan latar, dan cara menempel wajah dapat memberi kontribusi tersendiri.

**Solusi:** Dokumentasikan model, versi, GPU, pengaturan background enhancement, ukuran input/output, dan format penyimpanan kedua metode. Bandingkan crop wajah yang sejajar untuk menilai restorasi wajah. Setelah itu bandingkan latar dan area batas tempel untuk menentukan apakah peningkatan latar atau blending perlu ditambahkan ke SDK.

## 5. Optimasi waktu membutuhkan perubahan pada inferensi

**Bukti:** Profiling RK3576 sebelumnya menunjukkan inferensi menghabiskan lebih dari 99% waktu panggilan setelah model siap. Pengujian NNAPI dan XNNPACK pada konfigurasi tersebut lebih lambat. Upaya konversi RKNN belum menghasilkan model yang dapat diuji karena batas memori pada mesin konversi.

**Implikasi:** Memperkecil gambar sumber atau mengoptimalkan proses tempel wajah saja tidak mungkin memangkas latensi secara besar karena model tetap memproses wajah 512×512. Memori juga harus menjadi syarat penerimaan: puncak RSS rilis FP16 sebelumnya sekitar **708 MB**.

**Solusi:** Pertahankan FP16 CPU sebagai baseline. Lanjutkan konversi RKNN pada mesin Linux dengan RAM memadai, lalu validasi cakupan operator, kualitas output, latensi panggilan pertama dan berikutnya, serta puncak memori di RK3576. Jika RKNN tidak memenuhi target, uji arsitektur model yang lebih ringan dengan ambang kualitas yang ditetapkan lebih dahulu. Kandidat INT8 terdahulu belum memenuhi ambang kualitas dan tidak mempercepat proses secara berarti.

## Urutan kerja yang dapat dilakukan sekarang

1. Standarkan benchmark waktu dan memori untuk build debug serta AAR rilis, tanpa mencampur hasil keduanya.
2. Simpan output mentah kedua metode dan lakukan perbandingan wajah, latar, serta batas tempel pada input yang sama.
3. Kunci baseline kualitas dan performa FP16 CPU; gunakan sebagai syarat penerimaan untuk eksperimen RKNN atau model lain.
4. Kembali ke analisis kegagalan integrasi asli setelah input `Low Light Cropped` atau log/kode aplikasi pemanggil SDK tersedia.

## Referensi implementasi di repositori

- Alur deteksi, timing, dan penyimpanan hasil aplikasi uji: `android-test-app/app/src/main/java/com/enhancer/test/MainActivity.kt`.
- Inferensi dan penempelan wajah SDK: `android-test-app/enhancer-sdk/src/main/cpp/ai_filter_enhancer.cpp`.
- Konfigurasi REST API: `enhancer/enhancer.py`.
- Hasil profiling dan eksperimen akselerasi: `android-test-app/PHASE2_RESULTS.md` dan `android-test-app/PHASE3_RESULTS.md`.
