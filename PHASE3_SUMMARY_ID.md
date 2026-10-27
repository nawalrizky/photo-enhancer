# Ringkasan Phase 3 — model apa yang kita pakai?

## 1. Jawaban singkat

**Modelnya tetap GFPGAN v1.4. Kandidat baru dibuat dari ONNX FP32, dikonversi menjadi RKNN, lalu komputasi besarnya dijalankan dengan FP16 pada NPU RK3576.**

Jadi, FP32 dan FP16 muncul pada tahap yang berbeda:

| Tahap | Yang dipakai |
| --- | --- |
| Model sumber untuk kandidat baru | `GFPGANv1.4.onnx` — FP32 |
| Model setelah perbaikan susunan perhitungan warna | `GFPGANv1.4_rgb_bias_consolidated.onnx` — FP32 |
| File yang dibaca aplikasi eksperimen | `GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn` |
| Komputasi besar di NPU | FP16, berdasarkan profil runtime yang diuji |
| Beberapa operasi kecil di CPU | Masih memakai FLOAT/FP32 |
| Output yang diminta aplikasi dari runtime | Float32, kemudian diubah menjadi piksel gambar |

**Model sumber FP32 tidak berarti seluruh perhitungan di perangkat berjalan FP32.** File RKNN kandidat ini memakai kombinasi sesuai hasil kompilasi dan kemampuan perangkat.

## 2. FP32 dan FP16 itu apa?

Keduanya adalah cara komputer menyimpan angka pecahan dalam perhitungan AI.

- **FP32:** menggunakan 32 bit untuk satu angka; presisi lebih tinggi.
- **FP16:** menggunakan 16 bit; angka memakai ruang lebih kecil, dengan presisi lebih rendah.

FP16 dapat membantu efisiensi pada hardware yang mendukungnya. Namun, memakai FP16 saja tidak menjamin aplikasi cepat atau ukuran seluruh file menjadi separuh. Format model, operasi dan tempat perhitungannya dijalankan juga berpengaruh.

SDK lama kita sudah memakai **ONNX FP16**, tetapi berjalan di **CPU**, sehingga masih lambat. Peningkatan kecepatan pada eksperimen ini terutama berasal dari pemindahan komputasi besar ke **NPU**.

## 3. RKNN itu apa?

**RKNN adalah format model dan perangkat lunak Rockchip untuk menjalankan model AI pada perangkat seperti RK3576.** RKNN bukan model pengganti GFPGAN.

| Istilah | Penjelasan sederhana |
| --- | --- |
| GFPGAN | Model AI yang memperbaiki wajah |
| ONNX | Format file model yang dipakai SDK CPU sebelumnya |
| RKNN | Format hasil kompilasi model untuk perangkat Rockchip |
| CPU | Prosesor umum yang menjalankan aplikasi dan berbagai pekerjaan |
| NPU | Prosesor khusus untuk mempercepat perhitungan AI |
| RKNN Toolkit2 | Alat di host Linux untuk mengubah ONNX menjadi RKNN |
| RKNN Runtime | Pustaka di Android (`librknnrt.so`) untuk menjalankan file RKNN |

Alur kandidat baru:

```text
GFPGAN v1.4, ONNX FP32
          ↓ perbaikan susunan bias warna
ONNX FP32 dengan hasil yang ekuivalen
          ↓ konversi memakai RKNN Toolkit2 di Linux
File .rknn untuk RK3576
          ↓ RKNN Runtime di Android
Komputasi besar pada NPU memakai FP16
          ↓
Wajah hasil enhancement ditempel kembali ke foto asli
```

**Host Linux hanya diperlukan untuk membuat model. Saat memproses foto, aplikasi eksperimen menjalankan model secara lokal di Android dan tidak memanggil REST API.**

## 4. Kenapa memilih kandidat dari FP32?

Kita mencoba sumber FP32 dan FP16. Kedua konversi awal berhasil dijalankan, tetapi hasilnya terlalu terang dan gagal uji kualitas.

Pemeriksaan menunjukkan selisih warna yang hampir sama dengan kontribusi bias warna yang hilang. Bias adalah angka tambahan dalam perhitungan model yang ikut menentukan hasil warna.

Solusi yang berhasil adalah **mengumpulkan penambahan bias RGB ke bagian akhir jalur warna yang linear**, menggunakan sumber FP32. Kita tidak melatih model baru dan tidak mengganti bobotnya. Ekuivalensi hasil sebelum/sesudah perubahan diuji di CPU terlebih dahulu, kemudian kandidat RKNN diuji kembali di perangkat.

Kandidat inilah yang lolos pengujian. **Dua file RKNN awal bukan kandidat yang dipilih.**

## 5. Apa hasilnya?

Waktu berikut mencakup pembacaan gambar, deteksi wajah, alignment, enhancement dan penempelan wajah kembali. Ekspor PNG untuk pengujian berada di luar timer.

| Foto | Pemrosesan pertama RKNN | Pemrosesan berikutnya, tiga run |
| --- | ---: | --- |
| Normal | 3,45 detik | 1,58 / 1,57 / 1,59 detik |
| Close-up | 4,79 detik | 1,77 / 1,83 / 1,84 detik |
| Elderly | **5,98 detik** | 2,56 / 2,56 / 2,67 detik |
| Low-light | 4,29 detik | 1,81 / 1,79 / 1,72 detik |

- **Warm run:** model sudah diinisialisasi dan sesi dipakai kembali. Semua 12 warm run memenuhi target <5 detik.
- **Cold call:** pemrosesan pertama masih memerlukan inisialisasi. Elderly pertama belum memenuhi target <5 detik.
- **Kualitas:** output mentah dibandingkan dengan SDK CPU FP16 pada foto dan landmark yang sama. Selisih maksimum kanal warna hanya **1–2 pada skala 0–255** pada empat foto tersebut.
- **Stabilitas:** 16 panggilan RKNN selesai tanpa crash/OOM; hasil empat run per foto identik.

Ini membuktikan kesesuaian dengan baseline CPU yang diuji. Perbandingan piksel dengan REST API belum dilakukan karena output mentah pembanding online belum tersedia.

## 6. Apa batasannya?

- File model baru masih besar: **sekitar 459 MB desimal atau 437,5 MiB**.
- Peak memori proses aplikasi saat pengujian dengan ekspor PNG sekitar **0,89–1,03 GiB**. Kebutuhan memori lebih tinggi daripada referensi CPU yang diuji.
- Beberapa operasi kecil tetap di CPU; sebagian besar komputasi berjalan di NPU.
- Kandidat dikompilasi dan diuji untuk **RK3576**. Hasilnya belum membuktikan kompatibilitas atau kecepatan yang sama pada chipset lain.
- Pengujian memakai empat foto, belum pengujian ketahanan jangka panjang. Gambar `Low Light Cropped` yang belum tersedia tidak termasuk.

## 7. Jadi sekarang aplikasi kita memakai yang mana?

| Artefak | Status |
| --- | --- |
| AAR/ZIP production 1.0.0 | Tetap SDK ONNX CPU sebelumnya |
| Paket SDK RKNN 1.1.0 | AAR baru untuk aplikasi sebenarnya, disertai model dan contoh integrasi |
| Aplikasi uji yang dibuka biasa | Menggunakan kandidat RKNN pada NPU |
| Aplikasi uji dengan `benchmark_backend=cpu` | Menggunakan ONNX FP16 CPU untuk pembanding |
| Aplikasi uji dengan `benchmark_backend=rknn` | Menggunakan kandidat RKNN dari sumber FP32, dengan komputasi besar FP16 di NPU |

**Keputusan Phase 3: kandidat RKNN diterima untuk diteruskan ke integrasi production.** Ini belum berarti release production lama sudah berubah. Setelah perbaikan pilihan default, aplikasi uji yang dibuka biasa memakai RKNN. Backend CPU masih dapat dipilih melalui intent benchmark untuk pembanding.

SDK RKNN release 1.1.0 sudah dikemas dan diuji melalui aplikasi dengan package berbeda serta R8/minification. Paket ada di [engine-delivery-rknn](android-test-app/engine-delivery-rknn/README.md). Tim aplikasi dapat menambahkan AAR dan model tersebut mengikuti panduan integrasi. Pengujian alur aplikasi pelanggan dan ketahanan lebih lanjut tetap diperlukan. Cold call dan kebutuhan memori tetap perlu diperhatikan.

Detail angka, lokasi model, APK, log dan hasil pembanding tersedia di [laporan eksekusi lengkap](PHASE3_RKNN_EXECUTION_RESULTS.md).
