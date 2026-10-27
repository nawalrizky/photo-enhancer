# Phase 3 RKNN — hasil eksekusi 7 Oktober 2026

## Keputusan

### Perbaikan pilihan default aplikasi uji

Uji manual setelah benchmark masih mencatat `backend=cpu`, total SDK call 14,827 detik. Aplikasi saat itu memilih CPU ketika dibuka biasa. Default aplikasi uji sudah diubah menjadi RKNN, label model menjadi `RKNN / NPU`, dan APK baru dipasang pada perangkat. Pengujian tanpa extra pemilihan backend mengonfirmasi `backend=rknn`: SDK call pertama 2,694 detik dan pipeline lengkap 4,309 detik pada foto low-light. Log: `android-test-app/rknn-experiment/logs/device-rknn-default.log`. Backend CPU masih tersedia melalui `benchmark_backend=cpu` untuk pembanding. AAR/ZIP production 1.0.0 tetap belum berubah.

**ACCEPT sebagai kandidat integrasi production berdasarkan gate warm latency dalam execution plan.** Keempat milestone selesai pada perangkat RK3576 yang diuji. APK dan backend eksperimen tersedia; AAR/ZIP production 1.0.0 belum diganti.

Seluruh **12 warm full-pipeline calls <5 detik**, kualitas cocok dengan baseline FP16 CPU, dan 16 panggilan RKNN selesai tanpa crash/OOM. Hasil ini berlaku untuk empat foto dan perangkat yang diuji, bukan bukti pengujian ketahanan jangka panjang.

## Latency pipeline lengkap

Timer mencakup antrean processing, decode/orientasi/resize gambar, ML Kit face detection, inisialisasi model bila diperlukan, alignment, normalisasi, inferensi, konversi output, dan paste-back. Pemilihan foto, startup Activity sebelum processing, ekspor PNG diagnostik, dan transfer ADB berada di luar timer. Deteksi wajah diulang pada setiap warm run.

| Foto | Ukuran setelah decode | CPU pertama | RKNN pertama | Tiga warm calls RKNN |
| --- | --- | ---: | ---: | --- |
| Normal | 256×256 | 15,578 s | 3,451 s | 1,580 / 1,573 / 1,585 s |
| Close-up | 401×498 | 16,397 s | 4,792 s | 1,770 / 1,827 / 1,840 s |
| Elderly | 1519×2048 | 18,198 s | **5,980 s** | 2,564 / 2,558 / 2,672 s |
| Low-light | 365×547 | 16,644 s | 4,292 s | 1,805 / 1,793 / 1,715 s |

**Batasan cold call:** elderly pertama masih melewati 5 detik. Execution plan menggunakan tiga warm full calls untuk keputusan latency; hasil ini tidak menyatakan semua cold calls <5 detik. CPU diuji satu cold call per foto untuk menghasilkan referensi mentah; tabel tidak membandingkan warm CPU dengan warm RKNN.

## Kualitas output mentah

PNG hasil aplikasi dibandingkan piksel RGB dengan CPU pada input dan landmark yang identik. Pemeriksaan visual juga dilakukan pada keempat gambar. Identitas/detail dan paste-back tidak menunjukkan perubahan tambahan yang terlihat terhadap baseline pada foto yang diuji.

| Foto | Rata-rata error RGB, skala 255 | Error kanal maksimum | PSNR |
| --- | ---: | ---: | ---: |
| Normal | 0,0266 | 2 | 63,88 dB |
| Close-up | 0,0564 | 1 | 60,62 dB |
| Elderly | 0,0683 | 1 | 59,79 dB |
| Low-light | 0,0052 | 2 | 71,00 dB |

Semua empat output RKNN per foto identik secara SHA256. Referensi ini adalah output mentah **ONNX FP16 CPU pada aplikasi debug 8-thread**, bukan screenshot atau keluaran REST API. `Low Light Cropped` yang belum tersedia tetap tidak termasuk pengujian.

## Temuan dan solusi yang diperlukan

Konversi awal FP32 dan repaired-FP16 berhasil tetapi **gagal kualitas**: output menjadi terlalu terang, tensor MAE sekitar 0,625. Pergantian format input saja tidak menyelesaikannya.

Selisih per kanal hampir sama dengan negatif jumlah tujuh bias RGB skip model. Script [consolidate_gfpgan_rgb_bias.py](android-test-app/scripts/consolidate_gfpgan_rgb_bias.py) mengumpulkan delapan bias melalui jalur RGB linear menjadi satu Add di output. Rewrite ini mempertahankan bobot dan matematika model; ekuivalensi CPU terhadap FP32 asli diuji pada foto dan tensor sintetis, dengan error maksimum <3×10⁻⁷.

Setelah kompilasi ulang, tensor MAE terhadap baseline FP16 CPU turun menjadi **0,000380 pada foto** dan **0,001391 pada input sintetis**. Kandidat tersebut yang dipakai dalam benchmark aplikasi di atas. Dua kandidat awal ditolak untuk integrasi.

Adapter input aplikasi eksperimen memakai **FP16 NHWC native, `pass_through=1`**; output diambil sebagai float32 NCHW. Alignment lima landmark, normalisasi RGB, mask dan paste-back memakai algoritma SDK CPU yang sama.

## Memori dan operator

- Peak process VmHWM RKNN aplikasi: **911–1.059 MiB**, termasuk ML Kit, UI, bitmap dan ekspor PNG diagnostik. CPU referensi: 723–796 MiB. Ini bukan penghematan memori terhadap CPU.
- Perangkat menyediakan 8.104.736 KiB RAM; setelah benchmark low-light, saat aplikasi RKNN masih aktif, MemAvailable **4.980.176 KiB (~4,75 GiB)**. Tidak ada crash/OOM dalam 16 calls yang diuji.
- Probe runtime final melaporkan DMA allocated **1.136.504.832 byte (~1,06 GiB)**. Angka DMA dicatat terpisah; jangan menjumlahkannya langsung dengan RSS karena alokasi dapat tumpang tindih.
- Profil kandidat final menunjukkan convolution/resize dan komputasi besar di NPU. Delapan `Mul` kecil serta input/output masih berjalan di CPU: **12,374 ms dari 1.139,148 ms (~1,09%)** waktu operator pada run profiling. Runtime ini menggunakan kombinasi NPU dan CPU. Cakupan tersebut dinilai dapat diterima untuk kandidat ini karena kualitas, latency dan stabilitas terukur lolos; hasil tidak mengklaim seluruh graph berjalan di NPU.

Memory gate dinilai lolos untuk perangkat 8 GB ini dengan batas pengujian di atas. Pengujian banyak sesi/ketahanan lama dan verifikasi pada perangkat lain merupakan pekerjaan release berikutnya.

## Artefak dan lingkungan

- Host: `ubuntu@10.147.20.16`.
- Direktori host: `/home/ubuntu/Documents/work/photo-enhancer-rknn`.
- Konversi: Python 3.10, RKNN Toolkit2 2.3.2; kandidat final selesai 60,86 s, sampled process-tree peak RSS 5.749,02 MiB.
- Device: YF_021B / RK3576, Android 14, arm64; runtime API 2.3.2, driver 0.9.8.
- Model final: `android-test-app/rknn-experiment/artifacts/GFPGANv1.4_rgb_bias_consolidated_rk3576.rknn`, **458.741.110 byte (~437,5 MiB)**.
- SHA256 model: `ac5cb93f894599e26d4a3ea71f0b546073678c86016f15ca21b1efcadc4b7a1c`.
- [APK debug eksperimen](android-test-app/app/build/outputs/apk/debug/app-debug.apk).
- [Backend Android eksperimen](android-test-app/rknn-experiment/android/src/main/cpp/ai_filter_enhancer.cpp).
- [Petunjuk reproduksi](android-test-app/rknn-experiment/README.md) dan [catatan eksekusi rinci](android-test-app/rknn-experiment/EXECUTION_RESULTS.md).
- [Ringkasan JSON](android-test-app/rknn-experiment/artifacts/pipeline/results-summary.json), [konsistensi](android-test-app/rknn-experiment/artifacts/pipeline/consistency-validation.json), serta PNG/logcat setiap kasus di `artifacts/pipeline/`.

Model final masih besar dan kebutuhan memori lebih tinggi daripada CPU. Kandidat sudah layak diteruskan ke integrasi release dengan catatan tersebut; artefak production lama dipertahankan selama release baru disiapkan dan diuji.
