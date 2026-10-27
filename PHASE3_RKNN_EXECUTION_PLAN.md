# Phase 3 — RKNN Conversion and Sub-5-Second Image Enhancer Validation

## 1. Objective

Mengembangkan pipeline **image enhancement offline/on-device** pada device **RK3576** untuk menggantikan dependency REST API, dengan target:

- End-to-end processing time **< 5 detik**
- Kualitas enhancement tetap acceptable dibanding baseline GFPGAN saat ini
- Berjalan sepenuhnya offline
- Stabil pada representative test images
- Penggunaan memory masih aman pada device target

> RKNN bukan goal utama. GFPGAN juga bukan goal utama.
> **Goal utama adalah offline image enhancement < 5 detik dengan kualitas yang acceptable.**

---

## 2. Current Baseline

### REST API

- Model: GFPGAN v1.4
- Format: PyTorch (`GFPGANv1.4.pth`)
- Runtime: server-side
- Dapat menggunakan CUDA
- Dapat menggunakan Real-ESRGAN untuk background enhancement
- End-to-end latency baseline: sekitar **6,05 detik**

### Offline SDK

- Model: GFPGAN v1.4
- Format: FP16 ONNX (`GFPGANv1.4_fp16.onnx`)
- Runtime: ONNX Runtime CPU
- Device: RK3576
- Warm latency: sekitar **13–15 detik**
- Lebih dari **99% latency** berasal dari inference GFPGAN

Optimasi preprocessing, bitmap conversion, alignment, dan paste-back tidak cukup untuk mencapai target karena seluruh tahap tersebut hanya menyumbang sebagian kecil dari total waktu.

---

## 3. Why RKNN

RK3576 memiliki NPU Rockchip.

Saat ini GFPGAN berjalan seperti berikut:

```text
GFPGAN FP16 ONNX
        ↓
ONNX Runtime
        ↓
CPU RK3576
        ↓
~13–15 detik
```

Target eksperimen:

```text
GFPGAN ONNX
        ↓
RKNN-Toolkit2
        ↓
GFPGAN .rknn
        ↓
RKNN Runtime
        ↓
NPU RK3576
        ↓
Target < 5 detik
```

RKNN digunakan agar model dapat dijalankan menggunakan NPU RK3576, bukan hanya CPU.

---

# 4. Execution Plan

## Milestone 1 — Build GFPGAN RKNN Model

Tujuan milestone pertama:

> Menghasilkan file `.rknn` yang valid dari model GFPGAN.

### 4.1 Prepare Linux Conversion Host

Gunakan Linux host dengan:

```text
Recommended:
Ubuntu 22.04
Architecture: x86_64
RAM: 16 GB minimum practical target
RAM: 32 GB recommended
Disk: 40–80 GB
Python: 3.10
RKNN-Toolkit2: 2.3.2
```

RAM besar hanya diperlukan untuk **proses conversion/compilation ONNX → RKNN**, bukan untuk menjalankan aplikasi di device Android.

Percobaan sebelumnya:

```text
3 GB Docker RAM      → OOM
3.83 GB              → OOM
5.3 GB               → OOM
Optimization level 0 → tetap OOM
```

Karena itu conversion berikutnya dilakukan pada host dengan memory yang jauh lebih besar.

### 4.2 Verify Linux Host

Jalankan:

```bash
free -h
uname -m
python3 --version
```

Expected:

```text
Architecture: x86_64
RAM: >= 16 GB
```

---

## 5. Prepare RKNN Environment

Install dependency dasar:

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3-pip
```

Buat virtual environment:

```bash
python3.10 -m venv ~/rknn-env
source ~/rknn-env/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install **RKNN-Toolkit2 2.3.2** dan dependency yang digunakan project.

Gunakan versi yang sama dengan eksperimen sebelumnya agar hasil benchmark tetap comparable.

---

## 6. Required Project Files

Pastikan host memiliki minimal:

```text
models/
├── GFPGANv1.4.onnx
└── GFPGANv1.4_fp16.onnx

scripts/
├── convert_model_rknn.py
└── repair_fp16_resize_roi.py
```

Jika repository tersedia melalui Git:

```bash
git clone <repository-url>
cd <repository-directory>
```

---

# 7. Convert FP32 Model First

FP32 menjadi kandidat pertama karena eksperimen sebelumnya sudah berhasil melewati tahap graph parsing dan constant folding.

Kegagalan sebelumnya terjadi ketika:

```text
I rknn building ...
```

dan process dihentikan karena kehabisan RAM.

Jalankan:

```bash
python scripts/convert_model_rknn.py \
  --input models/GFPGANv1.4.onnx \
  --output models/GFPGANv1.4_rk3576.rknn
```

Monitor memory selama conversion:

```bash
watch -n 1 free -h
```

atau:

```bash
htop
```

### Success Condition

File berikut berhasil dibuat:

```text
models/GFPGANv1.4_rk3576.rknn
```

Cek ukuran:

```bash
ls -lh models/GFPGANv1.4_rk3576.rknn
```

Simpan seluruh conversion log.

Perhatikan warning terkait:

- unsupported operator
- operator fallback
- graph optimization
- tensor layout
- datatype conversion
- quantization
- input/output contract

---

# 8. Convert Repaired FP16 Model

Setelah FP32 berhasil, lanjutkan FP16.

FP16 original sebelumnya mengalami error pada empty ROI di beberapa `Resize` nodes.

Repair model terlebih dahulu:

```bash
python scripts/repair_fp16_resize_roi.py \
  --input models/GFPGANv1.4_fp16.onnx \
  --output models/GFPGANv1.4_fp16_repaired.onnx
```

Kemudian convert:

```bash
python scripts/convert_model_rknn.py \
  --input models/GFPGANv1.4_fp16_repaired.onnx \
  --output models/GFPGANv1.4_fp16_rk3576.rknn
```

Jika keduanya berhasil, pertahankan dua kandidat:

```text
GFPGANv1.4_rk3576.rknn
GFPGANv1.4_fp16_rk3576.rknn
```

Jangan langsung menghapus salah satunya sebelum quality dan latency dibandingkan.

---

# 9. Milestone 2 — Validate RKNN Runtime on RK3576

Tujuan milestone:

> Membuktikan RKNN Runtime dapat berjalan dengan benar pada device YF_021B / RK3576.

Verifikasi device:

```bash
adb devices
adb shell getprop ro.product.cpu.abi
adb shell uname -a
adb shell ls -l /dev/dri/
```

Jika tersedia:

```bash
adb shell cat /sys/kernel/debug/rknpu/load 2>/dev/null
```

Device sebelumnya sudah terdeteksi memiliki RKNPU.

Yang perlu divalidasi berikutnya adalah compatibility RKNN Runtime.

---

# 10. Prepare RKNN Runtime

Gunakan `librknnrt.so` yang kompatibel dengan:

```text
Android
arm64-v8a
RK3576
```

Secara umum runtime Android ARM64 berada pada struktur seperti:

```text
runtime/Android/librknn_api/arm64-v8a/librknnrt.so
```

Jangan langsung mengganti AAR production.

Gunakan **test build terpisah** sampai:

- model berhasil load
- inference berhasil
- kualitas tervalidasi
- latency tervalidasi
- memory tervalidasi

---

# 11. Smoke Test RKNN Runtime

Sebelum memasukkan GFPGAN ke SDK utama, jalankan terlebih dahulu contoh/demo RKNN sederhana.

Tujuan:

> Memisahkan masalah runtime/device dengan masalah implementasi GFPGAN.

Jika RKNN demo berhasil:

```text
Android
    ↓
librknnrt.so
    ↓
RKNPU
    ↓
successful inference
```

baru lanjutkan GFPGAN.

---

# 12. Milestone 3 — Integrate GFPGAN RKNN

Pipeline lama:

```text
Bitmap
  ↓
Face Detection
  ↓
Alignment 512×512
  ↓
ONNX Runtime CPU
  ↓
GFPGAN
  ↓
Output Conversion
  ↓
Paste-back
```

Pipeline baru:

```text
Bitmap
  ↓
Face Detection
  ↓
Alignment 512×512
  ↓
RKNN Runtime
  ↓
RK3576 NPU
  ↓
GFPGAN
  ↓
Output Conversion
  ↓
Paste-back
```

Pada tahap awal, jangan mengubah:

- face detection
- alignment
- preprocessing
- paste-back
- output handling

Sebisa mungkin hanya ganti **inference backend**.

Tujuannya agar perbandingan CPU vs NPU tetap fair.

---

# 13. Validate Tensor Contract

Pastikan kontrak input/output RKNN sesuai dengan pipeline ONNX lama.

Periksa:

```text
Input shape
Output shape
NCHW / NHWC
FP16 / FP32 / INT8
RGB / BGR
Normalization
Input range
Output range
Tensor ordering
```

Masalah seperti wajah berubah, warna aneh, atau output rusak belum tentu berarti RKNN gagal.

Kemungkinan penyebab:

```text
NCHW vs NHWC
RGB vs BGR
normalization mismatch
FP16/FP32 conversion
output scaling
tensor ordering
```

Semua poin ini harus divalidasi sebelum menilai kualitas model.

---

# 14. Milestone 4 — Benchmark

Gunakan input yang sama dengan baseline.

Minimal benchmark:

```text
Run 1 → cold call
Run 2 → warm
Run 3 → warm
Run 4 → warm
```

Jangan mengambil hanya nilai tercepat.

Catat:

```text
Model initialization
Image decode
Face detection
Alignment
Preprocessing
RKNN inference
Output conversion
Paste-back
Total end-to-end latency
Peak memory
```

Untuk latency decision gate, gunakan **warm full-call latency**.

---

# 15. Representative Test Set

Gunakan dataset tetap.

Minimal:

```text
Normal
Close Up
Elderly
Low Light
Low Light Cropped — jika file asli ditemukan
```

Status temuan sebelumnya:

```text
Close Up
Initial test : FAIL
Retest       : PASS

Elderly
Initial test : FAIL
Retest       : PASS

Low Light Cropped
Initial test : FAIL
Exact retest : BELUM TERSEDIA

Low Light
Retest       : PASS
```

Karena failure Close Up dan Elderly tidak reproducible pada retest, failure awal belum dapat dianggap sebagai limitation SDK yang terkonfirmasi.

---

# 16. Quality Gate

RKNN hanya dapat diterima jika tidak menyebabkan penurunan kualitas signifikan terhadap baseline FP16 CPU.

Minimal quality check:

- Identity wajah tetap terjaga
- Struktur wajah tidak berubah signifikan
- Tidak muncul artifact signifikan
- Tidak ada kerusakan warna yang jelas
- Detail wajah tetap acceptable
- Hasil konsisten pada representative images

Gunakan raw output image, bukan hanya screenshot aplikasi.

---

# 17. Memory and Stability Gate

Latency bukan satu-satunya acceptance criteria.

Catat:

```text
Peak RSS
VmHWM
crash / OOM
repeated inference stability
session initialization behavior
```

Baseline sebelumnya:

```text
FP16 release peak RSS ≈ 708 MB
```

Eksperimen CPU arena/pattern pernah meningkatkan memory high-water mark hingga sekitar:

```text
≈ 1.19 GB
```

RKNN harus diuji kembali secara terpisah.

---

# 18. Final Decision Gate

## ACCEPT

GFPGAN RKNN dapat menjadi kandidat production apabila:

```text
3 warm full calls < 5 detik
+
quality PASS
+
memory PASS
+
stability PASS
```

Keputusan:

```text
ACCEPT
→ lanjut production integration
```

---

## CONDITIONAL

Jika:

```text
Warm latency = 5–6 detik
+
quality PASS
+
memory PASS
```

Keputusan:

```text
CONDITIONAL
```

Optimasi tambahan masih layak dilakukan apabila terdapat improvement yang jelas dan terukur.

Namun hasil tersebut belum memenuhi target utama `<5s`.

---

## REJECT GFPGAN PATH

Jika:

```text
Warm latency > 6 detik
```

atau:

```text
quality degradation signifikan
operator coverage bermasalah
runtime tidak stabil
memory tidak acceptable
```

Keputusan:

```text
STOP GFPGAN optimization path
        ↓
Evaluate lighter image restoration model
```

GFPGAN bukan requirement final.

Requirement final tetap:

```text
Offline
+
<5 second end-to-end latency
+
acceptable quality
```

---

# 19. Milestone Summary

## Milestone 1

```text
GFPGAN ONNX
    ↓
RKNN Toolkit
    ↓
.rknn berhasil dibuat
```

**Gate:** `.rknn` valid tersedia.

---

## Milestone 2

```text
Android RK3576
    ↓
RKNN Runtime
    ↓
RKNPU
```

**Gate:** RKNN runtime berhasil menjalankan model/demo pada device.

---

## Milestone 3

```text
GFPGAN.rknn
    ↓
RK3576 NPU
    ↓
valid enhanced output
```

**Gate:** inference berhasil dan output secara visual/numerik masuk akal.

---

## Milestone 4

```text
Representative benchmark
    ↓
Latency
Quality
Memory
Stability
```

**Gate:** final acceptance decision.

---

# 20. Immediate Next Action

Pada Linux host baru, mulai dengan:

```bash
free -h
uname -m
python3 --version
```

Kemudian setup environment dan jalankan **FP32 conversion terlebih dahulu**:

```bash
python scripts/convert_model_rknn.py \
  --input models/GFPGANv1.4.onnx \
  --output models/GFPGANv1.4_rk3576.rknn
```

Untuk sementara:

> Jangan mengubah production AAR, Android integration, atau pipeline preprocessing sebelum `.rknn` yang valid berhasil dihasilkan.

---

# 21. Final Goal

```text
Current REST API
~6.05 detik
      ↓

Current Offline CPU
~13–15 detik
      ↓

GFPGAN RKNN / RK3576 NPU
      ↓

TARGET
<5 detik end-to-end
+
offline
+
acceptable quality
+
acceptable memory usage
```
