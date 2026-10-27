# Image Enhancer Experiment Results

> Validasi lanjutan dan batas perbandingan tersedia di [validation-2026-10-07.md](validation-2026-10-07.md). Dua kasus gagal dalam laporan ini berhasil saat diuji ulang pada aplikasi debug/perangkat berbeda; penyebab kegagalan pada integrasi asli belum teridentifikasi.

## 1. Objective

Eksperimen ini dilakukan untuk membandingkan hasil **image enhancement** menggunakan dua metode:

1. **REST API Image Enhancer**
2. **Nawal SDK Image Enhancer**

Perbandingan dilakukan berdasarkan:

* Keberhasilan proses enhancement.
* Waktu yang dibutuhkan hingga hasil enhancement didapatkan.
* Kualitas hasil gambar setelah enhancement.

> **Note:** Waktu yang tercantum merupakan waktu yang dibutuhkan hingga hasil enhancement berhasil diperoleh.

---

## 2. Test Cases

Enam kategori gambar digunakan sebagai bahan pengujian:

* Close Up
* Low Light
* Low Resolution
* Elderly
* Low Light Cropped
* Low Resolution Cropped

---

## 3. REST API Image Enhancer

| Test Case              | Result |    Time |
| ---------------------- | ------ | ------: |
| Close Up               | DONE   |  5.83 s |
| Low Light              | DONE   |  6.24 s |
| Low Resolution         | DONE   |  6.21 s |
| Elderly                | DONE   | 16.16 s |
| Low Light Cropped      | DONE   |  4.05 s |
| Low Resolution Cropped | DONE   |  5.70 s |

### Average Processing Time

**7.37 seconds**

Semua test case berhasil diproses oleh REST API Image Enhancer.

---

## 4. Nawal SDK Image Enhancer

| Test Case              | Result |    Time |
| ---------------------- | ------ | ------: |
| Close Up               | Failed |       - |
| Low Light              | DONE   | 18.77 s |
| Low Resolution         | DONE   | 18.27 s |
| Elderly                | Failed |       - |
| Low Light Cropped      | Failed |       - |
| Low Resolution Cropped | DONE   | 18.16 s |

### Failed Cases

Pada test case yang gagal, SDK menampilkan pesan error bawaan:

> "Gagal enhance. Pastikan wajah terdeteksi dan model tersedia"

Failed test cases:

* Close Up
* Elderly
* Low Light Cropped

Dari 6 test case, **3 berhasil dan 3 gagal**.

---

## 5. Comparison

### 5.1 Processing Speed

REST API memiliki waktu pemrosesan yang lebih cepat dibandingkan Nawal SDK.

Rata-rata waktu REST API untuk seluruh test case yang berhasil:

**7.37 seconds**

Rata-rata waktu Nawal SDK untuk test case yang berhasil:

**18.40 seconds**

Untuk perbandingan berpasangan pada tiga test case yang sama-sama berhasil, rata-rata REST API adalah **6.05 seconds** dan Nawal SDK **18.40 seconds**. Dengan demikian, Nawal SDK membutuhkan waktu sekitar **3.04x lebih lama** pada test case yang sama.

> Catatan: rata-rata REST API untuk seluruh enam test case tetap 7.37 seconds, tetapi tidak digunakan sebagai pembagi rasio karena SDK hanya berhasil pada tiga test case. Hasil aktual dapat berbeda tergantung kondisi gambar dan lingkungan pengujian.

---

### 5.2 Enhancement Success Rate

| Method    | Successful | Failed | Success Rate |
| --------- | ---------: | -----: | -----------: |
| REST API  |        6/6 |    0/6 |         100% |
| Nawal SDK |        3/6 |    3/6 |          50% |

REST API berhasil melakukan enhancement pada seluruh gambar yang diuji.

Sementara itu, Nawal SDK gagal melakukan enhancement pada 3 dari 6 gambar.

---

### 5.3 Image Quality

Berdasarkan perbandingan visual pada gambar kategori **Low Light**, hasil enhancement dari REST API terlihat lebih halus.

Hasil dari Nawal SDK masih terlihat memiliki pixelation, sedangkan hasil dari REST API terlihat lebih smooth.

Dengan demikian, berdasarkan pengamatan visual pada eksperimen ini, hasil REST API **terlihat lebih halus pada gambar Low Light**. Pada dua gambar Low Resolution yang sama-sama berhasil, perbedaannya lebih kecil dan penilaiannya masih subjektif. File perbandingan yang tersedia berupa tangkapan layar dengan ukuran berbeda, bukan pasangan output mentah yang siap dibandingkan secara piksel.

---

## 6. Observations

### Nawal SDK

* SDK terkadang gagal melakukan enhancement pada gambar **Close Up**.
* Pada beberapa kondisi, gambar Close Up dapat menyebabkan SDK menganggap wajah tidak terdeteksi.
* Gambar Close Up juga dapat diperoleh dari hasil **auto crop**, namun hasil enhancement melalui SDK masih dapat gagal.
* SDK memberikan error bawaan:

  > "Gagal enhance. Pastikan wajah terdeteksi dan model tersedia"
* Dari sisi processing time, SDK lebih lambat dibandingkan REST API.
* Pada gambar Low Light, hasil enhancement SDK terlihat lebih pixelated dibandingkan hasil REST API.

### REST API

* Semua test case berhasil di-enhance.
* Processing time secara umum lebih cepat.
* Hasil enhancement pada gambar Low Light terlihat lebih smooth dibandingkan hasil dari Nawal SDK.
* Low Light Cropped memiliki waktu processing tercepat pada eksperimen ini, yaitu **4.05 seconds**.

---

## 7. Conclusion

Berdasarkan eksperimen yang dilakukan, **REST API Image Enhancer menunjukkan hasil yang lebih konsisten dibandingkan Nawal SDK** pada test cases yang digunakan.

REST API berhasil memproses seluruh 6 gambar, sedangkan Nawal SDK hanya berhasil memproses 3 dari 6 gambar.

Dari sisi kecepatan, REST API juga lebih cepat. Pada tiga test case yang sama-sama berhasil, rata-rata REST API adalah **6.05 seconds** dan Nawal SDK **18.40 seconds** (sekitar **3.04x lebih lama**). Rata-rata REST API pada keenam test case adalah **7.37 seconds**.

Dari sisi kualitas visual, hasil REST API terlihat lebih smooth, terutama pada gambar **Low Light**, sementara hasil dari Nawal SDK masih menunjukkan pixelation.

Secara keseluruhan, eksperimen menunjukkan adanya perbedaan pada:

* **Reliability:** REST API lebih konsisten pada test cases.
* **Processing Speed:** REST API lebih cepat.
* **Image Quality:** REST API menghasilkan output yang secara visual lebih smooth pada pengujian Low Light.
