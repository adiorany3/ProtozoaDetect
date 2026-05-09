# Sistem Deteksi dan Penghitung Jumlah Protozoa

Aplikasi ini dibuat untuk menghitung jumlah protozoa pada gambar mikroskop.

Footer aplikasi:

**Created by Galuh Adi Insani**

## Fitur

- Upload gambar mikroskop.
- Deteksi objek protozoa secara adaptif.
- Mode objek gelap/terang.
- Segmentasi menggunakan foreground enhancement.
- Pemisahan objek menempel menggunakan watershed.
- Filter ukuran dan bentuk objek.
- Tabel data deteksi.
- Debug visual proses deteksi.
- Analisis kualitas gambar otomatis.

## Cara Menjalankan Aplikasi Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Menjalankan dari Terminal

```bash
python count_cli.py gambar_protozoa.jpg --output hasil.jpg
```

Contoh dengan sensitivitas lebih tinggi:

```bash
python count_cli.py gambar_protozoa.jpg --sensitivity 1.25 --output hasil.jpg
```

## Kriteria Gambar yang Baik

Agar hasil deteksi lebih presisi:

1. Fokus jelas, tepi protozoa tidak blur.
2. Kontras cukup antara protozoa dan background.
3. Pencahayaan merata.
4. Background bersih dari kotoran, gelembung, dan debris.
5. Objek tidak terlalu menumpuk.
6. Resolusi minimal 300 px pada sisi terpendek.
7. Pembesaran mikroskop konsisten.
8. Hindari screenshot atau gambar yang terkompres berat.
9. Ambil beberapa bidang pandang lalu gunakan rata-rata.

## Catatan

Protozoa memiliki bentuk yang sangat beragam. Sistem ini memakai computer vision klasik, sehingga hasil bisa berubah tergantung kualitas gambar, jenis protozoa, pembesaran, fokus, dan pencahayaan.

Untuk akurasi tinggi pada banyak jenis protozoa, kumpulkan dataset berlabel lalu latih model object detection seperti YOLO, Faster R-CNN, atau Mask R-CNN.
