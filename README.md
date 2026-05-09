# Deteksi Jumlah Protozoa — Whole Body Detection v2

Aplikasi ini menghitung jumlah protozoa dari gambar mikroskop dengan pendekatan **whole-body detection**.

Tujuan versi ini adalah mengurangi kesalahan ketika sistem menghitung tekstur/organ kecil di dalam badan protozoa sebagai objek terpisah.

Footer aplikasi:

**Created by Galuh Adi Insani**

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Menjalankan dari Terminal

```bash
python count_cli.py protozoa_sample.jpg --output hasil.jpg
```

## Yang Diperbaiki

- Deteksi berfokus pada badan protozoa utuh.
- Bagian tubuh/tekstur internal tidak mudah dihitung sebagai objek baru.
- Mask badan diperkuat dengan penggabungan bagian tubuh.
- Watershed dibuat lebih konservatif agar tidak memecah badan protozoa.
- Filter area minimum dinaikkan agar bercak kecil/noise tidak ikut terhitung.
- Debug visual menampilkan **Whole Body Mask**.

## Tips Pengaturan

Jika satu protozoa masih terpecah menjadi beberapa deteksi:

- Naikkan **Penggabungan bagian tubuh**.
- Naikkan **Ukuran minimum badan**.
- Naikkan **Kekuatan pemisahan objek menempel** agar watershed lebih konservatif.

Jika beberapa protozoa menempel belum terpisah:

- Turunkan sedikit **Kekuatan pemisahan objek menempel**.
- Turunkan sedikit **Penggabungan bagian tubuh**.

Jika objek terlalu sedikit terdeteksi:

- Naikkan **Sensitivitas deteksi**.
- Turunkan sedikit **Ukuran minimum badan**.

## Kriteria Gambar yang Baik

1. Fokus jelas.
2. Kontras cukup.
3. Pencahayaan merata.
4. Background bersih.
5. Objek tidak terlalu menumpuk.
6. Resolusi minimal 300 px pada sisi terpendek.
7. Pembesaran mikroskop konsisten.
8. Hindari gambar yang terlalu terkompres.
9. Gunakan beberapa bidang pandang lalu ambil rata-rata.

## Catatan

Protozoa memiliki variasi bentuk yang sangat besar. Untuk akurasi tinggi pada berbagai jenis protozoa, metode terbaik adalah membuat dataset berlabel dan melatih model object detection seperti YOLO atau Mask R-CNN.
