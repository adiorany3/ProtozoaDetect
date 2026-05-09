# YOLO Deteksi dan Penghitung Jumlah Protozoa

Package ini dibuat untuk deteksi protozoa menggunakan YOLO.

Footer aplikasi:

**Created by Galuh Adi Insani**

## Kenapa YOLO?

YOLO lebih cocok untuk deteksi terbaik karena model bisa belajar langsung dari contoh gambar protozoa yang sudah diberi label. Dibanding threshold, contour, watershed, atau template matching, YOLO lebih tahan terhadap variasi:

- warna protozoa,
- ukuran objek,
- bentuk objek,
- pencahayaan,
- background,
- objek menempel,
- noise mikroskop.

## Struktur Folder

```txt
protozoa_yolo_detector/
├── app.py
├── train_yolo.py
├── predict_cli.py
├── data.yaml
├── requirements.txt
├── README.md
├── weights/
│   └── best.pt                 # simpan model hasil training di sini
├── datasets/
│   └── protozoa/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── sample_images/
└── outputs/
```

## 1. Install Package

```bash
pip install -r requirements.txt
```

## 2. Label Dataset

Gunakan salah satu tool labeling:

- LabelImg
- CVAT
- Roboflow
- makesense.ai

Format label harus **YOLO txt**.

Contoh struktur label:

```txt
datasets/protozoa/images/train/gambar001.jpg
datasets/protozoa/labels/train/gambar001.txt
```

Isi label YOLO:

```txt
0 x_center y_center width height
```

Semua nilai koordinat harus normalized dari 0 sampai 1.

## 3. Aturan Labeling Protozoa

Agar model tidak salah menghitung:

1. Label **satu badan protozoa utuh**.
2. Jangan label tekstur/organ kecil di dalam badan.
3. Jangan label noise, debris, gelembung, atau bercak.
4. Label protozoa yang menempel sebagai objek terpisah jika secara visual masih terlihat sebagai individu berbeda.
5. Label protozoa di pinggir gambar jika memang ingin ikut dihitung.
6. Gunakan class `protozoa` saja.

## 4. Training YOLO

```bash
python train_yolo.py
```

Model terbaik akan muncul di:

```txt
runs/detect/protozoa_yolo/weights/best.pt
```

Salin ke:

```txt
weights/best.pt
```

## 5. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

## 6. Jalankan CLI

```bash
python predict_cli.py sample_images/gambar.jpg --weights weights/best.pt --output outputs/hasil.jpg
```

## Rekomendasi Dataset

Untuk hasil awal:
- minimal 50 gambar berlabel.

Untuk hasil cukup baik:
- 100–300 gambar berlabel.

Untuk hasil lebih stabil:
- 500+ gambar berlabel dari berbagai kondisi mikroskop.

## Tips Parameter

Di aplikasi:

- Jika protozoa kurang terdeteksi, turunkan **Confidence threshold**.
- Jika terlalu banyak false positive, naikkan **Confidence threshold**.
- Jika objek rapat sering hilang, naikkan sedikit **IoU threshold**.
- Jika objek kecil sulit terdeteksi, naikkan **Ukuran input YOLO** ke 960 atau 1280.

## Catatan Penting

File `weights/best.pt` belum disertakan karena harus dibuat dari hasil training dataset protozoa milikmu. Tanpa model hasil training, YOLO tidak akan mengetahui bentuk protozoa yang ingin dihitung.

Created by Galuh Adi Insani
