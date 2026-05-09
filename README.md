# Deteksi Protozoa

Versi ini memakai **multi-angle oval template matching**.

Metode ini dibuat untuk sample protozoa berbentuk oval/daun, termasuk gambar hijau/kebiruan dengan target **29 protozoa**.

## Jalankan Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Jalankan CLI

```bash
python count_cli.py protozoa_green_sample_29.jpg --output hasil.jpg
```

## Setting default untuk sample hijau target 29

```txt
Threshold kemiripan bentuk: 0.55
Jarak minimum antar protozoa: 45
Skala minimum objek: 0.85
Skala maksimum objek: 1.15
```

## Tips tuning

- Jika hasil kurang dari target: turunkan Threshold ke 0.52.
- Jika hasil terlalu banyak: naikkan Threshold ke 0.57–0.60.
- Jika satu protozoa terhitung ganda: naikkan Jarak minimum.
- Jika protozoa rapat belum terpisah: turunkan Jarak minimum.

Created by Galuh Adi Insani
