# Deteksi Protozoa Color Body v3

Versi ini diperbaiki untuk sample protozoa yang memiliki badan hijau/kebiruan di background abu-abu.

Fokus utama: **menghitung badan protozoa utuh**, bukan tekstur internal.

Footer aplikasi:

**Created by Galuh Adi Insani**

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara CLI

```bash
python count_cli.py protozoa_sample.jpg --output hasil.jpg
```

Jika jumlah masih 0:

```bash
python count_cli.py protozoa_sample.jpg --threshold 35 --output hasil.jpg
```

## Tips Pengaturan

- Jika tidak terdeteksi: turunkan **Ambang warna/badan** ke 35–40.
- Jika bagian kecil tubuh ikut dihitung: naikkan **Ukuran minimum badan**.
- Jika satu protozoa terpecah: naikkan **Penggabungan bagian tubuh**.
- Jika protozoa menempel belum terpisah: turunkan sedikit **Kekuatan pemisahan**.
