# Deteksi Protozoa Whole Body v5

Versi ini memperbaiki deteksi agar lebih fokus pada **satu badan protozoa utuh**, bukan bercak/tekstur internal.

## Jalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI

```bash
python count_cli.py protozoa_sample.jpg --output hasil.jpg
```

## Tips tuning

- Jika protozoa belum terdeteksi: turunkan **Ambang badan** ke 20.
- Jika bercak kecil ikut terhitung: naikkan **Ukuran minimum badan**.
- Jika protozoa menempel belum terpisah: turunkan **Kekuatan pemisahan**.
- Jika satu badan protozoa pecah: naikkan **Gabungkan bagian badan** dan **Kekuatan pemisahan**.

Created by Galuh Adi Insani
