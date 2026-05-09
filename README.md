# Deteksi Protozoa

Versi ini mendukung:
- protozoa hijau/kebiruan,
- protozoa coklat/abu-abu seperti sample dengan target 4 protozoa.

## Jalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI

```bash
python count_cli.py protozoa_brown_sample_4.jpeg --output hasil.jpg
```

## Tips tuning

- Jika kurang terdeteksi: turunkan **Ambang badan** ke 20–25.
- Jika terlalu banyak: naikkan **Ukuran minimum badan** atau **Ambang badan**.
- Jika protozoa menempel belum terpisah: turunkan **Kekuatan pemisahan**.
- Jika satu badan pecah: naikkan **Gabungkan bagian badan**.

Created by Galuh Adi Insani
