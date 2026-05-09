import numpy as np
import streamlit as st
from PIL import Image

from detector import (
    count_protozoa,
    draw_detections,
    make_debug_grid,
    image_quality_report
)


st.set_page_config(
    page_title="Deteksi Jumlah Protozoa",
    page_icon="🔬",
    layout="wide"
)

FOOTER = "Created by Galuh Adi Insani"

st.title("🔬 Sistem Deteksi dan Penghitung Jumlah Protozoa")
st.write(
    "Versi ini diperbaiki untuk sample protozoa yang kamu kirim. "
    "Deteksi difokuskan ke **badan protozoa utuh**, bukan bercak/tekstur kecil di dalam badannya."
)

with st.expander("📌 Keterangan gambar yang baik agar hasil lebih presisi", expanded=True):
    st.markdown(
        """
        Gunakan gambar dengan:
        1. **Fokus jelas**, tepi badan protozoa terlihat.
        2. **Kontras cukup**, badan protozoa berbeda dari background.
        3. **Pencahayaan merata**, tidak terlalu gelap/terlalu terang.
        4. **Background bersih**, minim kotoran, gelembung, dan debris.
        5. **Objek tidak terlalu bertumpuk**.
        6. **Resolusi cukup**, minimal 300 px pada sisi terpendek.
        7. **Pembesaran konsisten**.
        """
    )

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop protozoa",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("⚙️ Pengaturan Deteksi")

threshold_value = st.sidebar.slider(
    "Ambang warna/badan",
    min_value=20,
    max_value=90,
    value=45,
    step=1,
    help="Turunkan jika tidak ada protozoa terdeteksi. Naikkan jika background/noise ikut terdeteksi."
)

merge_strength = st.sidebar.slider(
    "Penggabungan bagian tubuh",
    min_value=0.50,
    max_value=2.50,
    value=1.25,
    step=0.05,
    help="Naikkan jika satu protozoa masih pecah menjadi beberapa deteksi."
)

min_area_ratio = st.sidebar.slider(
    "Ukuran minimum badan",
    min_value=0.00020,
    max_value=0.00500,
    value=0.00100,
    step=0.00010,
    format="%.5f",
    help="Naikkan jika bagian dalam tubuh/noise masih ikut dihitung."
)

max_area_ratio = st.sidebar.slider(
    "Ukuran maksimum badan",
    min_value=0.005,
    max_value=0.120,
    value=0.060,
    step=0.005,
    format="%.3f",
    help="Turunkan jika cluster besar ikut dihitung sebagai satu objek."
)

split_touching = st.sidebar.checkbox(
    "Pisahkan protozoa yang menempel",
    value=True
)

split_strength = st.sidebar.slider(
    "Kekuatan pemisahan",
    min_value=0.25,
    max_value=0.70,
    value=0.48,
    step=0.01,
    help="Naikkan agar tidak memecah badan. Turunkan jika protozoa menempel belum terpisah."
)

show_debug = st.checkbox("Tampilkan proses deteksi", value=True)
show_quality = st.checkbox("Tampilkan analisis kualitas gambar", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    result = count_protozoa(
        image_rgb,
        threshold_value=threshold_value,
        merge_strength=merge_strength,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        split_touching=split_touching,
        split_strength=split_strength
    )

    output = draw_detections(image_rgb, result["detections"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah protozoa terdeteksi: {result['count']}")

    st.info(
        "Jika jumlah masih 0, turunkan **Ambang warna/badan** ke 35–40. "
        "Jika bagian dalam badan masih ikut dihitung, naikkan **Ukuran minimum badan** dan "
        "**Penggabungan bagian tubuh**."
    )

    if show_quality:
        st.subheader("Analisis Kualitas Gambar")
        for item in image_quality_report(image_rgb):
            if item["status"] == "baik":
                st.success(item["message"])
            elif item["status"] == "cukup":
                st.warning(item["message"])
            else:
                st.error(item["message"])

    if show_debug:
        st.subheader("Debug Visual")
        st.write(
            "Yang paling penting dilihat adalah **Whole Body Mask** dan **Separated Mask**. "
            "Mask yang baik menutup satu badan protozoa secara utuh."
        )
        st.image(make_debug_grid(result["debug"]), use_container_width=True)

    with st.expander("Data Deteksi Protozoa"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar mikroskop protozoa terlebih dahulu.")

st.markdown("---")
st.markdown(
    f"""<div style="text-align:center; color:gray; font-size:14px;">{FOOTER}</div>""",
    unsafe_allow_html=True
)
