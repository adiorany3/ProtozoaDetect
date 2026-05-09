import numpy as np
import streamlit as st
from PIL import Image

from detector import (
    template_match_protozoa,
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

st.title("🔬 Deteksi Jumlah Protozoa")
st.write(
    "Versi ini memakai **multi-angle oval template matching**. "
    "Metode ini lebih cocok untuk protozoa berbentuk oval/daun dan mengurangi kesalahan menghitung tekstur di dalam badan."
)

with st.expander("📌 Keterangan gambar yang baik", expanded=True):
    st.markdown(
        """
        Agar hasil lebih presisi:
        1. Protozoa terlihat sebagai badan oval/daun yang cukup jelas.
        2. Tepi badan tidak terlalu blur.
        3. Kontras antara objek dan background cukup.
        4. Pencahayaan merata.
        5. Objek tidak terlalu bertumpuk.
        6. Gunakan pembesaran yang konsisten.
        7. Hindari kompresi gambar berlebihan.
        """
    )

uploaded = st.file_uploader(
    "Upload gambar protozoa",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("⚙️ Pengaturan Deteksi")

threshold = st.sidebar.slider(
    "Threshold kemiripan bentuk",
    min_value=0.30,
    max_value=0.80,
    value=0.55,
    step=0.01,
    help="Turunkan jika protozoa kurang terdeteksi. Naikkan jika terlalu banyak noise terhitung."
)

min_distance = st.sidebar.slider(
    "Jarak minimum antar protozoa",
    min_value=15,
    max_value=100,
    value=45,
    step=1,
    help="Turunkan jika protozoa rapat belum terpisah. Naikkan jika satu protozoa terhitung ganda."
)

min_scale = st.sidebar.slider(
    "Skala minimum objek",
    min_value=0.50,
    max_value=1.50,
    value=0.85,
    step=0.05
)

max_scale = st.sidebar.slider(
    "Skala maksimum objek",
    min_value=0.70,
    max_value=2.00,
    value=1.15,
    step=0.05
)

show_debug = st.checkbox("Tampilkan proses deteksi", value=True)
show_quality = st.checkbox("Tampilkan analisis kualitas gambar", value=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    result = template_match_protozoa(
        image_rgb,
        threshold=threshold,
        min_distance=min_distance,
        min_scale=min_scale,
        max_scale=max_scale
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
        "Untuk sample hijau yang kamu kirim, targetnya 29. "
        "Default yang disarankan: Threshold 0.55 dan Jarak minimum 45. "
        "Jika hasil kurang dari 29, turunkan Threshold ke 0.52. "
        "Jika hasil lebih dari 29, naikkan Threshold ke 0.57–0.60."
    )

    if show_quality:
        st.subheader("Analisis Kualitas Gambar")
        for status, message in image_quality_report(image_rgb):
            if status == "baik":
                st.success(message)
            elif status == "cukup":
                st.warning(message)
            else:
                st.error(message)

    if show_debug:
        st.subheader("Debug Visual")
        st.write(
            "Saliency yang baik membuat badan protozoa terlihat terang, sedangkan background lebih redup."
        )
        st.image(make_debug_grid(result["debug"]), use_container_width=True)

    with st.expander("Data Deteksi"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar protozoa terlebih dahulu.")

st.markdown("---")
st.markdown(
    f"""<div style="text-align:center; color:gray; font-size:14px;">{FOOTER}</div>""",
    unsafe_allow_html=True
)
