import numpy as np
import streamlit as st
from PIL import Image

from detector import count_protozoa, draw_detections, make_debug_grid, image_quality_report


st.set_page_config(
    page_title="Deteksi Jumlah Protozoa",
    page_icon="🔬",
    layout="wide"
)

FOOTER = "Created by Galuh Adi Insani"

st.title("🔬 Sistem Deteksi dan Penghitung Jumlah Protozoa")
st.write(
    "Upload gambar mikroskop protozoa. Sistem akan menghitung objek protozoa "
    "menggunakan preprocessing adaptif, segmentasi, watershed, dan filter bentuk."
)

with st.expander("📌 Kriteria gambar yang baik agar hasil lebih presisi", expanded=True):
    st.markdown(
        """
        **Agar deteksi protozoa lebih akurat, gunakan gambar dengan kondisi berikut:**

        1. **Fokus jelas**, bentuk tepi protozoa terlihat dan tidak blur.
        2. **Kontras cukup**, protozoa harus terlihat berbeda dari background.
        3. **Pencahayaan merata**, hindari area terlalu gelap/terang pada satu sisi.
        4. **Background bersih**, minim kotoran, gelembung, bercak, dan debris.
        5. **Objek tidak terlalu menumpuk**, protozoa yang saling bertumpuk sulit dipisahkan.
        6. **Resolusi cukup**, disarankan minimal 300 px pada sisi terpendek.
        7. **Skala pembesaran konsisten**, supaya filter ukuran objek lebih stabil.
        8. **Hindari kompresi berat**, gambar buram dari screenshot/WhatsApp dapat mengurangi akurasi.
        9. **Ambil beberapa bidang pandang**, lalu gunakan rata-rata untuk estimasi yang lebih representatif.
        """
    )

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop protozoa",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("⚙️ Pengaturan")
mode = st.sidebar.selectbox(
    "Mode objek",
    ["Auto", "Objek gelap di background terang", "Objek terang di background gelap"],
    index=0
)

sensitivity = st.sidebar.slider(
    "Sensitivitas deteksi",
    min_value=0.50,
    max_value=2.00,
    value=1.00,
    step=0.05,
    help="Naikkan jika protozoa banyak yang belum terdeteksi. Turunkan jika noise ikut terhitung."
)

min_area_ratio = st.sidebar.slider(
    "Ukuran minimum objek",
    min_value=0.00001,
    max_value=0.00200,
    value=0.00008,
    step=0.00001,
    format="%.5f",
    help="Naikkan jika banyak titik kecil/noise ikut terhitung."
)

max_area_ratio = st.sidebar.slider(
    "Ukuran maksimum objek",
    min_value=0.001,
    max_value=0.100,
    value=0.030,
    step=0.001,
    format="%.3f",
    help="Turunkan jika area besar/debris ikut terhitung."
)

split_touching = st.sidebar.checkbox(
    "Pisahkan objek yang saling menempel",
    value=True
)

show_debug = st.checkbox("Tampilkan proses deteksi", value=True)
show_quality = st.checkbox("Tampilkan analisis kualitas gambar", value=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    result = count_protozoa(
        image_rgb,
        mode=mode,
        sensitivity=sensitivity,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        split_touching=split_touching
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
        debug_grid = make_debug_grid(result["debug"])
        st.image(debug_grid, use_container_width=True)

    with st.expander("Data Deteksi Protozoa"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar mikroskop protozoa terlebih dahulu.")

st.markdown("---")
st.markdown(
    f"""<div style="text-align:center; color:gray; font-size:14px;">{FOOTER}</div>""",
    unsafe_allow_html=True
)
