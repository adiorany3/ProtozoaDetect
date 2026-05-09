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
    "Versi ini memakai pendekatan **whole-body detection**, sehingga sistem berusaha "
    "menghitung satu badan protozoa utuh, bukan tekstur/organ kecil di dalam tubuhnya."
)

with st.expander("📌 Kriteria gambar yang baik agar hasil lebih presisi", expanded=True):
    st.markdown(
        """
        **Agar deteksi protozoa lebih akurat, gunakan gambar dengan kondisi berikut:**

        1. **Fokus jelas**, tepi badan protozoa terlihat dan tidak blur.
        2. **Kontras cukup**, badan protozoa berbeda jelas dari background.
        3. **Pencahayaan merata**, hindari sisi gambar terlalu gelap atau terlalu terang.
        4. **Background bersih**, minim kotoran, gelembung, bercak, dan debris.
        5. **Objek tidak terlalu menumpuk**, protozoa yang saling menempel/tumpang tindih sulit dipisahkan.
        6. **Resolusi cukup**, disarankan minimal 300 px pada sisi terpendek.
        7. **Skala pembesaran konsisten**, supaya filter ukuran badan lebih stabil.
        8. **Hindari kompresi berat**, gambar dari screenshot/WhatsApp yang buram dapat mengurangi akurasi.
        9. **Ambil beberapa bidang pandang**, lalu gunakan rata-rata untuk estimasi yang lebih representatif.
        """
    )

uploaded_file = st.file_uploader(
    "Upload gambar mikroskop protozoa",
    type=["jpg", "jpeg", "png"]
)

st.sidebar.header("⚙️ Pengaturan Deteksi")

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
    help="Naikkan jika protozoa banyak yang belum terdeteksi. Turunkan jika noise/bagian tubuh ikut terhitung."
)

merge_strength = st.sidebar.slider(
    "Penggabungan bagian tubuh",
    min_value=0.50,
    max_value=2.00,
    value=1.20,
    step=0.05,
    help="Naikkan jika satu protozoa masih terpecah menjadi beberapa bagian."
)

min_area_ratio = st.sidebar.slider(
    "Ukuran minimum badan",
    min_value=0.00005,
    max_value=0.00500,
    value=0.00035,
    step=0.00005,
    format="%.5f",
    help="Naikkan jika bagian kecil/noise masih ikut terhitung."
)

max_area_ratio = st.sidebar.slider(
    "Ukuran maksimum badan",
    min_value=0.005,
    max_value=0.120,
    value=0.055,
    step=0.005,
    format="%.3f",
    help="Turunkan jika cluster besar ikut dihitung sebagai satu objek."
)

split_touching = st.sidebar.checkbox(
    "Pisahkan protozoa yang saling menempel",
    value=True
)

split_strength = st.sidebar.slider(
    "Kekuatan pemisahan objek menempel",
    min_value=0.25,
    max_value=0.65,
    value=0.42,
    step=0.01,
    help="Naikkan agar pemisahan lebih konservatif. Turunkan jika objek menempel belum terpisah."
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
        "Jika hasil masih menghitung bagian dalam tubuh protozoa, naikkan "
        "**Ukuran minimum badan** dan **Penggabungan bagian tubuh**. "
        "Jika protozoa menempel belum terpisah, turunkan sedikit **Kekuatan pemisahan objek menempel**."
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
            "Perhatikan bagian **Whole Body Mask**. Mask yang baik harus menutup badan protozoa secara utuh, "
            "bukan hanya tekstur/bercak kecil di dalam badannya."
        )
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
