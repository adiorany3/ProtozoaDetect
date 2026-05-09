import numpy as np
import streamlit as st
from PIL import Image

from detector import count_protozoa, draw_detections, make_debug_grid, image_quality_report


st.set_page_config(page_title="Deteksi Jumlah Protozoa", page_icon="🔬", layout="wide")

FOOTER = "Created by Galuh Adi Insani"

st.title("🔬 Deteksi Jumlah Protozoa — Whole Body v5")
st.write(
    "Versi ini memperbaiki masalah protozoa belum terhitung sempurna dengan pendekatan "
    "**whole body + color saliency + split objek menempel**. Fokusnya menghitung satu badan protozoa utuh, "
    "bukan bercak/tekstur di dalam badan."
)

with st.expander("📌 Keterangan gambar yang baik", expanded=True):
    st.markdown(
        """
        Agar hasil lebih presisi:
        1. Badan protozoa terlihat fokus dan tepinya jelas.
        2. Kontras antara protozoa dan background cukup.
        3. Pencahayaan rata, tidak terlalu gelap/terang.
        4. Background bersih dari debris/gelembung.
        5. Objek tidak terlalu bertumpuk.
        6. Resolusi minimal 300 px pada sisi terpendek.
        7. Pembesaran mikroskop konsisten.
        """
    )

uploaded = st.file_uploader("Upload gambar protozoa", type=["jpg", "jpeg", "png"])

st.sidebar.header("⚙️ Pengaturan")
body_threshold = st.sidebar.slider(
    "Ambang badan",
    10, 80, 25, 1,
    help="Turunkan jika protozoa belum terdeteksi. Naikkan jika background/noise ikut terdeteksi."
)

merge_strength = st.sidebar.slider(
    "Gabungkan bagian badan",
    0.40, 2.00, 0.85, 0.05,
    help="Naikkan jika satu protozoa terpecah. Turunkan jika banyak protozoa menempel menjadi satu."
)

split_touching = st.sidebar.checkbox("Pisahkan protozoa yang menempel", value=True)

split_strength = st.sidebar.slider(
    "Kekuatan pemisahan",
    0.20, 0.60, 0.34, 0.01,
    help="Turunkan jika protozoa menempel belum terpisah. Naikkan jika badan protozoa pecah."
)

min_area_ratio = st.sidebar.slider(
    "Ukuran minimum badan",
    0.00020, 0.00300, 0.00055, 0.00005, format="%.5f",
    help="Naikkan jika bercak kecil masih ikut dihitung."
)

max_area_ratio = st.sidebar.slider(
    "Ukuran maksimum badan",
    0.010, 0.120, 0.070, 0.005, format="%.3f",
    help="Turunkan jika cluster besar ikut dihitung."
)

count_edge_objects = st.sidebar.checkbox("Hitung protozoa di pinggir gambar", value=True)

show_debug = st.checkbox("Tampilkan proses deteksi", value=True)
show_quality = st.checkbox("Tampilkan analisis kualitas gambar", value=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    result = count_protozoa(
        image_rgb,
        body_threshold=body_threshold,
        merge_strength=merge_strength,
        split_touching=split_touching,
        split_strength=split_strength,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        count_edge_objects=count_edge_objects
    )

    output = draw_detections(image_rgb, result["detections"])

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with c2:
        st.subheader("Hasil Deteksi")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah protozoa terdeteksi: {result['count']}")

    st.info(
        "Untuk sample yang kamu kirim, coba default dulu. Jika masih kurang, turunkan **Ambang badan** ke 20. "
        "Jika bagian kecil ikut terhitung, naikkan **Ukuran minimum badan**."
    )

    if show_quality:
        st.subheader("Analisis Kualitas Gambar")
        for status, msg in image_quality_report(image_rgb):
            if status == "baik":
                st.success(msg)
            elif status == "cukup":
                st.warning(msg)
            else:
                st.error(msg)

    if show_debug:
        st.subheader("Debug Visual")
        st.write("Cek **Whole Body Mask** dan **Separated Mask**. Mask yang baik menutup satu protozoa sebagai satu objek.")
        st.image(make_debug_grid(result["debug"]), use_container_width=True)

    with st.expander("Data Deteksi"):
        st.dataframe(result["table"], use_container_width=True)

else:
    st.warning("Silakan upload gambar protozoa terlebih dahulu.")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:gray;font-size:14px'>{FOOTER}</div>", unsafe_allow_html=True)
