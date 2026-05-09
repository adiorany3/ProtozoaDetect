import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


FOOTER = "Created by Galuh Adi Insani"

st.set_page_config(
    page_title="YOLO Deteksi Protozoa",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 YOLO Deteksi dan Penghitung Jumlah Protozoa")
st.write(
    "Aplikasi ini menggunakan model YOLO custom untuk mendeteksi protozoa. "
    "Untuk hasil terbaik, gunakan model hasil training dari dataset protozoa yang sudah diberi label bounding box."
)

with st.expander("📌 Cara agar YOLO akurat", expanded=True):
    st.markdown(
        """
        **YOLO akan jauh lebih akurat jika dilatih dengan dataset protozoa milikmu sendiri.**

        Dataset ideal:
        1. Minimal **100–300 gambar** dari mikroskop yang sama atau mirip.
        2. Setiap protozoa diberi label **bounding box satu badan utuh**.
        3. Jangan memberi label pada tekstur/organ kecil di dalam badan.
        4. Label juga protozoa yang sebagian terlihat di pinggir gambar jika ingin ikut dihitung.
        5. Variasikan gambar: terang, gelap, blur ringan, protozoa rapat, protozoa terpisah.
        6. Pisahkan data menjadi **train** dan **val**.
        7. Gunakan gambar asli, jangan terlalu terkompres.
        """
    )

if YOLO is None:
    st.error(
        "Package ultralytics belum terpasang. Jalankan: pip install -r requirements.txt"
    )
    st.stop()

st.sidebar.header("⚙️ Pengaturan YOLO")

default_weight = "weights/best.pt"
weight_path = st.sidebar.text_input(
    "Path model YOLO",
    value=default_weight,
    help="Isi dengan path model hasil training, misalnya weights/best.pt"
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.01,
    help="Turunkan jika protozoa banyak yang belum terdeteksi. Naikkan jika terlalu banyak false positive."
)

iou = st.sidebar.slider(
    "IoU threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.01,
    help="Atur NMS. Naikkan jika objek rapat sering hilang, turunkan jika satu objek terdeteksi ganda."
)

imgsz = st.sidebar.selectbox(
    "Ukuran input YOLO",
    [416, 512, 640, 768, 960, 1024, 1280],
    index=2,
    help="Ukuran lebih besar bisa membantu objek kecil, tetapi lebih berat."
)

uploaded_file = st.file_uploader(
    "Upload gambar protozoa",
    type=["jpg", "jpeg", "png"]
)

@st.cache_resource
def load_model(path):
    return YOLO(path)


def draw_boxes(image_rgb, results):
    output = image_rgb.copy()
    count = 0
    table = []

    if len(results) == 0:
        return output, count, table

    result = results[0]

    if result.boxes is None:
        return output, count, table

    boxes = result.boxes

    for idx, box in enumerate(boxes, start=1):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy

        count += 1

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{idx} {conf:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        table.append(
            {
                "No": idx,
                "Class": cls_id,
                "Confidence": round(conf, 4),
                "X1": int(x1),
                "Y1": int(y1),
                "X2": int(x2),
                "Y2": int(y2),
                "Width": int(x2 - x1),
                "Height": int(y2 - y1)
            }
        )

    return output, count, table


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    if not Path(weight_path).exists():
        st.warning(
            f"Model belum ditemukan di `{weight_path}`. "
            "Latih model terlebih dahulu dengan `python train_yolo.py`, "
            "lalu salin `runs/detect/protozoa_yolo/weights/best.pt` ke folder `weights/best.pt`."
        )

        st.image(image_rgb, caption="Gambar input", use_container_width=True)
        st.stop()

    model = load_model(weight_path)

    with st.spinner("YOLO sedang mendeteksi protozoa..."):
        results = model.predict(
            source=image_rgb,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

    output, count, table = draw_boxes(image_rgb, results)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi YOLO")
        st.image(output, use_container_width=True)

    st.success(f"Jumlah protozoa terdeteksi: {count}")

    with st.expander("Data Deteksi"):
        st.dataframe(table, use_container_width=True)

else:
    st.warning("Silakan upload gambar protozoa terlebih dahulu.")

st.markdown("---")
st.markdown(
    f"""<div style="text-align:center; color:gray; font-size:14px;">{FOOTER}</div>""",
    unsafe_allow_html=True
)
