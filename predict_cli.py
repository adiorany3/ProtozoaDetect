import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Deteksi jumlah protozoa menggunakan YOLO."
    )

    parser.add_argument(
        "image",
        help="Path gambar input."
    )

    parser.add_argument(
        "--weights",
        default="weights/best.pt",
        help="Path model YOLO hasil training."
    )

    parser.add_argument(
        "--output",
        default="outputs/hasil_yolo_protozoa.jpg",
        help="Path output gambar."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold untuk NMS."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran input YOLO."
    )

    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {args.weights}. "
            "Latih model terlebih dahulu dengan python train_yolo.py."
        )

    image_bgr = cv2.imread(args.image)

    if image_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")

    model = YOLO(args.weights)

    results = model.predict(
        source=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    count = 0

    if result.boxes is not None:
        count = len(result.boxes)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, annotated)

    print(f"Jumlah protozoa terdeteksi: {count}")
    print(f"Hasil disimpan ke: {args.output}")
    print("Created by Galuh Adi Insani")


if __name__ == "__main__":
    main()
