from ultralytics import YOLO


def main():
    """
    Training YOLO untuk deteksi protozoa.

    Sebelum menjalankan:
    1. Masukkan gambar train ke datasets/protozoa/images/train
    2. Masukkan label train YOLO txt ke datasets/protozoa/labels/train
    3. Masukkan gambar val ke datasets/protozoa/images/val
    4. Masukkan label val YOLO txt ke datasets/protozoa/labels/val

    Jalankan:
    python train_yolo.py
    """

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        patience=25,
        name="protozoa_yolo",
        project="runs/detect",
        workers=2,
        device="cpu"
    )

    print()
    print("Training selesai.")
    print("Model terbaik biasanya tersimpan di:")
    print("runs/detect/protozoa_yolo/weights/best.pt")
    print()
    print("Salin file best.pt ke:")
    print("weights/best.pt")


if __name__ == "__main__":
    main()
