import argparse
import cv2

from detector import count_protozoa, draw_detections, image_quality_report


def main():
    parser = argparse.ArgumentParser(description="Hitung jumlah protozoa dari gambar mikroskop.")
    parser.add_argument("image", help="Path gambar input.")
    parser.add_argument("--output", default="hasil_deteksi_protozoa.jpg", help="Path gambar output.")
    parser.add_argument(
        "--mode",
        default="Auto",
        choices=["Auto", "Objek gelap di background terang", "Objek terang di background gelap"],
        help="Mode polaritas objek."
    )
    parser.add_argument("--sensitivity", type=float, default=1.0)
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = count_protozoa(rgb, mode=args.mode, sensitivity=args.sensitivity)

    output_rgb = draw_detections(rgb, result["detections"])
    cv2.imwrite(args.output, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

    print(f"Jumlah protozoa terdeteksi: {result['count']}")
    print(f"Hasil gambar disimpan ke: {args.output}")
    print("\nAnalisis kualitas gambar:")
    for item in image_quality_report(rgb):
        print(f"- [{item['status'].upper()}] {item['message']}")
    print("\nCreated by Galuh Adi Insani")


if __name__ == "__main__":
    main()
