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
    parser.add_argument("--merge-strength", type=float, default=1.2)
    parser.add_argument("--min-area-ratio", type=float, default=0.00035)
    parser.add_argument("--max-area-ratio", type=float, default=0.055)
    parser.add_argument("--no-split", action="store_true")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {args.image}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = count_protozoa(
        rgb,
        mode=args.mode,
        sensitivity=args.sensitivity,
        merge_strength=args.merge_strength,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        split_touching=not args.no_split
    )

    output_rgb = draw_detections(rgb, result["detections"])
    cv2.imwrite(args.output, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

    print(f"Jumlah protozoa terdeteksi: {result['count']}")
    print(f"Hasil gambar disimpan ke: {args.output}")
    print()
    print("Analisis kualitas gambar:")
    for item in image_quality_report(rgb):
        print(f"- [{item['status'].upper()}] {item['message']}")

    print()
    print("Created by Galuh Adi Insani")


if __name__ == "__main__":
    main()
