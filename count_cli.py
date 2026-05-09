import argparse
import cv2

from detector import template_match_protozoa, draw_detections


def main():
    parser = argparse.ArgumentParser(
        description="Hitung jumlah protozoa menggunakan template oval multi-angle."
    )

    parser.add_argument("image", help="Path gambar input.")
    parser.add_argument("--output", default="hasil_protozoa.jpg")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-distance", type=int, default=45)
    parser.add_argument("--min-scale", type=float, default=0.85)
    parser.add_argument("--max-scale", type=float, default=1.15)

    args = parser.parse_args()

    bgr = cv2.imread(args.image)

    if bgr is None:
        raise FileNotFoundError(args.image)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = template_match_protozoa(
        rgb,
        threshold=args.threshold,
        min_distance=args.min_distance,
        min_scale=args.min_scale,
        max_scale=args.max_scale
    )

    output_rgb = draw_detections(rgb, result["detections"])
    cv2.imwrite(args.output, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))

    print(f"Jumlah protozoa terdeteksi: {result['count']}")
    print(f"Hasil disimpan ke: {args.output}")
    print("Created by Galuh Adi Insani")


if __name__ == "__main__":
    main()
