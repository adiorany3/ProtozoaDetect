import argparse
import cv2
from detector import count_protozoa, draw_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", default="hasil_protozoa.jpg")
    parser.add_argument("--body-threshold", type=int, default=25)
    parser.add_argument("--merge-strength", type=float, default=0.85)
    parser.add_argument("--split-strength", type=float, default=0.34)
    parser.add_argument("--min-area-ratio", type=float, default=0.00055)
    parser.add_argument("--max-area-ratio", type=float, default=0.070)
    parser.add_argument("--no-split", action="store_true")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = count_protozoa(
        rgb,
        body_threshold=args.body_threshold,
        merge_strength=args.merge_strength,
        split_touching=not args.no_split,
        split_strength=args.split_strength,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio
    )

    out = draw_detections(rgb, result["detections"])
    cv2.imwrite(args.output, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print(f"Jumlah protozoa terdeteksi: {result['count']}")
    print(f"Hasil disimpan ke: {args.output}")
    print("Created by Galuh Adi Insani")


if __name__ == "__main__":
    main()
