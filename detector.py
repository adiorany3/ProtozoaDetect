import cv2
import numpy as np
import math


def normalize_u8(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def create_protozoa_saliency(image_rgb):
    """
    Membuat peta objek protozoa.
    Cocok untuk sample hijau/kebiruan dan coklat/abu-abu.
    """
    r, g, b = cv2.split(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv)

    # Protozoa hijau/kebiruan.
    green_score = (
        2.0 * g.astype(np.float32)
        - 0.80 * r.astype(np.float32)
        - 0.80 * b.astype(np.float32)
    )
    green_score = np.clip(green_score, 0, 255)
    green_score = normalize_u8(green_score)

    # Protozoa coklat/abu-abu biasanya sedikit lebih gelap dan bersaturasi dari background.
    brown_score = (
        0.65 * saturation.astype(np.float32)
        + 0.65 * (255 - value).astype(np.float32)
        + 0.20 * r.astype(np.float32)
    )
    brown_score = normalize_u8(brown_score)

    # Outline gelap dan isi badan.
    h, w = gray.shape[:2]
    base = min(h, w)

    sizes = [
        max(13, int(base * 0.045)),
        max(21, int(base * 0.080)),
        max(31, int(base * 0.120)),
    ]

    top_hat = np.zeros_like(gray)
    black_hat = np.zeros_like(gray)

    denoised = cv2.bilateralFilter(gray, 7, 45, 45)

    for size in sizes:
        if size % 2 == 0:
            size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        top_hat = np.maximum(top_hat, cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel))
        black_hat = np.maximum(black_hat, cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel))

    top_hat = normalize_u8(top_hat)
    black_hat = normalize_u8(black_hat)

    color_score = np.maximum(green_score, brown_score)

    saliency = (
        0.60 * color_score.astype(np.float32)
        + 0.30 * black_hat.astype(np.float32)
        + 0.20 * top_hat.astype(np.float32)
        + 0.18 * saturation.astype(np.float32)
    )

    saliency = normalize_u8(saliency)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    return saliency, green_score, brown_score, black_hat, top_hat


def create_ellipse_template(width, height, angle):
    pad = 6
    canvas_w = int(width + 2 * pad)
    canvas_h = int(height + 2 * pad)

    template = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    cv2.ellipse(
        template,
        (canvas_w // 2, canvas_h // 2),
        (max(2, width // 2), max(2, height // 2)),
        angle,
        0,
        360,
        255,
        -1
    )

    template = cv2.GaussianBlur(template, (0, 0), 3)
    return template.astype(np.float32) / 255.0


def template_match_protozoa(
    image_rgb,
    threshold=0.55,
    min_distance=45,
    min_scale=0.85,
    max_scale=1.15
):
    """
    Template matching multi-angle agar protozoa oval dihitung sebagai satu objek.
    Default disesuaikan untuk sample hijau dengan target 29 protozoa.
    """
    saliency, green_score, brown_score, black_hat, top_hat = create_protozoa_saliency(image_rgb)

    h, w = saliency.shape[:2]
    base = min(h, w)

    # Ukuran template otomatis berdasarkan tinggi gambar.
    # Untuk sample 1280x720, ukuran ini cocok untuk protozoa oval.
    proto_w = int(base * 0.095)
    proto_h = int(base * 0.150)

    proto_w = max(22, proto_w)
    proto_h = max(34, proto_h)

    scales = np.linspace(min_scale, max_scale, 3)

    saliency_float = saliency.astype(np.float32) / 255.0

    candidates = []

    for scale in scales:
        tw = int(proto_w * scale)
        th = int(proto_h * scale)

        if tw < 8 or th < 8:
            continue

        for angle in range(0, 180, 15):
            template = create_ellipse_template(tw, th, angle)

            if template.shape[0] >= h or template.shape[1] >= w:
                continue

            result = cv2.matchTemplate(
                saliency_float,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            peak_kernel_size = max(9, int(min_distance))
            peak_kernel = np.ones((peak_kernel_size, peak_kernel_size), dtype=np.float32)
            local_max = cv2.dilate(result, peak_kernel)

            ys, xs = np.where((result == local_max) & (result >= threshold))

            for y, x in zip(ys, xs):
                cx = float(x + template.shape[1] / 2)
                cy = float(y + template.shape[0] / 2)
                score = float(result[y, x])

                candidates.append(
                    {
                        "x": cx,
                        "y": cy,
                        "score": score,
                        "width": int(tw),
                        "height": int(th),
                        "angle": int(angle),
                    }
                )

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)

    detections = []

    for candidate in candidates:
        duplicate = False

        for saved in detections:
            distance = math.hypot(candidate["x"] - saved["x"], candidate["y"] - saved["y"])

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            detections.append(candidate)

    detections = sorted(detections, key=lambda item: (item["y"], item["x"]))

    table = []

    for i, item in enumerate(detections, start=1):
        table.append(
            {
                "No": i,
                "X": int(round(item["x"])),
                "Y": int(round(item["y"])),
                "Lebar Template": item["width"],
                "Tinggi Template": item["height"],
                "Angle": item["angle"],
                "Score": round(item["score"], 3),
            }
        )

    debug = {
        "Saliency": saliency,
        "Green Score": green_score,
        "Brown Score": brown_score,
        "Black-hat Outline": black_hat,
        "Top-hat Body": top_hat,
    }

    return {
        "count": len(detections),
        "detections": detections,
        "table": table,
        "debug": debug,
    }


def draw_detections(image_rgb, detections):
    output = image_rgb.copy()

    for i, item in enumerate(detections, start=1):
        x = int(round(item["x"]))
        y = int(round(item["y"]))
        width = int(item["width"])
        height = int(item["height"])
        angle = int(item["angle"])

        cv2.ellipse(
            output,
            (x, y),
            (max(3, width // 2), max(3, height // 2)),
            angle,
            0,
            360,
            (0, 255, 0),
            2
        )

        cv2.circle(output, (x, y), 3, (255, 0, 0), 3)

        cv2.putText(
            output,
            str(i),
            (x - 10, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output


def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def make_debug_grid(debug):
    names = list(debug.keys())
    images = [to_rgb(debug[name]) for name in names]

    target_w = 340
    target_h = 210
    tiles = []

    for name, image in zip(names, images):
        image = cv2.resize(image, (target_w, target_h))
        canvas = image.copy()

        cv2.rectangle(canvas, (0, 0), (target_w, 30), (255, 255, 255), -1)
        cv2.putText(
            canvas,
            name,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        tiles.append(canvas)

    while len(tiles) % 3 != 0:
        tiles.append(np.zeros_like(tiles[0]))

    rows = []
    for i in range(0, len(tiles), 3):
        rows.append(np.hstack(tiles[i:i + 3]))

    return np.vstack(rows)


def image_quality_report(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()

    report = []

    if min(h, w) >= 300:
        report.append(("baik", f"Resolusi cukup baik: {w} x {h} px."))
    else:
        report.append(("cukup", f"Resolusi kecil: {w} x {h} px. Lebih baik gunakan gambar minimal 300 px pada sisi terpendek."))

    if blur >= 80:
        report.append(("baik", f"Fokus gambar cukup baik. Nilai ketajaman: {blur:.1f}."))
    elif blur >= 35:
        report.append(("cukup", f"Fokus cukup, tetapi bisa lebih tajam. Nilai ketajaman: {blur:.1f}."))
    else:
        report.append(("kurang", f"Gambar cenderung blur. Nilai ketajaman: {blur:.1f}."))

    if contrast >= 20:
        report.append(("baik", f"Kontras cukup. Nilai kontras: {contrast:.1f}."))
    else:
        report.append(("cukup", f"Kontras rendah. Nilai kontras: {contrast:.1f}."))

    report.append(("baik", f"Brightness rata-rata: {brightness:.1f}."))
    return report
