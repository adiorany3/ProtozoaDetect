import cv2
import numpy as np
import math


def odd(value):
    value = max(3, int(round(value)))
    return value if value % 2 == 1 else value + 1


def normalize_u8(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # Koreksi pencahayaan mikroskop yang tidak rata.
    bg_kernel = odd(max(41, min(h, w) * 0.18))
    background = cv2.medianBlur(gray, bg_kernel)
    corrected = cv2.addWeighted(gray, 1.55, background, -0.55, 18)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Kontras lokal.
    clahe = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    # Denoising yang tetap mempertahankan tepi.
    denoised = cv2.bilateralFilter(enhanced, 9, 55, 55)

    return gray, corrected, enhanced, denoised


def build_whole_body_saliency(image_rgb, denoised, mode="Auto"):
    """
    Saliency dibuat untuk mendeteksi SATU BADAN UTUH protozoa,
    bukan tekstur/organ bagian dalamnya.

    Kombinasi:
    - top-hat untuk area tubuh yang terang,
    - black-hat untuk outline/tepi gelap,
    - warna kehijauan/kebiruan yang umum muncul pada sample mikroskop.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv)

    r, g, b = cv2.split(image_rgb)
    green_blue_score = (
        1.4 * g.astype(np.float32)
        + 0.8 * b.astype(np.float32)
        - 0.9 * r.astype(np.float32)
    )
    green_blue_score = normalize_u8(green_blue_score)

    h, w = gray.shape[:2]
    base = min(h, w)

    sizes = [
        odd(max(21, base * 0.055)),
        odd(max(35, base * 0.090)),
        odd(max(51, base * 0.130)),
    ]

    bright_body = np.zeros_like(gray)
    dark_outline = np.zeros_like(gray)

    for size in sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        top_hat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

        bright_body = np.maximum(bright_body, top_hat)
        dark_outline = np.maximum(dark_outline, black_hat)

    bright_body = normalize_u8(bright_body)
    dark_outline = normalize_u8(dark_outline)

    if mode == "Objek terang di background gelap":
        saliency = (
            0.55 * bright_body.astype(np.float32)
            + 0.25 * dark_outline.astype(np.float32)
            + 0.55 * green_blue_score.astype(np.float32)
            + 0.15 * saturation.astype(np.float32)
        )
    elif mode == "Objek gelap di background terang":
        saliency = (
            0.35 * bright_body.astype(np.float32)
            + 0.65 * dark_outline.astype(np.float32)
            + 0.45 * green_blue_score.astype(np.float32)
            + 0.15 * saturation.astype(np.float32)
        )
    else:
        saliency = (
            0.50 * bright_body.astype(np.float32)
            + 0.45 * dark_outline.astype(np.float32)
            + 0.55 * green_blue_score.astype(np.float32)
            + 0.15 * saturation.astype(np.float32)
        )

    saliency = normalize_u8(saliency)
    saliency = cv2.convertScaleAbs(saliency, alpha=1.65, beta=0)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    return saliency, bright_body, dark_outline, green_blue_score


def threshold_whole_body(saliency, sensitivity=1.0, merge_strength=1.0):
    """
    Threshold dibuat konservatif agar tekstur kecil di dalam tubuh
    tidak dihitung sebagai protozoa baru.
    """
    h, w = saliency.shape[:2]
    base = min(h, w)

    percentile = 88 - (sensitivity - 1.0) * 8
    percentile = float(np.clip(percentile, 78, 93))
    threshold_value = max(18, np.percentile(saliency, percentile))

    _, binary_global = cv2.threshold(
        saliency,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    block = odd(max(31, base * 0.085))
    adaptive = cv2.adaptiveThreshold(
        saliency,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        -1
    )

    binary = cv2.bitwise_or(binary_global, adaptive)

    # Buang noise kecil terlebih dahulu.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Kunci utama: closing besar agar bagian dalam tubuh menyatu menjadi satu objek.
    close_size = odd(max(11, base * 0.030 * merge_strength))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Fill hole agar badan utuh, bukan pecahan tekstur.
    binary = fill_holes(binary)

    # Smooth tepi mask.
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)

    # Hilangkan artefak pinggir tipis.
    border = max(2, int(base * 0.006))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    return binary


def fill_holes(binary):
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, flood_inv)

    return filled


def conservative_watershed(image_rgb, binary, enabled=True, split_strength=0.42):
    """
    Watershed dibuat konservatif. Tujuannya memisahkan protozoa yang benar-benar
    menempel, bukan memecah tekstur/badan protozoa.
    """
    if not enabled:
        return binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

    if dist.max() <= 0:
        return binary

    _, sure_fg = cv2.threshold(dist, split_strength * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Marker yang terlalu kecil dihapus agar tekstur internal tidak menjadi marker.
    min_marker_area = max(20, binary.size * 0.00004)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg)
    filtered = np.zeros_like(sure_fg)

    for label_id in range(1, num):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_marker_area:
            filtered[labels == label_id] = 255

    if cv2.countNonZero(filtered) == 0:
        return binary

    sure_bg = cv2.dilate(clean, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, filtered)

    _, markers = cv2.connectedComponents(filtered)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    separated = np.zeros(binary.shape, dtype=np.uint8)
    separated[markers > 1] = 255
    separated = cv2.morphologyEx(separated, cv2.MORPH_CLOSE, kernel, iterations=1)

    return separated


def contour_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0 if perimeter <= 0 else 4 * math.pi * area / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(1, w * h)
    extent = area / bbox_area
    aspect = max(w / max(h, 1), h / max(w, 1))

    hull = cv2.convexHull(contour)
    hull_area = max(1, cv2.contourArea(hull))
    solidity = area / hull_area

    return area, circularity, (x, y, w, h), extent, aspect, solidity


def filter_whole_protozoa(mask, image_shape, min_area_ratio=0.00035, max_area_ratio=0.055):
    """
    Filter lebih menekankan badan protozoa utuh:
    - area minimal dinaikkan agar bagian kecil tubuh/noise tidak dihitung,
    - bentuk boleh oval/panjang, tapi bukan garis tipis,
    - solidity dan extent menjaga supaya objek cukup utuh.
    """
    h, w = image_shape[:2]
    image_area = h * w

    min_area = max(80, image_area * min_area_ratio)
    max_area = max(min_area + 10, image_area * max_area_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        area, circularity, bbox, extent, aspect, solidity = contour_features(contour)
        x, y, bw, bh = bbox

        if area < min_area or area > max_area:
            continue

        if bw < 12 or bh < 12:
            continue

        if bw > w * 0.45 or bh > h * 0.55:
            continue

        # Protozoa pada sample berbentuk oval/daun. Jangan ambil garis/debris.
        if aspect > 5.2:
            continue

        if solidity < 0.32:
            continue

        if extent < 0.16:
            continue

        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + bw / 2
            cy = y + bh / 2

        radius = int(max(bw, bh) / 2)

        score = (
            math.sqrt(area) * 1.15
            + solidity * 22
            + min(circularity, 1.0) * 8
            + extent * 10
        )

        detections.append(
            {
                "x": float(cx),
                "y": float(cy),
                "r": int(max(radius, 4)),
                "area": float(area),
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "circularity": float(circularity),
                "aspect": float(aspect),
                "solidity": float(solidity),
                "extent": float(extent),
                "score": float(score),
                "contour": contour
            }
        )

    detections = sorted(detections, key=lambda item: (item["y"], item["x"]))
    return detections


def count_protozoa(
    image_rgb,
    mode="Auto",
    sensitivity=1.0,
    merge_strength=1.0,
    min_area_ratio=0.00035,
    max_area_ratio=0.055,
    split_touching=True,
    split_strength=0.42
):
    gray, corrected, enhanced, denoised = preprocess(image_rgb)

    saliency, bright_body, dark_outline, color_score = build_whole_body_saliency(
        image_rgb,
        denoised,
        mode=mode
    )

    whole_body_mask = threshold_whole_body(
        saliency,
        sensitivity=sensitivity,
        merge_strength=merge_strength
    )

    separated = conservative_watershed(
        image_rgb,
        whole_body_mask,
        enabled=split_touching,
        split_strength=split_strength
    )

    detections = filter_whole_protozoa(
        separated,
        gray.shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )

    table = []
    for index, item in enumerate(detections, start=1):
        x, y, w, h = item["bbox"]
        table.append(
            {
                "No": index,
                "X": int(round(item["x"])),
                "Y": int(round(item["y"])),
                "Area Badan": round(item["area"], 1),
                "Lebar": w,
                "Tinggi": h,
                "Circularity": round(item["circularity"], 3),
                "Aspect Ratio": round(item["aspect"], 3),
                "Solidity": round(item["solidity"], 3),
                "Extent": round(item["extent"], 3),
                "Score": round(item["score"], 2)
            }
        )

    debug = {
        "Gray": gray,
        "Corrected": corrected,
        "Bright Body": bright_body,
        "Dark Outline": dark_outline,
        "Whole Body Saliency": saliency,
        "Whole Body Mask": whole_body_mask,
        "Watershed": separated
    }

    return {
        "count": len(detections),
        "detections": detections,
        "table": table,
        "debug": debug
    }


def draw_detections(image_rgb, detections):
    output = image_rgb.copy()

    for index, item in enumerate(detections, start=1):
        contour = item.get("contour")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        x = int(round(item["x"]))
        y = int(round(item["y"]))
        r = int(item["r"])

        cv2.circle(output, (x, y), max(4, r), (255, 0, 0), 1)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 2)

        cv2.putText(
            output,
            str(index),
            (x - 9, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
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
    rendered = []

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

        rendered.append(canvas)

    while len(rendered) % 4 != 0:
        rendered.append(np.zeros_like(rendered[0]))

    rows = []
    for i in range(0, len(rendered), 4):
        rows.append(np.hstack(rendered[i:i + 4]))

    return np.vstack(rows)


def image_quality_report(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast_value = gray.std()
    brightness_value = gray.mean()
    min_side = min(h, w)

    report = []

    if min_side >= 300:
        report.append({"status": "baik", "message": f"Resolusi cukup baik: {w} x {h} px."})
    elif min_side >= 200:
        report.append({"status": "cukup", "message": f"Resolusi cukup, tetapi lebih baik minimal 300 px pada sisi terpendek. Saat ini: {w} x {h} px."})
    else:
        report.append({"status": "kurang", "message": f"Resolusi terlalu kecil: {w} x {h} px. Detail protozoa bisa hilang."})

    if blur_value >= 120:
        report.append({"status": "baik", "message": f"Fokus gambar baik. Nilai ketajaman: {blur_value:.1f}."})
    elif blur_value >= 55:
        report.append({"status": "cukup", "message": f"Fokus cukup, tetapi masih bisa lebih tajam. Nilai ketajaman: {blur_value:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Gambar cenderung blur. Nilai ketajaman: {blur_value:.1f}. Tepi protozoa bisa sulit terbaca."})

    if contrast_value >= 35:
        report.append({"status": "baik", "message": f"Kontras gambar baik. Nilai kontras: {contrast_value:.1f}."})
    elif contrast_value >= 22:
        report.append({"status": "cukup", "message": f"Kontras cukup. Nilai kontras: {contrast_value:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Kontras rendah. Nilai kontras: {contrast_value:.1f}. Protozoa sulit dipisahkan dari background."})

    if 60 <= brightness_value <= 205:
        report.append({"status": "baik", "message": f"Pencahayaan berada pada rentang baik. Brightness: {brightness_value:.1f}."})
    elif 40 <= brightness_value < 60 or 205 < brightness_value <= 225:
        report.append({"status": "cukup", "message": f"Pencahayaan cukup, tetapi belum ideal. Brightness: {brightness_value:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Pencahayaan kurang ideal. Brightness: {brightness_value:.1f}. Hindari gambar terlalu gelap/terlalu terang."})

    return report
