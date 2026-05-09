import cv2
import numpy as np
import math


def odd(v):
    v = max(3, int(round(v)))
    return v if v % 2 == 1 else v + 1


def normalize_u8(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fill_holes(binary):
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, inv)


def preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    bg_kernel = odd(max(51, min(h, w) * 0.20))
    background = cv2.medianBlur(gray, bg_kernel)

    corrected = cv2.addWeighted(gray, 1.50, background, -0.50, 15)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    denoised = cv2.bilateralFilter(enhanced, 9, 60, 60)

    return gray, corrected, enhanced, denoised


def color_body_saliency(image_rgb, denoised):
    """
    Untuk sample protozoa yang tampak hijau/kebiruan di background abu-abu.
    Fokusnya mencari badan utuh, bukan tekstur kecil di dalam badan.
    """
    r, g, b = cv2.split(image_rgb)

    # Excess green-blue: cocok untuk sample protozoa yang berpendar hijau pucat.
    exg = (
        2.05 * g.astype(np.float32)
        - 0.95 * r.astype(np.float32)
        - 0.75 * b.astype(np.float32)
    )
    exg = np.clip(exg, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv)

    h, w = denoised.shape[:2]
    base = min(h, w)

    # Top-hat: bagian badan yang terang.
    # Black-hat: outline/tepi gelap.
    sizes = [
        odd(max(25, base * 0.055)),
        odd(max(41, base * 0.090)),
        odd(max(61, base * 0.135)),
    ]

    top_response = np.zeros_like(denoised)
    black_response = np.zeros_like(denoised)

    for size in sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        top = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
        black = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
        top_response = np.maximum(top_response, top)
        black_response = np.maximum(black_response, black)

    top_response = normalize_u8(top_response)
    black_response = normalize_u8(black_response)
    exg = normalize_u8(exg)

    saliency = (
        0.85 * exg.astype(np.float32)
        + 0.35 * saturation.astype(np.float32)
        + 0.40 * top_response.astype(np.float32)
        + 0.25 * black_response.astype(np.float32)
    )

    saliency = normalize_u8(saliency)
    saliency = cv2.GaussianBlur(saliency, (7, 7), 0)
    saliency = cv2.convertScaleAbs(saliency, alpha=1.55, beta=0)

    return saliency, exg, top_response, black_response


def make_whole_body_mask(saliency, threshold_value=45, merge_strength=1.25):
    """
    Threshold default disesuaikan agar sample protozoa.jpg tidak kosong.
    Closing besar menyatukan badan, sehingga bagian internal tidak dihitung terpisah.
    """
    h, w = saliency.shape[:2]
    base = min(h, w)

    # Kombinasi threshold manual stabil + Otsu rendah.
    _, manual = cv2.threshold(saliency, threshold_value, 255, cv2.THRESH_BINARY)

    otsu_val, otsu = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_val > threshold_value + 25:
        # Jangan terlalu ketat kalau Otsu gagal karena background luas.
        otsu = manual

    binary = cv2.bitwise_or(manual, otsu)

    # Bersihkan noise kecil.
    open_size = odd(max(5, base * 0.010))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Gabungkan tepi/isi badan agar satu protozoa menjadi satu blob.
    close_size = odd(max(15, base * 0.040 * merge_strength))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    binary = fill_holes(binary)

    smooth_size = odd(max(7, base * 0.015))
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_size, smooth_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)

    # Hilangkan artefak batas.
    border = max(2, int(base * 0.004))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    return binary


def conservative_watershed(image_rgb, mask, enabled=True, split_strength=0.48):
    """
    Watershed konservatif: memisahkan objek yang benar-benar menempel,
    tapi tidak terlalu agresif agar badan tidak pecah.
    """
    if not enabled:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask

    _, sure_fg = cv2.threshold(dist, split_strength * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Buang marker kecil agar tekstur internal tidak memecah badan.
    min_marker_area = max(60, mask.size * 0.00010)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg)
    filtered = np.zeros_like(sure_fg)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_marker_area:
            filtered[labels == i] = 255

    if cv2.countNonZero(filtered) == 0:
        return mask

    sure_bg = cv2.dilate(clean, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, filtered)

    _, markers = cv2.connectedComponents(filtered)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    result = np.zeros(mask.shape, dtype=np.uint8)
    result[markers > 1] = 255

    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    return result


def contour_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 0 if perimeter <= 0 else 4 * math.pi * area / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(1, w * h)
    aspect = max(w / max(h, 1), h / max(w, 1))

    hull = cv2.convexHull(contour)
    solidity = area / max(1, cv2.contourArea(hull))

    return area, circularity, (x, y, w, h), extent, aspect, solidity


def filter_protozoa(mask, image_shape, min_area_ratio=0.00100, max_area_ratio=0.060):
    """
    Filter area dibuat lebih besar agar bagian kecil di dalam badan tidak terhitung.
    """
    h, w = image_shape[:2]
    image_area = h * w

    min_area = max(700, image_area * min_area_ratio)
    max_area = max(min_area + 20, image_area * max_area_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        area, circularity, bbox, extent, aspect, solidity = contour_features(contour)
        x, y, bw, bh = bbox

        if area < min_area or area > max_area:
            continue

        if bw < 25 or bh < 25:
            continue

        if bw > w * 0.45 or bh > h * 0.60:
            continue

        # Protozoa sample berbentuk oval/daun; longgar tapi bukan garis tipis.
        if aspect > 4.8:
            continue

        if solidity < 0.26:
            continue

        if extent < 0.14:
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
            + min(circularity, 1.0) * 8
            + solidity * 20
            + extent * 8
        )

        detections.append({
            "x": float(cx),
            "y": float(cy),
            "r": int(max(radius, 5)),
            "area": float(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "circularity": float(circularity),
            "aspect": float(aspect),
            "solidity": float(solidity),
            "extent": float(extent),
            "score": float(score),
            "contour": contour
        })

    return sorted(detections, key=lambda d: (d["y"], d["x"]))


def count_protozoa(
    image_rgb,
    threshold_value=45,
    merge_strength=1.25,
    min_area_ratio=0.00100,
    max_area_ratio=0.060,
    split_touching=True,
    split_strength=0.48
):
    gray, corrected, enhanced, denoised = preprocess(image_rgb)

    saliency, exg, top_response, black_response = color_body_saliency(image_rgb, denoised)

    mask = make_whole_body_mask(
        saliency,
        threshold_value=threshold_value,
        merge_strength=merge_strength
    )

    separated = conservative_watershed(
        image_rgb,
        mask,
        enabled=split_touching,
        split_strength=split_strength
    )

    detections = filter_protozoa(
        separated,
        gray.shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )

    table = []
    for i, d in enumerate(detections, start=1):
        x, y, w, h = d["bbox"]
        table.append({
            "No": i,
            "X": int(round(d["x"])),
            "Y": int(round(d["y"])),
            "Area Badan": round(d["area"], 1),
            "Lebar": w,
            "Tinggi": h,
            "Circularity": round(d["circularity"], 3),
            "Aspect Ratio": round(d["aspect"], 3),
            "Solidity": round(d["solidity"], 3),
            "Extent": round(d["extent"], 3),
            "Score": round(d["score"], 2)
        })

    debug = {
        "Gray": gray,
        "Excess Green-Blue": exg,
        "Top-hat Body": top_response,
        "Black-hat Outline": black_response,
        "Whole Body Saliency": saliency,
        "Whole Body Mask": mask,
        "Separated Mask": separated,
    }

    return {
        "count": len(detections),
        "detections": detections,
        "table": table,
        "debug": debug
    }


def draw_detections(image_rgb, detections):
    output = image_rgb.copy()

    for i, d in enumerate(detections, start=1):
        contour = d.get("contour")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        x = int(round(d["x"]))
        y = int(round(d["y"]))
        r = int(d["r"])

        cv2.circle(output, (x, y), max(5, r), (255, 0, 0), 1)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)

        cv2.putText(
            output,
            str(i),
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
    images = [to_rgb(debug[n]) for n in names]

    target_w = 320
    target_h = 190
    rendered = []

    for name, image in zip(names, images):
        image = cv2.resize(image, (target_w, target_h))
        canvas = image.copy()

        cv2.rectangle(canvas, (0, 0), (target_w, 28), (255, 255, 255), -1)
        cv2.putText(
            canvas,
            name,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()
    min_side = min(h, w)

    report = []

    if min_side >= 300:
        report.append({"status": "baik", "message": f"Resolusi cukup baik: {w} x {h} px."})
    elif min_side >= 200:
        report.append({"status": "cukup", "message": f"Resolusi cukup, tetapi lebih baik minimal 300 px pada sisi terpendek. Saat ini: {w} x {h} px."})
    else:
        report.append({"status": "kurang", "message": f"Resolusi terlalu kecil: {w} x {h} px. Detail protozoa bisa hilang."})

    if blur >= 120:
        report.append({"status": "baik", "message": f"Fokus gambar baik. Nilai ketajaman: {blur:.1f}."})
    elif blur >= 55:
        report.append({"status": "cukup", "message": f"Fokus cukup, tetapi masih bisa lebih tajam. Nilai ketajaman: {blur:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Gambar cenderung blur. Nilai ketajaman: {blur:.1f}. Tepi badan protozoa bisa sulit terbaca."})

    if contrast >= 30:
        report.append({"status": "baik", "message": f"Kontras gambar baik. Nilai kontras: {contrast:.1f}."})
    elif contrast >= 18:
        report.append({"status": "cukup", "message": f"Kontras cukup. Nilai kontras: {contrast:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Kontras rendah. Nilai kontras: {contrast:.1f}. Protozoa sulit dipisahkan dari background."})

    if 55 <= brightness <= 215:
        report.append({"status": "baik", "message": f"Pencahayaan berada pada rentang baik. Brightness: {brightness:.1f}."})
    else:
        report.append({"status": "cukup", "message": f"Pencahayaan belum ideal. Brightness: {brightness:.1f}."})

    return report
