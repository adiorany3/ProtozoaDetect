import cv2
import numpy as np
import math


def odd(v):
    v = max(3, int(round(v)))
    return v if v % 2 == 1 else v + 1


def norm_u8(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    bg_kernel = odd(max(31, min(h, w) * 0.20))
    background = cv2.medianBlur(gray, bg_kernel)
    corrected = cv2.addWeighted(gray, 1.55, background, -0.55, 18)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)

    return gray, corrected, enhanced, denoised


def foreground_map(denoised, mode="Auto"):
    h, w = denoised.shape[:2]
    base = min(h, w)

    if mode == "Auto":
        polarity = "dark" if float(np.mean(denoised)) > 120 else "bright"
    elif mode == "Objek terang di background gelap":
        polarity = "bright"
    else:
        polarity = "dark"

    sizes = [
        odd(max(13, base * 0.045)),
        odd(max(21, base * 0.075)),
        odd(max(31, base * 0.110)),
    ]

    response = np.zeros_like(denoised)

    for size in sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if polarity == "dark":
            tmp = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
        else:
            tmp = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
        response = np.maximum(response, tmp)

    response = norm_u8(response)
    response = cv2.convertScaleAbs(response, alpha=1.8, beta=0)
    response = cv2.GaussianBlur(response, (3, 3), 0)

    return response, polarity


def make_binary(fg, sensitivity=1.0):
    p = 90 - (sensitivity - 1.0) * 10
    p = float(np.clip(p, 76, 94))
    th = max(10, np.percentile(fg, p))

    _, binary_p = cv2.threshold(fg, th, 255, cv2.THRESH_BINARY)

    block = odd(max(21, min(fg.shape[:2]) * 0.09))
    adaptive = cv2.adaptiveThreshold(
        fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, -2
    )

    binary = cv2.bitwise_or(binary_p, adaptive)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)

    h, w = binary.shape[:2]
    border = max(2, int(min(h, w) * 0.006))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    return binary


def watershed_split(image_rgb, binary, enabled=True):
    if not enabled:
        return binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return binary

    _, sure_fg = cv2.threshold(dist, 0.34 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
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
    perim = cv2.arcLength(contour, True)
    circ = 0 if perim <= 0 else 4 * math.pi * area / (perim ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(1, w * h)
    aspect = max(w / max(h, 1), h / max(w, 1))

    hull = cv2.convexHull(contour)
    solidity = area / max(1, cv2.contourArea(hull))

    return area, circ, (x, y, w, h), extent, aspect, solidity


def filter_components(binary, image_shape, min_area_ratio=0.00008, max_area_ratio=0.030):
    h, w = image_shape[:2]
    img_area = h * w
    min_area = max(4, img_area * min_area_ratio)
    max_area = max(min_area + 5, img_area * max_area_ratio)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area, circ, bbox, extent, aspect, solidity = contour_features(contour)
        x, y, bw, bh = bbox

        if area < min_area or area > max_area:
            continue
        if bw < 3 or bh < 3:
            continue
        if bw > w * 0.55 or bh > h * 0.55:
            continue
        if aspect > 8.5:
            continue
        if solidity < 0.22:
            continue
        if extent < 0.08:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + bw / 2
            cy = y + bh / 2

        r = int(max(bw, bh) / 2)
        score = math.sqrt(area) * 1.2 + solidity * 20 + min(circ, 1.0) * 10

        detections.append({
            "x": float(cx), "y": float(cy), "r": int(max(r, 3)),
            "area": float(area), "bbox": (int(x), int(y), int(bw), int(bh)),
            "circularity": float(circ), "aspect": float(aspect),
            "solidity": float(solidity), "extent": float(extent),
            "score": float(score), "contour": contour
        })

    detections = sorted(detections, key=lambda d: (d["y"], d["x"]))
    return detections


def count_protozoa(image_rgb, mode="Auto", sensitivity=1.0,
                   min_area_ratio=0.00008, max_area_ratio=0.030,
                   split_touching=True):
    gray, corrected, enhanced, denoised = preprocess(image_rgb)
    fg, polarity = foreground_map(denoised, mode)
    binary = make_binary(fg, sensitivity)
    separated = watershed_split(image_rgb, binary, split_touching)

    detections = filter_components(
        separated, gray.shape,
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
            "Area": round(d["area"], 1),
            "Lebar": w,
            "Tinggi": h,
            "Circularity": round(d["circularity"], 3),
            "Aspect Ratio": round(d["aspect"], 3),
            "Solidity": round(d["solidity"], 3),
            "Score": round(d["score"], 2)
        })

    debug = {
        "Gray": gray,
        "Corrected": corrected,
        "Enhanced": enhanced,
        "Foreground Map": fg,
        "Binary": binary,
        "Watershed": separated
    }

    return {
        "count": len(detections),
        "detections": detections,
        "table": table,
        "debug": debug,
        "polarity": polarity
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

        cv2.circle(output, (x, y), max(3, r), (255, 0, 0), 1)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 2)

        cv2.putText(
            output, str(i), (x - 8, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 0), 2, cv2.LINE_AA
        )

    return output


def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def make_debug_grid(debug):
    names = list(debug.keys())
    images = [to_rgb(debug[n]) for n in names]

    target_w, target_h = 360, 240
    rendered = []

    for name, image in zip(names, images):
        image = cv2.resize(image, (target_w, target_h))
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (target_w, 30), (255, 255, 255), -1)
        cv2.putText(canvas, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(canvas)

    return np.vstack([np.hstack(rendered[:3]), np.hstack(rendered[3:6])])


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
        report.append({"status": "kurang", "message": f"Gambar cenderung blur. Nilai ketajaman: {blur:.1f}. Tepi protozoa bisa sulit terbaca."})

    if contrast >= 38:
        report.append({"status": "baik", "message": f"Kontras gambar baik. Nilai kontras: {contrast:.1f}."})
    elif contrast >= 22:
        report.append({"status": "cukup", "message": f"Kontras cukup. Nilai kontras: {contrast:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Kontras rendah. Nilai kontras: {contrast:.1f}. Protozoa sulit dipisahkan dari background."})

    if 60 <= brightness <= 205:
        report.append({"status": "baik", "message": f"Pencahayaan berada pada rentang baik. Brightness: {brightness:.1f}."})
    elif 40 <= brightness < 60 or 205 < brightness <= 225:
        report.append({"status": "cukup", "message": f"Pencahayaan cukup, tetapi belum ideal. Brightness: {brightness:.1f}."})
    else:
        report.append({"status": "kurang", "message": f"Pencahayaan kurang ideal. Brightness: {brightness:.1f}. Hindari gambar terlalu gelap/terlalu terang."})

    return report
