import cv2
import numpy as np
import math


def _odd(v):
    v = max(3, int(round(v)))
    return v if v % 2 == 1 else v + 1


def _norm_u8(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _fill_holes(mask):
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def _remove_tiny_components(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    bg_kernel = _odd(max(51, min(h, w) * 0.22))
    background = cv2.medianBlur(gray, bg_kernel)
    corrected = cv2.addWeighted(gray, 1.55, background, -0.55, 15)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    denoised = cv2.bilateralFilter(enhanced, 9, 60, 60)
    return gray, corrected, enhanced, denoised


def body_saliency(image_rgb, denoised):
    """
    Saliency badan protozoa utuh.
    Untuk sample hijau-kebiruan, channel Excess Green paling kuat.
    Untuk sample lain, top-hat dan black-hat tetap membantu.
    """
    r, g, b = cv2.split(image_rgb)

    exg = 2.0 * g.astype(np.float32) - 0.95 * r.astype(np.float32) - 0.75 * b.astype(np.float32)
    exg = np.clip(exg, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    _, sat, val = cv2.split(hsv)

    h, w = denoised.shape[:2]
    base = min(h, w)

    sizes = [
        _odd(max(21, base * 0.045)),
        _odd(max(35, base * 0.075)),
        _odd(max(55, base * 0.115)),
    ]

    top_body = np.zeros_like(denoised)
    outline = np.zeros_like(denoised)

    for s in sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        top_body = np.maximum(top_body, cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, k))
        outline = np.maximum(outline, cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k))

    exg = _norm_u8(exg)
    top_body = _norm_u8(top_body)
    outline = _norm_u8(outline)

    saliency = (
        0.90 * exg.astype(np.float32)
        + 0.25 * sat.astype(np.float32)
        + 0.38 * top_body.astype(np.float32)
        + 0.20 * outline.astype(np.float32)
    )

    saliency = _norm_u8(saliency)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    saliency = cv2.convertScaleAbs(saliency, alpha=1.55, beta=0)

    return saliency, exg, top_body, outline


def whole_body_mask(saliency, body_threshold=25, merge_strength=0.85, min_noise_area=350):
    """
    Mask dibuat dari badan utuh. Closing tidak terlalu besar agar protozoa yang berdekatan
    tidak langsung bergabung menjadi satu cluster besar.
    """
    h, w = saliency.shape[:2]
    base = min(h, w)

    _, mask = cv2.threshold(saliency, body_threshold, 255, cv2.THRESH_BINARY)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(base * 0.006), _odd(base * 0.006)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    close_size = _odd(max(7, base * 0.018 * merge_strength))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    mask = _remove_tiny_components(mask, min_noise_area)
    mask = _fill_holes(mask)

    # Smooth ringan.
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_smooth, iterations=1)

    border = max(1, int(base * 0.003))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    return mask


def split_touching_objects(image_rgb, mask, split_enabled=True, split_strength=0.34, min_marker_area=450):
    """
    Split memakai distance transform. Default lebih agresif daripada v3,
    tapi marker kecil tetap dibuang agar badan tidak pecah karena tekstur internal.
    """
    if not split_enabled:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask

    _, sure_fg = cv2.threshold(dist, split_strength * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_fg = _remove_tiny_components(sure_fg, min_marker_area)

    if cv2.countNonZero(sure_fg) == 0:
        return mask

    sure_bg = cv2.dilate(clean, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    sep = np.zeros(mask.shape, dtype=np.uint8)
    sep[markers > 1] = 255

    sep = cv2.morphologyEx(sep, cv2.MORPH_CLOSE, k, iterations=1)
    return sep


def contour_features(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    circularity = 0 if peri <= 0 else 4 * math.pi * area / (peri * peri)

    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(1, w * h)
    aspect = max(w / max(h, 1), h / max(w, 1))

    hull = cv2.convexHull(contour)
    solidity = area / max(1, cv2.contourArea(hull))
    return area, circularity, (x, y, w, h), extent, aspect, solidity


def filter_protozoa(mask, image_shape, min_area_ratio=0.00055, max_area_ratio=0.070, count_edge_objects=True):
    h, w = image_shape[:2]
    image_area = h * w

    min_area = max(420, image_area * min_area_ratio)
    max_area = max(min_area + 50, image_area * max_area_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area, circ, bbox, extent, aspect, solidity = contour_features(contour)
        x, y, bw, bh = bbox

        if area < min_area or area > max_area:
            continue
        if bw < 18 or bh < 18:
            continue
        if aspect > 5.2:
            continue
        if solidity < 0.24:
            continue
        if extent < 0.12:
            continue

        # Opsional: jangan hitung objek yang hanya terlihat sangat sedikit di pinggir.
        if not count_edge_objects:
            border = max(4, int(min(h, w) * 0.012))
            touches_border = x <= border or y <= border or (x + bw) >= w - border or (y + bh) >= h - border
            if touches_border and area < min_area * 1.8:
                continue

        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = x + bw / 2
            cy = y + bh / 2

        score = math.sqrt(area) * 1.2 + solidity * 20 + min(circ, 1.0) * 8 + extent * 8

        detections.append({
            "x": float(cx),
            "y": float(cy),
            "r": int(max(bw, bh) / 2),
            "area": float(area),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "circularity": float(circ),
            "aspect": float(aspect),
            "solidity": float(solidity),
            "extent": float(extent),
            "score": float(score),
            "contour": contour
        })

    return sorted(detections, key=lambda d: (d["y"], d["x"]))


def count_protozoa(
    image_rgb,
    body_threshold=25,
    merge_strength=0.85,
    split_touching=True,
    split_strength=0.34,
    min_area_ratio=0.00055,
    max_area_ratio=0.070,
    count_edge_objects=True
):
    gray, corrected, enhanced, denoised = preprocess(image_rgb)
    saliency, exg, top_body, outline = body_saliency(image_rgb, denoised)

    mask = whole_body_mask(
        saliency,
        body_threshold=body_threshold,
        merge_strength=merge_strength
    )

    separated = split_touching_objects(
        image_rgb,
        mask,
        split_enabled=split_touching,
        split_strength=split_strength
    )

    detections = filter_protozoa(
        separated,
        gray.shape,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        count_edge_objects=count_edge_objects
    )

    table = []
    for i, d in enumerate(detections, start=1):
        x, y, bw, bh = d["bbox"]
        table.append({
            "No": i,
            "X": int(round(d["x"])),
            "Y": int(round(d["y"])),
            "Area Badan": round(d["area"], 1),
            "Lebar": bw,
            "Tinggi": bh,
            "Circularity": round(d["circularity"], 3),
            "Aspect Ratio": round(d["aspect"], 3),
            "Solidity": round(d["solidity"], 3),
            "Extent": round(d["extent"], 3),
            "Score": round(d["score"], 2)
        })

    debug = {
        "Gray": gray,
        "Excess Green": exg,
        "Top-hat Body": top_body,
        "Outline": outline,
        "Body Saliency": saliency,
        "Whole Body Mask": mask,
        "Separated Mask": separated,
    }

    return {"count": len(detections), "detections": detections, "table": table, "debug": debug}


def draw_detections(image_rgb, detections):
    out = image_rgb.copy()

    for i, d in enumerate(detections, start=1):
        contour = d.get("contour")
        if contour is not None:
            cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)

        x = int(round(d["x"]))
        y = int(round(d["y"]))

        cv2.circle(out, (x, y), 3, (255, 0, 0), 3)
        cv2.putText(out, str(i), (x - 10, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

    return out


def _to_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def make_debug_grid(debug):
    names = list(debug.keys())
    imgs = [_to_rgb(debug[n]) for n in names]

    target_w, target_h = 320, 190
    tiles = []

    for name, img in zip(names, imgs):
        img = cv2.resize(img, (target_w, target_h))
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (target_w, 28), (255, 255, 255), -1)
        cv2.putText(canvas, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)

    while len(tiles) % 4 != 0:
        tiles.append(np.zeros_like(tiles[0]))

    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
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
        report.append(("cukup", f"Resolusi sebaiknya minimal 300 px pada sisi terpendek. Saat ini: {w} x {h} px."))

    if blur >= 100:
        report.append(("baik", f"Fokus gambar cukup baik. Nilai ketajaman: {blur:.1f}."))
    elif blur >= 45:
        report.append(("cukup", f"Fokus cukup, tetapi bisa lebih tajam. Nilai ketajaman: {blur:.1f}."))
    else:
        report.append(("kurang", f"Gambar cenderung blur. Nilai ketajaman: {blur:.1f}."))

    if contrast >= 25:
        report.append(("baik", f"Kontras cukup baik. Nilai kontras: {contrast:.1f}."))
    else:
        report.append(("cukup", f"Kontras masih rendah. Nilai kontras: {contrast:.1f}."))

    if 55 <= brightness <= 220:
        report.append(("baik", f"Pencahayaan berada pada rentang cukup. Brightness: {brightness:.1f}."))
    else:
        report.append(("cukup", f"Pencahayaan belum ideal. Brightness: {brightness:.1f}."))

    return report
