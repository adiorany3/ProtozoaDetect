import cv2
import numpy as np
import math


def odd(v):
    v = max(3, int(round(v)))
    return v if v % 2 == 1 else v + 1


def norm_u8(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fill_holes(mask):
    h, w = mask.shape[:2]
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def remove_small(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def preprocess(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    bg_kernel = odd(max(21, min(h, w) * 0.18))
    background = cv2.medianBlur(gray, bg_kernel)

    corrected = cv2.addWeighted(gray, 1.55, background, -0.55, 12)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)
    return gray, corrected, enhanced, denoised


def saliency_maps(image_rgb, denoised):
    """
    Multi-sample saliency:
    - green/blue protozoa pada sample pertama,
    - brown/gray protozoa pada sample kedua,
    - outline gelap,
    - isi badan yang sedikit lebih gelap/terang dari background.
    """
    r, g, b = cv2.split(image_rgb)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_ch, sat, val = cv2.split(hsv)

    # Untuk protozoa hijau/kebiruan.
    ex_green = 2.0 * g.astype(np.float32) - 0.90 * r.astype(np.float32) - 0.70 * b.astype(np.float32)
    ex_green = np.clip(ex_green, 0, 255).astype(np.uint8)

    # Untuk protozoa coklat/abu-abu pada background krem.
    # Objek cenderung lebih gelap dan lebih bersaturasi dibanding background.
    brown_score = (
        0.70 * sat.astype(np.float32)
        + 0.70 * (255 - val).astype(np.float32)
        + 0.20 * r.astype(np.float32)
    )
    brown_score = norm_u8(brown_score)

    h0, w0 = denoised.shape[:2]
    base = min(h0, w0)

    sizes = [
        odd(max(13, base * 0.045)),
        odd(max(21, base * 0.080)),
        odd(max(31, base * 0.120)),
    ]

    top_body = np.zeros_like(denoised)
    black_outline = np.zeros_like(denoised)

    for size in sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        top_body = np.maximum(top_body, cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, k))
        black_outline = np.maximum(black_outline, cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k))

    ex_green = norm_u8(ex_green)
    top_body = norm_u8(top_body)
    black_outline = norm_u8(black_outline)

    color_score = np.maximum(ex_green, brown_score)

    saliency = (
        0.52 * color_score.astype(np.float32)
        + 0.38 * black_outline.astype(np.float32)
        + 0.28 * top_body.astype(np.float32)
        + 0.22 * sat.astype(np.float32)
    )

    saliency = norm_u8(saliency)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    saliency = cv2.convertScaleAbs(saliency, alpha=1.55, beta=0)

    return saliency, color_score, ex_green, brown_score, top_body, black_outline


def body_mask(saliency, body_threshold=28, merge_strength=0.80, min_noise_area=35):
    """
    Body mask dibuat cukup sensitif untuk gambar kecil seperti images.jpeg.
    """
    h, w = saliency.shape[:2]
    base = min(h, w)

    _, manual = cv2.threshold(saliency, body_threshold, 255, cv2.THRESH_BINARY)

    otsu_value, otsu = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu_value > body_threshold + 35:
        mask = manual
    else:
        mask = cv2.bitwise_or(manual, otsu)

    open_size = odd(max(3, base * 0.008))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    close_size = odd(max(5, base * 0.025 * merge_strength))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    mask = fill_holes(mask)
    mask = remove_small(mask, min_noise_area)

    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_smooth, iterations=1)

    return mask


def split_objects(image_rgb, mask, enabled=True, split_strength=0.34, min_marker_area=25):
    if not enabled:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask

    _, sure_fg = cv2.threshold(dist, split_strength * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = remove_small(sure_fg, min_marker_area)

    if cv2.countNonZero(sure_fg) == 0:
        return mask

    sure_bg = cv2.dilate(clean, k, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[markers > 1] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


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


def filter_protozoa(mask, image_shape, min_area_ratio=0.00035, max_area_ratio=0.090, count_edge_objects=True):
    h, w = image_shape[:2]
    image_area = h * w

    min_area = max(20, image_area * min_area_ratio)
    max_area = max(min_area + 20, image_area * max_area_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area, circ, bbox, extent, aspect, solidity = contour_features(contour)
        x, y, bw, bh = bbox

        if area < min_area or area > max_area:
            continue
        if bw < 6 or bh < 6:
            continue
        if aspect > 6.5:
            continue
        if solidity < 0.20:
            continue
        if extent < 0.10:
            continue

        if not count_edge_objects:
            border = max(2, int(min(h, w) * 0.015))
            touches_border = x <= border or y <= border or x + bw >= w - border or y + bh >= h - border
            if touches_border:
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
    body_threshold=28,
    merge_strength=0.80,
    split_touching=True,
    split_strength=0.34,
    min_area_ratio=0.00035,
    max_area_ratio=0.090,
    count_edge_objects=True
):
    gray, corrected, enhanced, denoised = preprocess(image_rgb)
    saliency, color_score, ex_green, brown_score, top_body, outline = saliency_maps(image_rgb, denoised)

    mask = body_mask(
        saliency,
        body_threshold=body_threshold,
        merge_strength=merge_strength,
        min_noise_area=max(20, int(gray.size * 0.00020))
    )

    separated = split_objects(
        image_rgb,
        mask,
        enabled=split_touching,
        split_strength=split_strength,
        min_marker_area=max(20, int(gray.size * 0.00018))
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
            "Area": round(d["area"], 1),
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
        "Color Score": color_score,
        "Green Score": ex_green,
        "Brown Score": brown_score,
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
        cv2.putText(out, str(i), (x - 9, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2, cv2.LINE_AA)

    return out


def to_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def make_debug_grid(debug):
    names = list(debug.keys())
    imgs = [to_rgb(debug[n]) for n in names]
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

    return np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])


def image_quality_report(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()

    report = []

    if min(h, w) >= 250:
        report.append(("baik", f"Resolusi cukup baik: {w} x {h} px."))
    else:
        report.append(("cukup", f"Resolusi kecil: {w} x {h} px. Hasil tetap bisa, tapi lebih baik gunakan gambar lebih besar."))

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
