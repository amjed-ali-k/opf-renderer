import cv2
import numpy as np


image_path = "DJI_20240531160419_0006_V.JPG"
output_path = "detected_markers_final.jpg"


def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_sheets(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -5,
    )

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sheets = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        quad = approx.reshape(4, 2)
        quad = order_quad_points(quad)

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)

        if ratio < 0.45 or ratio > 2.2:
            continue

        sheets.append(
            {
                "quad": quad,
                "bbox": (x, y, w, h),
            }
        )

    return sheets


def warp_sheet(img, quad, out_w=600, out_h=800):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(quad, dst)
    Minv = cv2.getPerspectiveTransform(dst, quad)

    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped, M, Minv


def component_circularity(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return 0.0
    return 4.0 * np.pi * area / (peri * peri)


def detect_4_centers_on_warp(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark print on white sheet
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8)

    H, W = gray.shape

    # Expected centers of the 4 markers in normalized sheet coordinates
    expected = [
        (W * 0.25, H * 0.25),  # top-left
        (W * 0.75, H * 0.25),  # top-right
        (W * 0.25, H * 0.75),  # bottom-left
        (W * 0.75, H * 0.75),  # bottom-right
    ]

    detected = []

    for ex, ey in expected:
        best = None

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            # keep only medium dark blobs
            if area < 500 or area > 6000:
                continue

            # stay near this quadrant
            if not (ex - W * 0.22 <= cx <= ex + W * 0.22):
                continue
            if not (ey - H * 0.22 <= cy <= ey + H * 0.22):
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            circ = component_circularity(comp_mask)
            if circ < 0.65:
                continue

            # Prefer circular blobs near expected quadrant center
            dist = np.hypot(cx - ex, cy - ey)
            score = circ * 10.0 + area * 0.001 - dist * 0.02

            if best is None or score > best[0]:
                best = (score, (float(cx), float(cy)))

        detected.append(best[1] if best is not None else None)

    return detected, th


def map_points_back(points, Minv):
    mapped = []
    for p in points:
        if p is None:
            mapped.append(None)
            continue

        px = np.array([[[p[0], p[1]]]], dtype=np.float32)
        world = cv2.perspectiveTransform(px, Minv)[0, 0]
        mapped.append((float(world[0]), float(world[1])))

    return mapped


def compute_sheet_angle_from_quad(quad):
    # use top edge of sheet
    tl, tr, br, bl = quad
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def main():
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output = img.copy()

    sheets = detect_sheets(img)
    print("Sheets found:", len(sheets))

    marker_id = 0

    for sheet in sheets:
        quad = sheet["quad"]
        x, y, w, h = sheet["bbox"]

        warped, M, Minv = warp_sheet(img, quad, out_w=600, out_h=800)
        centers_warp, debug_th = detect_4_centers_on_warp(warped)
        centers_img = map_points_back(centers_warp, Minv)

        found_count = sum(c is not None for c in centers_img)
        if found_count < 4:
            # still draw the sheet box, but skip marker labeling if centers incomplete
            cv2.polylines(output, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            continue

        marker_id += 1

        angle = compute_sheet_angle_from_quad(quad)

        # Draw sheet outline
        cv2.polylines(output, [quad.astype(np.int32)], True, (0, 255, 0), 3)

        # Draw detected centers
        for c in centers_img:
            cx, cy = int(round(c[0])), int(round(c[1]))
            cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(output, (cx, cy), 16, (255, 0, 0), 2)

        # Label
        label = f"Marker {marker_id} | Angle {angle:.2f}"
        text_x = int(quad[0][0])
        text_y = max(30, int(quad[0][1]) - 10)

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    print("Markers detected:", marker_id)
    cv2.imwrite(output_path, output)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
