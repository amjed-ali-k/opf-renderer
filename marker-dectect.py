import cv2
import numpy as np

image_path = "DJI_20240531160419_0006_V.JPG"

img = cv2.imread(image_path)
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect white sheets
blur = cv2.GaussianBlur(gray, (7, 7), 0)

th = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

marker_id = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 20000:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        continue

    x, y, w, h = cv2.boundingRect(approx)

    sheet = img[y : y + h, x : x + w]
    sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)

    sheet_gray = cv2.GaussianBlur(sheet_gray, (5, 5), 0)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 1200
    params.maxArea = 10000

    params.filterByCircularity = True
    params.minCircularity = 0.75

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(sheet_gray)

    # Keep largest blobs
    keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)

    centers = []

    sh, sw = sheet_gray.shape

    for kp in keypoints:
        px, py = kp.pt

        if px < sw * 0.15 or px > sw * 0.85:
            continue

        if py < sh * 0.15 or py > sh * 0.85:
            continue

        centers.append((int(px), int(py)))

        if len(centers) == 4:
            break

    if len(centers) != 4:
        continue

    # Compute orientation
    pts = np.array(centers)

    max_dist = 0
    p1 = None
    p2 = None

    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(pts[i] - pts[j])

            if d > max_dist:
                max_dist = d
                p1 = pts[i]
                p2 = pts[j]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = np.degrees(np.arctan2(dy, dx))

    marker_id += 1

    # Draw bounding box
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Draw center dots
    for cx, cy in centers:
        cv2.circle(output, (x + cx, y + cy), 8, (0, 0, 255), -1)

    # Label
    label = f"Marker {marker_id} | Angle {angle:.2f}"

    cv2.putText(
        output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

print("Markers detected:", marker_id)

cv2.imwrite("detected_markers.jpg", output)
