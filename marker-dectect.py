image_path = "DJI_20240531160419_0006_V.JPG"
import cv2
import numpy as np


img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect white sheets
blur = cv2.GaussianBlur(gray, (7, 7), 0)

th = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sheet_id = 0

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

    sheet_id += 1

    # ---- Detect center dots ----

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 2000

    params.filterByCircularity = True
    params.minCircularity = 0.6

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(sheet_gray)

    centers = []

    for kp in keypoints:
        cx = int(kp.pt[0])
        cy = int(kp.pt[1])

        centers.append((cx, cy))

        cv2.circle(sheet, (cx, cy), 6, (0, 255, 0), 2)

    print(f"Sheet {sheet_id} center dots:", len(centers))

    # ---- Compute sheet orientation ----

    if len(centers) >= 4:
        pts = np.array(centers[:4], dtype=np.float32)

        rect = cv2.minAreaRect(pts)

        angle = rect[2]

        print("Sheet angle:", angle)

        cv2.putText(
            sheet,
            f"Angle {angle:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(f"sheet_{sheet_id}.jpg", sheet)
