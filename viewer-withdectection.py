#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path

import rerun as rr
from pyopf.io import load
from pyopf.resolve import resolve


OPF_PATH = Path("opf/project.opf")

# Real distance between neighboring target centers in one 4-target marker.
# If unknown, keep any reasonable value. Tilt angle is still usually OK.
MARKER_CENTER_SPACING_M = 0.10

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_sheets(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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


def warp_sheet(img_bgr, quad, out_w=600, out_h=800):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(quad, dst)
    Minv = cv2.getPerspectiveTransform(dst, quad)

    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
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


def detect_4_centers_on_warp(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8)

    H, W = gray.shape

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

            if area < 500 or area > 6000:
                continue

            if not (ex - W * 0.22 <= cx <= ex + W * 0.22):
                continue
            if not (ey - H * 0.22 <= cy <= ey + H * 0.22):
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            circ = component_circularity(comp_mask)
            if circ < 0.65:
                continue

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


def compute_angle_from_4_centers(centers):
    pts = np.array(centers, dtype=np.float32)

    ordered = order_quad_points(pts)
    tl, tr, br, bl = ordered

    top_vec = tr - tl
    bottom_vec = br - bl

    vec = (top_vec + bottom_vec) / 2.0

    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return angle, ordered


def get_image_path(opf_path, camera_uri):
    # First try the URI relative to project.opf, as OPF intends.
    p1 = (opf_path.parent / Path(camera_uri)).resolve()
    if p1.exists():
        return p1

    # Fallback for your current layout: opf/images/<filename>
    p2 = (opf_path.parent / "images" / Path(camera_uri).name).resolve()
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Could not resolve image path for URI: {camera_uri}")


def focal_xy(focal_length_px):
    f = np.asarray(focal_length_px, dtype=np.float64).reshape(-1)
    if f.size == 1:
        return float(f[0]), float(f[0])
    return float(f[0]), float(f[1])


def principal_xy(principal_point_px):
    p = np.asarray(principal_point_px, dtype=np.float64).reshape(-1)
    return float(p[0]), float(p[1])


def cv_cam_to_rub(v):
    """Convert vector from OpenCV camera coords (x right, y down, z forward)
    to RUB camera coords (x right, y up, z back)."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], -v[1], -v[2]], dtype=np.float64)


def pose_marker_from_centers(ordered_centers, camera_matrix, dist_coeffs):
    s = MARKER_CENTER_SPACING_M

    # Marker-local coordinates for the 4 detected target centers.
    # Ordered as: tl, tr, br, bl
    object_points = np.array(
        [
            [-s / 2.0, -s / 2.0, 0.0],
            [s / 2.0, -s / 2.0, 0.0],
            [s / 2.0, s / 2.0, 0.0],
            [-s / 2.0, s / 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    image_points = np.asarray(ordered_centers, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    R_obj_to_cam_cv, _ = cv2.Rodrigues(rvec)
    return object_points, R_obj_to_cam_cv, tvec.reshape(3)


def marker_world_geometry(
    object_points,
    R_obj_to_cam_cv,
    tvec_cv,
    cam_rot_rub_to_world,
    cam_pos_world,
):
    # Marker points in OpenCV camera coordinates
    pts_cam_cv = (R_obj_to_cam_cv @ object_points.T).T + tvec_cv.reshape(1, 3)

    # Convert to RUB camera coordinates
    pts_cam_rub = np.array([cv_cam_to_rub(p) for p in pts_cam_cv], dtype=np.float64)

    # Camera -> world
    pts_world = (cam_rot_rub_to_world @ pts_cam_rub.T).T + np.asarray(
        cam_pos_world, dtype=np.float64
    )

    # Marker normal in object local coordinates
    n_obj = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Object -> OpenCV cam
    n_cam_cv = R_obj_to_cam_cv @ n_obj

    # OpenCV cam -> RUB cam -> world
    n_cam_rub = cv_cam_to_rub(n_cam_cv)
    n_world = cam_rot_rub_to_world @ n_cam_rub
    n_world = n_world / np.linalg.norm(n_world)

    # Keep normal pointing upward for a more intuitive tilt angle
    if n_world[2] < 0:
        n_world = -n_world

    return pts_world, n_world


def angle_to_ground(normal_world):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = np.clip(np.dot(normal_world, up), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def close_polyline(pts):
    pts = np.asarray(pts, dtype=np.float32)
    return np.vstack([pts, pts[0:1]])


def detect_markers_and_annotate(
    img_bgr,
    camera_matrix,
    dist_coeffs,
    cam_rot_rub_to_world,
    cam_pos_world,
):
    output = img_bgr.copy()
    sheets = detect_sheets(img_bgr)

    marker_results = []
    marker_id = 0

    for sheet in sheets:
        quad = sheet["quad"]
        warped, _, Minv = warp_sheet(img_bgr, quad, out_w=600, out_h=800)
        centers_warp, _ = detect_4_centers_on_warp(warped)
        centers_img = map_points_back(centers_warp, Minv)

        found_count = sum(c is not None for c in centers_img)
        if found_count < 4:
            cv2.polylines(output, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            continue

        angle_2d_deg, ordered_centers = compute_angle_from_4_centers(centers_img)

        pose = pose_marker_from_centers(ordered_centers, camera_matrix, dist_coeffs)
        if pose is None:
            cv2.polylines(output, [quad.astype(np.int32)], True, (0, 165, 255), 2)
            continue

        object_points, R_obj_to_cam_cv, tvec_cv = pose
        world_pts, normal_world = marker_world_geometry(
            object_points,
            R_obj_to_cam_cv,
            tvec_cv,
            cam_rot_rub_to_world,
            cam_pos_world,
        )
        tilt_deg = angle_to_ground(normal_world)

        marker_id += 1

        cv2.polylines(output, [quad.astype(np.int32)], True, (0, 255, 0), 3)

        for c in centers_img:
            cx, cy = int(round(c[0])), int(round(c[1]))
            cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(output, (cx, cy), 16, (255, 0, 0), 2)

        ordered_int = np.asarray(ordered_centers, dtype=np.int32)
        cv2.polylines(output, [ordered_int], True, (255, 255, 0), 2)

        label = f"M{marker_id} | img {angle_2d_deg:.1f} | tilt {tilt_deg:.1f}"
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

        marker_results.append(
            {
                "marker_id": marker_id,
                "quad_img": np.asarray(quad, dtype=np.float32),
                "centers_img": np.asarray(centers_img, dtype=np.float32),
                "ordered_img": np.asarray(ordered_centers, dtype=np.float32),
                "image_angle_deg": float(angle_2d_deg),
                "tilt_deg": float(tilt_deg),
                "world_pts": np.asarray(world_pts, dtype=np.float32),
                "normal_world": np.asarray(normal_world, dtype=np.float32),
            }
        )

    return output, marker_results


def main():
    project = resolve(load(str(OPF_PATH)))

    rr.init("opf_marker_plane_viewer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Point cloud
    points = project.point_cloud_objs[0].nodes[0]
    rr.log(
        "world/points",
        rr.Points3D(points.position, colors=points.color),
        static=True,
    )

    sensor_map = {s.id: s for s in project.input_cameras.sensors}
    calib_sensor_map = {s.id: s for s in project.calibration.calibrated_cameras.sensors}

    positions = []

    frame_idx = 0

    for camera, calib_camera in zip(
        project.camera_list.cameras,
        project.calibration.calibrated_cameras.cameras,
        strict=False,
    ):
        uri_str = str(camera.uri)
        if not uri_str.lower().endswith(VALID_IMAGE_EXTS):
            continue

        sensor = sensor_map[calib_camera.sensor_id]
        calib_sensor = calib_sensor_map[calib_camera.sensor_id]

        if calib_sensor.internals.type != "perspective":
            continue

        img_path = get_image_path(OPF_PATH, uri_str)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        rr.set_time("image", sequence=frame_idx)
        frame_idx += 1

        cam_pos_world = np.asarray(calib_camera.position, dtype=np.float64)
        positions.append(cam_pos_world)

        omega, phi, kappa = np.deg2rad(calib_camera.orientation_deg)

        # OPF omega/phi/kappa -> RUB camera-to-world rotation
        cam_rot_rub_to_world = (
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(omega), -np.sin(omega)],
                    [0, np.sin(omega), np.cos(omega)],
                ],
                dtype=np.float64,
            )
            @ np.array(
                [
                    [np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)],
                ],
                dtype=np.float64,
            )
            @ np.array(
                [
                    [np.cos(kappa), -np.sin(kappa), 0],
                    [np.sin(kappa), np.cos(kappa), 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        )

        entity = "world/camera"

        rr.log(
            entity,
            rr.Transform3D(
                translation=cam_pos_world,
                mat3x3=cam_rot_rub_to_world,
            ),
        )

        rr.log(
            entity + "/image",
            rr.Pinhole(
                resolution=sensor.image_size_px,
                focal_length=calib_sensor.internals.focal_length_px,
                principal_point=calib_sensor.internals.principal_point_px,
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )

        fx, fy = focal_xy(calib_sensor.internals.focal_length_px)
        cx, cy = principal_xy(calib_sensor.internals.principal_point_px)

        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        annotated_bgr, marker_results = detect_markers_and_annotate(
            img_bgr,
            camera_matrix,
            dist_coeffs,
            cam_rot_rub_to_world,
            cam_pos_world,
        )

        rr.log(
            entity + "/image/rgb",
            rr.Image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)),
        )

        for m in marker_results:
            marker_entity = f"world/markers/marker_{m['marker_id']}"

            rr.log(
                marker_entity + "/square",
                rr.LineStrips3D([close_polyline(m["world_pts"])]),
            )

            rr.log(
                marker_entity + "/points",
                rr.Points3D(m["world_pts"], colors=[255, 0, 0]),
            )

            center_world = m["world_pts"].mean(axis=0)
            normal_vec = m["normal_world"] * (MARKER_CENTER_SPACING_M * 0.75)

            rr.log(
                marker_entity + "/normal",
                rr.Arrows3D(
                    origins=[center_world],
                    vectors=[normal_vec],
                    colors=[0, 255, 0],
                ),
            )

            print(
                f"{img_path.name} | marker {m['marker_id']} | "
                f"image_angle={m['image_angle_deg']:.2f} deg | "
                f"tilt={m['tilt_deg']:.2f} deg"
            )

    if len(positions) >= 2:
        positions = np.asarray(positions, dtype=np.float32)
        rr.log(
            "world/camera_path",
            rr.LineStrips3D([positions]),
            static=True,
        )


if __name__ == "__main__":
    main()
