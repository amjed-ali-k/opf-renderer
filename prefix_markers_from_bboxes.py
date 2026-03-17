from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageDraw, ImageFont
from pyopf.io import load
from pyopf.resolve import resolve


@dataclass(frozen=True)
class CameraModel:
    image_name: str
    position: np.ndarray
    rotation_world_from_camera: np.ndarray
    focal_length_px: float
    principal_point_px: np.ndarray
    image_size_px: np.ndarray


@dataclass(frozen=True)
class Observation:
    image_name: str
    marker_id: str
    x_px: float
    y_px: float
    score: float


@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class BoundingBox:
    bbox_id: str
    bottom_x: float
    bottom_y: float
    top_x: float
    top_y: float
    bottom_z: float
    top_z: float


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPF_PATH = PROJECT_ROOT / "opf" / "project.opf"
DEFAULT_MARKER_CSV = PROJECT_ROOT / "opf" / "final.csv"
DEFAULT_BBOX_CSV = PROJECT_ROOT / "opf" / "paper_bboxes.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "opf" / "final_bbox_prefixed.csv"
DEFAULT_OUTPUT_IMAGES = PROJECT_ROOT / "modified_images_bbox_prefixed"

BOX_COLORS = [
    (239, 71, 111),
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
    (7, 59, 76),
]


def rotation_from_opk(orientation_deg: np.ndarray) -> np.ndarray:
    omega, phi, kappa = np.deg2rad(orientation_deg)
    return (
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(omega), -np.sin(omega)],
                [0.0, np.sin(omega), np.cos(omega)],
            ]
        )
        @ np.array(
            [
                [np.cos(phi), 0.0, np.sin(phi)],
                [0.0, 1.0, 0.0],
                [-np.sin(phi), 0.0, np.cos(phi)],
            ]
        )
        @ np.array(
            [
                [np.cos(kappa), -np.sin(kappa), 0.0],
                [np.sin(kappa), np.cos(kappa), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def build_camera_models(project) -> dict[str, CameraModel]:
    input_sensor_map = {sensor.id: sensor for sensor in project.input_cameras.sensors}
    calib_sensor_map = {
        sensor.id: sensor for sensor in project.calibration.calibrated_cameras.sensors
    }

    camera_models: dict[str, CameraModel] = {}
    for raw_camera, calib_camera in zip(
        project.camera_list.cameras,
        project.calibration.calibrated_cameras.cameras,
        strict=False,
    ):
        image_name = Path(raw_camera.uri).name
        sensor = input_sensor_map[calib_camera.sensor_id]
        calib_sensor = calib_sensor_map[calib_camera.sensor_id]
        camera_models[image_name] = CameraModel(
            image_name=image_name,
            position=np.asarray(calib_camera.position, dtype=float),
            rotation_world_from_camera=rotation_from_opk(
                np.asarray(calib_camera.orientation_deg, dtype=float)
            ),
            focal_length_px=float(calib_sensor.internals.focal_length_px),
            principal_point_px=np.asarray(
                calib_sensor.internals.principal_point_px, dtype=float
            ),
            image_size_px=np.asarray(sensor.image_size_px, dtype=float),
        )

    return camera_models


def fit_plane(project) -> Plane:
    points = np.asarray(
        [point.coordinates for point in project.calibration.calibrated_control_points.points],
        dtype=float,
    )
    if len(points) < 3:
        raise ValueError("OPF project does not contain enough calibrated control points")

    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin)
    normal = vh[2]
    normal /= np.linalg.norm(normal)

    camera_positions = np.asarray(
        [camera.position for camera in project.calibration.calibrated_cameras.cameras],
        dtype=float,
    )
    if np.dot(normal, camera_positions.mean(axis=0) - origin) < 0:
        normal = -normal

    return Plane(origin=origin, normal=normal)


def load_observations(csv_path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                raise ValueError(f"Invalid row {row_number} in {csv_path}: {row!r}")
            observations.append(
                Observation(
                    image_name=Path(row[0].strip()).name,
                    marker_id=row[1].strip(),
                    x_px=float(row[2].strip()),
                    y_px=float(row[3].strip()),
                    score=float(row[4].strip()) if len(row) > 4 and row[4].strip() else 1.0,
                )
            )
    return observations


def project_to_plane(camera: CameraModel, plane: Plane, x_px: float, y_px: float) -> np.ndarray:
    camera_ray = np.array(
        [
            (x_px - camera.principal_point_px[0]) / camera.focal_length_px,
            -(y_px - camera.principal_point_px[1]) / camera.focal_length_px,
            -1.0,
        ],
        dtype=float,
    )
    camera_ray /= np.linalg.norm(camera_ray)
    world_ray = camera.rotation_world_from_camera @ camera_ray

    denominator = float(np.dot(plane.normal, world_ray))
    if abs(denominator) < 1e-8:
        raise ValueError(f"Ray is parallel to the plane for image {camera.image_name}")

    distance = float(np.dot(plane.normal, plane.origin - camera.position) / denominator)
    if distance <= 0:
        raise ValueError(f"Plane intersection is behind the camera for image {camera.image_name}")

    return camera.position + distance * world_ray


def load_bboxes(bbox_csv_path: Path, default_bottom_z: float, default_top_z: float) -> list[BoundingBox]:
    with bbox_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "bottomX", "bottomY", "topX", "topY"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required bbox columns in {bbox_csv_path}: {sorted(missing)}")

        boxes: list[BoundingBox] = []
        for row in reader:
            bottom_z = float(row["bottomZ"]) if row.get("bottomZ") not in (None, "") else default_bottom_z
            top_z = float(row["topZ"]) if row.get("topZ") not in (None, "") else default_top_z
            boxes.append(
                BoundingBox(
                    bbox_id=row["id"].strip(),
                    bottom_x=float(row["bottomX"]),
                    bottom_y=float(row["bottomY"]),
                    top_x=float(row["topX"]),
                    top_y=float(row["topY"]),
                    bottom_z=bottom_z,
                    top_z=top_z,
                )
            )

    return boxes


def bbox_contains_xy(box: BoundingBox, point_world: np.ndarray) -> bool:
    return (
        box.bottom_x <= point_world[0] <= box.top_x
        and box.bottom_y <= point_world[1] <= box.top_y
    )


def assign_observations_to_bboxes(
    observations: list[Observation],
    camera_models: dict[str, CameraModel],
    plane: Plane,
    boxes: list[BoundingBox],
) -> list[dict[str, object]]:
    assigned_rows: list[dict[str, object]] = []

    for observation in observations:
        camera = camera_models.get(observation.image_name)
        if camera is None:
            raise KeyError(f"Missing OPF camera for image {observation.image_name}")

        world_point = project_to_plane(camera, plane, observation.x_px, observation.y_px)
        matching_boxes = [box for box in boxes if bbox_contains_xy(box, world_point)]

        if len(matching_boxes) > 1:
            matching_boxes.sort(
                key=lambda box: (box.top_x - box.bottom_x) * (box.top_y - box.bottom_y)
            )

        box = matching_boxes[0] if matching_boxes else None
        bbox_id = box.bbox_id if box is not None else "UNASSIGNED"
        prefixed_marker_id = (
            f"{bbox_id}_{observation.marker_id}" if box is not None else observation.marker_id
        )

        assigned_rows.append(
            {
                "image_name": observation.image_name,
                "bbox_id": bbox_id,
                "prefixed_marker_id": prefixed_marker_id,
                "marker_id": observation.marker_id,
                "x_px": observation.x_px,
                "y_px": observation.y_px,
                "score": observation.score,
                "world_point": world_point,
            }
        )

    return assigned_rows


def write_output_csv(rows: list[dict[str, object]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_name",
                "bbox_id",
                "prefixed_marker_id",
                "marker_id",
                "x_px",
                "y_px",
                "score",
                "world_x",
                "world_y",
                "world_z",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["image_name"]),
                str(item["bbox_id"]),
                str(item["marker_id"]),
                float(item["x_px"]),
                float(item["y_px"]),
            ),
        ):
            world_point = np.asarray(row["world_point"], dtype=float)
            writer.writerow(
                [
                    row["image_name"],
                    row["bbox_id"],
                    row["prefixed_marker_id"],
                    row["marker_id"],
                    f"{float(row['x_px']):.2f}",
                    f"{float(row['y_px']):.2f}",
                    f"{float(row['score']):.3f}",
                    f"{world_point[0]:.6f}",
                    f"{world_point[1]:.6f}",
                    f"{world_point[2]:.6f}",
                ]
            )


def project_world_to_image(camera: CameraModel, point_world: np.ndarray) -> np.ndarray | None:
    point_camera = camera.rotation_world_from_camera.T @ (point_world - camera.position)
    if point_camera[2] >= -1e-6:
        return None

    x_px = camera.focal_length_px * (point_camera[0] / -point_camera[2]) + camera.principal_point_px[0]
    y_px = camera.principal_point_px[1] - camera.focal_length_px * (point_camera[1] / -point_camera[2])
    return np.array([x_px, y_px], dtype=float)


def box_corners(box: BoundingBox) -> np.ndarray:
    return np.array(
        [
            [box.bottom_x, box.bottom_y, box.bottom_z],
            [box.top_x, box.bottom_y, box.bottom_z],
            [box.top_x, box.top_y, box.bottom_z],
            [box.bottom_x, box.top_y, box.bottom_z],
            [box.bottom_x, box.bottom_y, box.top_z],
            [box.top_x, box.bottom_y, box.top_z],
            [box.top_x, box.top_y, box.top_z],
            [box.bottom_x, box.top_y, box.top_z],
        ],
        dtype=float,
    )


def bbox_color(index: int) -> tuple[int, int, int]:
    return BOX_COLORS[index % len(BOX_COLORS)]


def annotate_images(
    rows: list[dict[str, object]],
    boxes: list[BoundingBox],
    camera_models: dict[str, CameraModel],
    images_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_image[str(row["image_name"])].append(row)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    font = ImageFont.load_default(size=22)
    box_index_map = {box.bbox_id: index for index, box in enumerate(sorted(boxes, key=lambda b: b.bbox_id))}

    for image_name, camera in sorted(camera_models.items()):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")
        draw = ImageDraw.Draw(annotated)

        for box in boxes:
            projected = [project_world_to_image(camera, corner) for corner in box_corners(box)]
            if any(point is None for point in projected):
                continue

            color = bbox_color(box_index_map[box.bbox_id])
            corners_2d = [tuple(point.tolist()) for point in projected if point is not None]
            for start_idx, end_idx in edges:
                draw.line((corners_2d[start_idx], corners_2d[end_idx]), fill=color, width=3)

            top_face = corners_2d[4:]
            label_x = sum(point[0] for point in top_face) / len(top_face)
            label_y = sum(point[1] for point in top_face) / len(top_face)
            label = box.bbox_id
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            draw.rectangle(
                (
                    label_x - text_width / 2 - 8,
                    label_y - text_height - 18,
                    label_x + text_width / 2 + 8,
                    label_y - 6,
                ),
                fill=(0, 0, 0),
            )
            draw.text((label_x - text_width / 2, label_y - text_height - 12), label, font=font, fill=color)

        for row in per_image.get(image_name, []):
            bbox_id = str(row["bbox_id"])
            if bbox_id == "UNASSIGNED":
                color = (255, 255, 255)
            else:
                color = bbox_color(box_index_map[bbox_id])

            x = int(round(float(row["x_px"])))
            y = int(round(float(row["y_px"])))
            label = str(row["prefixed_marker_id"])
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=color, width=3)
            draw.line((x - 12, y, x + 12, y), fill=color, width=2)
            draw.line((x, y - 12, x, y + 12), fill=color, width=2)
            draw.text((x + 10, y - 24), label, font=font, fill=color)

        annotated.save(output_dir / image_name, quality=95)


def log_to_rerun(project, rows: list[dict[str, object]], boxes: list[BoundingBox]) -> None:
    try:
        rr.init("BBox Marker Prefixing", spawn=True)
    except RuntimeError as exc:
        if "Failed to find Rerun Viewer executable" not in str(exc):
            raise
        rr.init("BBox Marker Prefixing", spawn=False)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points = project.point_cloud_objs[0].nodes[0]
    rr.log("world/points", rr.Points3D(points.position, colors=points.color))

    box_centers = []
    box_labels = []
    line_strips = []
    box_index_map = {box.bbox_id: index for index, box in enumerate(sorted(boxes, key=lambda b: b.bbox_id))}
    for box in boxes:
        corners = box_corners(box)
        box_centers.append(corners.mean(axis=0))
        box_labels.append(box.bbox_id)
        line_strips.extend(
            [
                np.array([corners[0], corners[1], corners[2], corners[3], corners[0]]),
                np.array([corners[4], corners[5], corners[6], corners[7], corners[4]]),
                np.array([corners[0], corners[4]]),
                np.array([corners[1], corners[5]]),
                np.array([corners[2], corners[6]]),
                np.array([corners[3], corners[7]]),
            ]
        )

    rr.log("world/bboxes", rr.LineStrips3D(line_strips), static=True)
    rr.log("world/bbox_centers", rr.Points3D(box_centers, labels=box_labels), static=True)

    assigned_rows = [row for row in rows if row["bbox_id"] != "UNASSIGNED"]
    if assigned_rows:
        rr.log(
            "world/assigned_markers",
            rr.Points3D(
                [np.asarray(row["world_point"], dtype=float) for row in assigned_rows],
                labels=[str(row["prefixed_marker_id"]) for row in assigned_rows],
            ),
            static=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefix markers by bbox ID using an OPF project, a raw marker CSV, and a bbox CSV. "
            "Bbox assignment uses projected world XY, so future bbox files can omit Z columns."
        )
    )
    parser.add_argument("--opf", type=Path, default=DEFAULT_OPF_PATH)
    parser.add_argument("--marker-csv", type=Path, default=DEFAULT_MARKER_CSV)
    parser.add_argument("--bbox-csv", type=Path, default=DEFAULT_BBOX_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-images", type=Path, default=DEFAULT_OUTPUT_IMAGES)
    parser.add_argument("--bbox-padding-z", type=float, default=0.05)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve(load(str(args.opf)))
    camera_models = build_camera_models(project)
    plane = fit_plane(project)
    observations = load_observations(args.marker_csv)

    world_points = [
        project_to_plane(camera_models[obs.image_name], plane, obs.x_px, obs.y_px)
        for obs in observations
    ]
    all_z = [point[2] for point in world_points]
    boxes = load_bboxes(
        args.bbox_csv,
        default_bottom_z=min(all_z) - args.bbox_padding_z,
        default_top_z=max(all_z) + args.bbox_padding_z,
    )

    rows = assign_observations_to_bboxes(observations, camera_models, plane, boxes)
    write_output_csv(rows, args.output_csv)

    if not args.skip_images:
        annotate_images(rows, boxes, camera_models, args.opf.parent / "images", args.output_images)

    if args.rerun:
        log_to_rerun(project, rows, boxes)

    assigned_count = sum(1 for row in rows if row["bbox_id"] != "UNASSIGNED")
    print(f"Wrote bbox-prefixed markers to {args.output_csv}")
    print(f"Assigned {assigned_count} of {len(rows)} markers to bbox IDs")
    if not args.skip_images:
        print(f"Wrote annotated images to {args.output_images}")
    if args.rerun:
        print("Opened Rerun viewer with world bboxes and labeled markers")


if __name__ == "__main__":
    main()
