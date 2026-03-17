from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
class BoundingBox:
    box_id: str
    bottom: np.ndarray
    top: np.ndarray


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPF_PATH = PROJECT_ROOT / "opf" / "project.opf"
DEFAULT_GROUPED_CSV = PROJECT_ROOT / "opf" / "final_grouped.csv"
DEFAULT_BBOX_CSV = PROJECT_ROOT / "opf" / "paper_bboxes.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modified_images_bboxes"

GROUP_COLORS = [
    (239, 71, 111),
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
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
                calib_sensor.internals.principal_point_px,
                dtype=float,
            ),
            image_size_px=np.asarray(sensor.image_size_px, dtype=float),
        )

    return camera_models


def load_grouped_rows(grouped_csv_path: Path) -> list[dict[str, str]]:
    with grouped_csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_bounding_boxes(
    rows: list[dict[str, str]],
    padding_xy: float,
    padding_z: float,
) -> list[BoundingBox]:
    grouped_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped_points[row["group_id"]].append(
            np.array(
                [
                    float(row["world_x"]),
                    float(row["world_y"]),
                    float(row["world_z"]),
                ],
                dtype=float,
            )
        )

    boxes: list[BoundingBox] = []
    for box_id in sorted(grouped_points):
        points = np.asarray(grouped_points[box_id], dtype=float)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        bottom = np.array(
            [mins[0] - padding_xy, mins[1] - padding_xy, mins[2] - padding_z],
            dtype=float,
        )
        top = np.array(
            [maxs[0] + padding_xy, maxs[1] + padding_xy, maxs[2] + padding_z],
            dtype=float,
        )
        boxes.append(BoundingBox(box_id=box_id, bottom=bottom, top=top))

    return boxes


def write_bbox_csv(boxes: list[BoundingBox], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "bottomX", "bottomY", "bottomZ", "topX", "topY", "topZ"])
        for box in boxes:
            writer.writerow(
                [
                    box.box_id,
                    f"{box.bottom[0]:.6f}",
                    f"{box.bottom[1]:.6f}",
                    f"{box.bottom[2]:.6f}",
                    f"{box.top[0]:.6f}",
                    f"{box.top[1]:.6f}",
                    f"{box.top[2]:.6f}",
                ]
            )


def box_corners(box: BoundingBox) -> np.ndarray:
    x0, y0, z0 = box.bottom
    x1, y1, z1 = box.top
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def project_world_to_image(
    camera: CameraModel, point_world: np.ndarray
) -> np.ndarray | None:
    point_camera = camera.rotation_world_from_camera.T @ (point_world - camera.position)
    if point_camera[2] >= -1e-6:
        return None

    x_px = (
        camera.focal_length_px * (point_camera[0] / -point_camera[2])
        + camera.principal_point_px[0]
    )
    y_px = camera.principal_point_px[1] - camera.focal_length_px * (
        point_camera[1] / -point_camera[2]
    )
    return np.array([x_px, y_px], dtype=float)


def annotate_images(
    boxes: list[BoundingBox],
    camera_models: dict[str, CameraModel],
    images_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=24)
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

    for image_name, camera in sorted(camera_models.items()):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")

        draw = ImageDraw.Draw(annotated)

        for box in boxes:
            color_index = int(box.box_id.split("_")[1]) - 1
            color = GROUP_COLORS[color_index % len(GROUP_COLORS)]
            projected = [
                project_world_to_image(camera, corner) for corner in box_corners(box)
            ]
            if any(point is None for point in projected):
                continue

            points_2d = [
                tuple(point.tolist()) for point in projected if point is not None
            ]
            for start_idx, end_idx in edges:
                draw.line(
                    (points_2d[start_idx], points_2d[end_idx]), fill=color, width=4
                )

            top_face = points_2d[4:]
            center_x = sum(point[0] for point in top_face) / len(top_face)
            center_y = sum(point[1] for point in top_face) / len(top_face)
            label = box.box_id
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_left = center_x - text_width / 2
            label_top = center_y - text_height - 12
            draw.rectangle(
                (
                    label_left - 8,
                    label_top - 6,
                    label_left + text_width + 8,
                    label_top + text_height + 6,
                ),
                fill=(0, 0, 0),
            )
            draw.text((label_left, label_top), label, font=font, fill=color)

        annotated.save(output_dir / image_name, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate temporary padded 3D bounding boxes for each paper group and annotate them on images."
    )
    parser.add_argument("--opf", type=Path, default=DEFAULT_OPF_PATH)
    parser.add_argument("--grouped-csv", type=Path, default=DEFAULT_GROUPED_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_BBOX_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--padding-xy", type=float, default=1)
    parser.add_argument("--padding-z", type=float, default=0.70)
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve(load(str(args.opf)))
    camera_models = build_camera_models(project)
    rows = load_grouped_rows(args.grouped_csv)
    boxes = build_bounding_boxes(
        rows, padding_xy=args.padding_xy, padding_z=args.padding_z
    )

    write_bbox_csv(boxes, args.output_csv)

    if not args.skip_images:
        annotate_images(
            boxes,
            camera_models,
            args.opf.parent / "images",
            args.output_dir,
        )

    print(f"Wrote bounding boxes to {args.output_csv}")
    if not args.skip_images:
        print(f"Wrote box annotations to {args.output_dir}")


if __name__ == "__main__":
    main()
