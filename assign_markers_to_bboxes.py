from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    image_name: str
    position: np.ndarray
    rotation_world_from_camera: np.ndarray
    focal_length_px: float
    principal_point_px: np.ndarray


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


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_opf_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.opf_json_dir is not None:
        opf_json_dir = args.opf_json_dir
    elif args.opf_root is not None:
        opf_json_dir = args.opf_root / "opf_files"
    else:
        raise ValueError("Provide either --opf-root or --opf-json-dir")

    return {
        "camera_list": args.camera_list_json or opf_json_dir / "camera_list.json",
        "calibrated_cameras": args.calibrated_cameras_json or opf_json_dir / "calibrated_cameras.json",
        "input_cameras": args.input_cameras_json or opf_json_dir / "input_cameras.json",
        "control_points": args.control_points_json or opf_json_dir / "calibrated_control_points.json",
    }


def build_camera_models(
    camera_list_json: Path,
    calibrated_cameras_json: Path,
    input_cameras_json: Path,
) -> dict[str, CameraModel]:
    camera_list = load_json(camera_list_json)
    calibrated_cameras = load_json(calibrated_cameras_json)
    input_cameras = load_json(input_cameras_json)

    image_name_by_camera_id = {
        int(camera["id"]): Path(camera["uri"]).name for camera in camera_list["cameras"]
    }
    sensor_ids = {int(sensor["id"]) for sensor in input_cameras["sensors"]}
    intrinsics_by_sensor_id = {
        int(sensor["id"]): sensor["internals"] for sensor in calibrated_cameras["sensors"]
    }

    camera_models: dict[str, CameraModel] = {}
    for camera in calibrated_cameras["cameras"]:
        camera_id = int(camera["id"])
        sensor_id = int(camera["sensor_id"])
        if camera_id not in image_name_by_camera_id:
            raise KeyError(f"Camera {camera_id} missing from camera_list.json")
        if sensor_id not in sensor_ids:
            raise KeyError(f"Sensor {sensor_id} missing from input_cameras.json")
        if sensor_id not in intrinsics_by_sensor_id:
            raise KeyError(f"Sensor {sensor_id} missing from calibrated_cameras.json")

        intrinsics = intrinsics_by_sensor_id[sensor_id]
        camera_models[image_name_by_camera_id[camera_id]] = CameraModel(
            image_name=image_name_by_camera_id[camera_id],
            position=np.asarray(camera["position"], dtype=float),
            rotation_world_from_camera=rotation_from_opk(
                np.asarray(camera["orientation_deg"], dtype=float)
            ),
            focal_length_px=float(intrinsics["focal_length_px"]),
            principal_point_px=np.asarray(intrinsics["principal_point_px"], dtype=float),
        )

    return camera_models


def fit_plane(control_points_json: Path) -> Plane:
    calibrated_control_points = load_json(control_points_json)
    points = np.asarray(
        [point["coordinates"] for point in calibrated_control_points["points"]],
        dtype=float,
    )
    if len(points) < 3:
        raise ValueError("Need at least 3 calibrated control points")

    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin)
    normal = vh[2]
    normal /= np.linalg.norm(normal)
    return Plane(origin=origin, normal=normal)


def orient_plane_toward_cameras(plane: Plane, camera_models: dict[str, CameraModel]) -> Plane:
    camera_positions = np.asarray([camera.position for camera in camera_models.values()], dtype=float)
    normal = plane.normal
    if np.dot(normal, camera_positions.mean(axis=0) - plane.origin) < 0:
        normal = -normal
    return Plane(origin=plane.origin, normal=normal)


def load_marker_rows(marker_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with marker_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                raise ValueError(f"Invalid marker row {row_number} in {marker_csv}: {row!r}")
            rows.append(
                {
                    "image_name": Path(row[0].strip()).name,
                    "marker_id": row[1].strip(),
                    "x_px": row[2].strip(),
                    "y_px": row[3].strip(),
                    "score": row[4].strip() if len(row) > 4 else "",
                }
            )
    return rows


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
        raise ValueError(f"Ray is parallel to the fitted plane for image {camera.image_name}")

    distance = float(np.dot(plane.normal, plane.origin - camera.position) / denominator)
    if distance <= 0:
        raise ValueError(f"Plane intersection is behind the camera for image {camera.image_name}")

    return camera.position + distance * world_ray


def load_bboxes(
    bbox_csv: Path,
    default_bottom_z: float,
    default_top_z: float,
) -> list[BoundingBox]:
    with bbox_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "bottomX", "bottomY", "topX", "topY"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing bbox columns: {sorted(missing)}")

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


def assign_rows(
    marker_rows: list[dict[str, str]],
    camera_models: dict[str, CameraModel],
    plane: Plane,
    boxes: list[BoundingBox],
) -> list[dict[str, str]]:
    assigned_rows: list[dict[str, str]] = []
    for row in marker_rows:
        image_name = row["image_name"]
        if image_name not in camera_models:
            raise KeyError(f"Image {image_name} not present in OPF camera metadata")

        world_point = project_to_plane(
            camera_models[image_name],
            plane,
            float(row["x_px"]),
            float(row["y_px"]),
        )
        matches = [
            box
            for box in boxes
            if box.bottom_x <= world_point[0] <= box.top_x
            and box.bottom_y <= world_point[1] <= box.top_y
        ]
        if len(matches) > 1:
            matches.sort(key=lambda box: (box.top_x - box.bottom_x) * (box.top_y - box.bottom_y))

        box = matches[0] if matches else None
        bbox_id = box.bbox_id if box is not None else "UNASSIGNED"
        prefixed_marker_id = (
            f"{bbox_id}_{row['marker_id']}" if box is not None else row["marker_id"]
        )

        assigned_rows.append(
            {
                "image_name": image_name,
                "bbox_id": bbox_id,
                "prefixed_marker_id": prefixed_marker_id,
                "marker_id": row["marker_id"],
                "x_px": f"{float(row['x_px']):.2f}",
                "y_px": f"{float(row['y_px']):.2f}",
                "score": row["score"] if row["score"] else "1.0",
                "world_x": f"{world_point[0]:.6f}",
                "world_y": f"{world_point[1]:.6f}",
                "world_z": f"{world_point[2]:.6f}",
            }
        )
    return assigned_rows


def write_output_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item["image_name"],
                item["bbox_id"],
                item["marker_id"],
                float(item["x_px"]),
                float(item["y_px"]),
            ),
        ):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign markers from a raw CSV into bounding boxes using OPF sidecar JSON metadata "
            "and write a bbox-prefixed output CSV."
        )
    )
    parser.add_argument("--marker-csv", type=Path, required=True)
    parser.add_argument("--bbox-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--opf-root",
        type=Path,
        help="Path to the OPF export root containing opf_files/",
    )
    parser.add_argument(
        "--opf-json-dir",
        type=Path,
        help="Path to the opf_files/ directory. Use this instead of --opf-root if preferred.",
    )
    parser.add_argument("--camera-list-json", type=Path)
    parser.add_argument("--calibrated-cameras-json", type=Path)
    parser.add_argument("--input-cameras-json", type=Path)
    parser.add_argument("--control-points-json", type=Path)
    parser.add_argument(
        "--bbox-padding-z",
        type=float,
        default=0.05,
        help="Used only when bottomZ/topZ are missing from the bbox CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_opf_paths(args)
    camera_models = build_camera_models(
        camera_list_json=paths["camera_list"],
        calibrated_cameras_json=paths["calibrated_cameras"],
        input_cameras_json=paths["input_cameras"],
    )
    plane = orient_plane_toward_cameras(fit_plane(paths["control_points"]), camera_models)

    marker_rows = load_marker_rows(args.marker_csv)
    world_points = [
        project_to_plane(camera_models[row["image_name"]], plane, float(row["x_px"]), float(row["y_px"]))
        for row in marker_rows
    ]
    all_z = [point[2] for point in world_points]
    boxes = load_bboxes(
        args.bbox_csv,
        default_bottom_z=min(all_z) - args.bbox_padding_z,
        default_top_z=max(all_z) + args.bbox_padding_z,
    )
    assigned_rows = assign_rows(marker_rows, camera_models, plane, boxes)
    write_output_csv(assigned_rows, args.output_csv)

    assigned_count = sum(1 for row in assigned_rows if row["bbox_id"] != "UNASSIGNED")
    print(f"Wrote {len(assigned_rows)} rows to {args.output_csv}")
    print(f"Assigned {assigned_count} markers to bbox IDs")


if __name__ == "__main__":
    main()
