from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image, ImageDraw, ImageFont
from pyopf.io import load
from pyopf.resolve import resolve
from sklearn.cluster import KMeans


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
    basis_u: np.ndarray
    basis_v: np.ndarray


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPF_PATH = PROJECT_ROOT / "opf" / "project.opf"
DEFAULT_CSV_PATH = PROJECT_ROOT / "opf" / "final.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "opf" / "final_grouped.csv"
DEFAULT_OUTPUT_IMAGES = PROJECT_ROOT / "modified_images_grouped"

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
    basis_u = vh[0]
    basis_v = vh[1]
    normal = vh[2]
    normal /= np.linalg.norm(normal)

    camera_positions = np.asarray(
        [camera.position for camera in project.calibration.calibrated_cameras.cameras],
        dtype=float,
    )
    if np.dot(normal, camera_positions.mean(axis=0) - origin) < 0:
        normal = -normal

    return Plane(origin=origin, normal=normal, basis_u=basis_u, basis_v=basis_v)


def load_observations(csv_path: Path) -> list[Observation]:
    observations: list[Observation] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                raise ValueError(f"Invalid row {row_number} in {csv_path}: {row!r}")

            score = float(row[4].strip()) if len(row) > 4 and row[4].strip() else 1.0
            observations.append(
                Observation(
                    image_name=Path(row[0].strip()).name,
                    marker_id=row[1].strip(),
                    x_px=float(row[2].strip()),
                    y_px=float(row[3].strip()),
                    score=score,
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


def determine_group_count(observations: list[Observation]) -> int:
    per_image_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for observation in observations:
        per_image_counts[observation.image_name][observation.marker_id] += 1

    group_count = max(max(counter.values()) for counter in per_image_counts.values())
    if group_count < 1:
        raise ValueError("Could not determine the number of duplicated paper groups")

    return group_count


def cluster_image_groups(
    observations: list[Observation],
    camera_models: dict[str, CameraModel],
    plane: Plane,
    group_count: int,
):
    per_image: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.image_name not in camera_models:
            raise KeyError(f"Missing OPF camera for image {observation.image_name}")
        per_image[observation.image_name].append(observation)

    grouped_rows: list[dict[str, object]] = []
    group_centers_world: list[np.ndarray] = []

    for image_name in sorted(per_image):
        image_observations = per_image[image_name]
        cluster_count = min(group_count, len(image_observations))
        sample = np.asarray([[obs.x_px, obs.y_px] for obs in image_observations], dtype=float)

        labels = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0).fit_predict(sample)
        camera = camera_models[image_name]

        cluster_world_points: dict[int, list[np.ndarray]] = defaultdict(list)
        observation_world_points: list[np.ndarray] = []
        for observation in image_observations:
            observation_world_points.append(
                project_to_plane(camera, plane, observation.x_px, observation.y_px)
            )

        for label, world_point in zip(labels, observation_world_points, strict=False):
            cluster_world_points[int(label)].append(world_point)

        cluster_world_centers = {
            label: np.mean(points, axis=0) for label, points in cluster_world_points.items()
        }

        for label, center in cluster_world_centers.items():
            group_centers_world.append(center)

        for observation, label, world_point in zip(
            image_observations, labels, observation_world_points, strict=False
        ):
            grouped_rows.append(
                {
                    "image_name": observation.image_name,
                    "marker_id": observation.marker_id,
                    "x_px": observation.x_px,
                    "y_px": observation.y_px,
                    "score": observation.score,
                    "local_group_index": int(label),
                    "world_point": world_point,
                    "group_world_center": cluster_world_centers[int(label)],
                }
            )

    return grouped_rows, np.asarray(group_centers_world, dtype=float)


def plane_coordinates(plane: Plane, point: np.ndarray) -> np.ndarray:
    delta = point - plane.origin
    return np.array([np.dot(delta, plane.basis_u), np.dot(delta, plane.basis_v)], dtype=float)


def assign_global_group_ids(
    grouped_rows: list[dict[str, object]],
    group_centers_world: np.ndarray,
    plane: Plane,
    group_count: int,
) -> dict[int, str]:
    if len(group_centers_world) < group_count:
        raise ValueError("Not enough grouped paper observations to assign global IDs")

    group_centers_2d = np.asarray(
        [plane_coordinates(plane, center) for center in group_centers_world],
        dtype=float,
    )
    kmeans = KMeans(n_clusters=group_count, n_init="auto", random_state=0)
    kmeans.fit(group_centers_2d)

    cluster_to_points: dict[int, list[np.ndarray]] = defaultdict(list)
    for center, label in zip(group_centers_2d, kmeans.labels_, strict=False):
        cluster_to_points[int(label)].append(center)

    ordered_cluster_ids = sorted(
        cluster_to_points,
        key=lambda cluster_id: (
            round(float(np.mean([point[1] for point in cluster_to_points[cluster_id]])), 6),
            round(float(np.mean([point[0] for point in cluster_to_points[cluster_id]])), 6),
        ),
    )

    return {
        cluster_id: f"paper_{index:02d}"
        for index, cluster_id in enumerate(ordered_cluster_ids, start=1)
    }


def label_rows_with_global_groups(
    grouped_rows: list[dict[str, object]],
    plane: Plane,
    group_count: int,
) -> list[dict[str, object]]:
    group_centers_world = np.asarray(
        [row["group_world_center"] for row in grouped_rows],
        dtype=float,
    )
    group_centers_2d = np.asarray(
        [plane_coordinates(plane, center) for center in group_centers_world],
        dtype=float,
    )
    kmeans = KMeans(n_clusters=group_count, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(group_centers_2d)
    global_group_map = assign_global_group_ids(grouped_rows, group_centers_world, plane, group_count)

    labeled_rows: list[dict[str, object]] = []
    for row, cluster_label in zip(grouped_rows, labels, strict=False):
        global_group_id = global_group_map[int(cluster_label)]
        labeled_row = dict(row)
        labeled_row["group_id"] = global_group_id
        labeled_row["grouped_marker_id"] = f"{global_group_id}_{row['marker_id']}"
        labeled_rows.append(labeled_row)

    return labeled_rows


def write_output_csv(rows: list[dict[str, object]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "group_id",
        "grouped_marker_id",
        "marker_id",
        "x_px",
        "y_px",
        "score",
        "world_x",
        "world_y",
        "world_z",
        "group_center_x",
        "group_center_y",
        "group_center_z",
    ]

    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["image_name"]),
                str(item["group_id"]),
                str(item["marker_id"]),
                float(item["x_px"]),
                float(item["y_px"]),
            ),
        ):
            world_point = np.asarray(row["world_point"], dtype=float)
            group_center = np.asarray(row["group_world_center"], dtype=float)
            writer.writerow(
                {
                    "image_name": row["image_name"],
                    "group_id": row["group_id"],
                    "grouped_marker_id": row["grouped_marker_id"],
                    "marker_id": row["marker_id"],
                    "x_px": f"{float(row['x_px']):.2f}",
                    "y_px": f"{float(row['y_px']):.2f}",
                    "score": f"{float(row['score']):.3f}",
                    "world_x": f"{world_point[0]:.6f}",
                    "world_y": f"{world_point[1]:.6f}",
                    "world_z": f"{world_point[2]:.6f}",
                    "group_center_x": f"{group_center[0]:.6f}",
                    "group_center_y": f"{group_center[1]:.6f}",
                    "group_center_z": f"{group_center[2]:.6f}",
                }
            )


def annotate_images(rows: list[dict[str, object]], images_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_image[str(row["image_name"])].append(row)

    font = ImageFont.load_default(size=24)

    for image_name, image_rows in per_image.items():
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")

        draw = ImageDraw.Draw(annotated)
        for row in image_rows:
            group_id = str(row["group_id"])
            group_index = int(group_id.split("_")[1]) - 1
            color = GROUP_COLORS[group_index % len(GROUP_COLORS)]

            x = int(round(float(row["x_px"])))
            y = int(round(float(row["y_px"])))
            label = str(row["grouped_marker_id"])

            draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline=color, width=4)
            draw.line((x - 16, y, x + 16, y), fill=color, width=3)
            draw.line((x, y - 16, x, y + 16), fill=color, width=3)

            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_left = x + 12
            label_top = y - text_height - 12
            draw.rectangle(
                (label_left - 8, label_top - 6, label_left + text_width + 8, label_top + text_height + 6),
                fill=(0, 0, 0),
            )
            draw.text((label_left, label_top), label, font=font, fill=color)

        annotated.save(output_dir / image_name, quality=95)


def log_to_rerun(project, camera_models: dict[str, CameraModel], rows: list[dict[str, object]]) -> None:
    rr.init("Duplicate Marker Groups", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points = project.point_cloud_objs[0].nodes[0]
    rr.log("world/points", rr.Points3D(points.position, colors=points.color))

    sensor_map = {sensor.id: sensor for sensor in project.input_cameras.sensors}
    calib_sensor_map = {
        sensor.id: sensor for sensor in project.calibration.calibrated_cameras.sensors
    }

    per_image_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_image_rows[str(row["image_name"])].append(row)

    positions = []
    paper_centers: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        paper_centers[str(row["group_id"])].append(np.asarray(row["group_world_center"], dtype=float))

    ordered_paper_ids = sorted(paper_centers)
    rr.log(
        "world/paper_groups",
        rr.Points3D(
            [np.mean(paper_centers[group_id], axis=0) for group_id in ordered_paper_ids],
            labels=ordered_paper_ids,
        ),
        static=True,
    )

    for i, (raw_camera, calib_camera) in enumerate(
        zip(
            project.camera_list.cameras,
            project.calibration.calibrated_cameras.cameras,
            strict=False,
        )
    ):
        image_name = Path(raw_camera.uri).name
        if image_name not in per_image_rows:
            continue

        positions.append(np.asarray(calib_camera.position, dtype=float))
        rr.set_time("image", sequence=i)
        entity = "world/camera"
        sensor = sensor_map[calib_camera.sensor_id]
        calib_sensor = calib_sensor_map[calib_camera.sensor_id]

        rr.log(
            entity,
            rr.Transform3D(
                translation=calib_camera.position,
                mat3x3=rotation_from_opk(np.asarray(calib_camera.orientation_deg, dtype=float)),
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

        image_path = DEFAULT_OPF_PATH.parent / "images" / image_name
        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")

        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default(size=24)
        for row in per_image_rows[image_name]:
            group_id = str(row["group_id"])
            group_index = int(group_id.split("_")[1]) - 1
            color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
            x = int(round(float(row["x_px"])))
            y = int(round(float(row["y_px"])))
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), outline=color, width=4)
            draw.text((x + 12, y - 24), str(row["grouped_marker_id"]), font=font, fill=color)

        rr.log(entity + "/image/rgb", rr.Image(annotated))

    if positions:
        rr.log("world/camera_path", rr.LineStrips3D([np.asarray(positions)]), static=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign stable paper IDs to duplicate marker groups using OPF camera geometry "
            "and export grouped marker observations."
        )
    )
    parser.add_argument("--opf", type=Path, default=DEFAULT_OPF_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-images", type=Path, default=DEFAULT_OUTPUT_IMAGES)
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip writing annotated images",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Open a Rerun viewer with grouped paper labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve(load(str(args.opf)))
    camera_models = build_camera_models(project)
    plane = fit_plane(project)
    observations = load_observations(args.csv)
    group_count = determine_group_count(observations)

    grouped_rows, _ = cluster_image_groups(observations, camera_models, plane, group_count)
    labeled_rows = label_rows_with_global_groups(grouped_rows, plane, group_count)

    write_output_csv(labeled_rows, args.output_csv)

    if not args.skip_images:
        annotate_images(labeled_rows, args.opf.parent / "images", args.output_images)

    if args.rerun:
        log_to_rerun(project, camera_models, labeled_rows)

    print(f"Wrote grouped CSV to {args.output_csv}")
    if not args.skip_images:
        print(f"Wrote annotated images to {args.output_images}")
    print(f"Assigned {group_count} unique paper groups")


if __name__ == "__main__":
    main()
