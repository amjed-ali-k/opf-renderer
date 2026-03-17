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
class PlaneFrame:
    center: np.ndarray
    normal: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray


@dataclass(frozen=True)
class PaperVisualization:
    bbox_id: str
    center: np.ndarray
    corners: np.ndarray
    normal: np.ndarray
    z_axis: np.ndarray
    ground_ref_x: np.ndarray
    ground_ref_y: np.ndarray
    tilt_deg: float
    tilt_x_deg: float
    tilt_y_deg: float
    corner_height_mm: dict[str, float]


PAPER_COLORS = [
    [239, 71, 111],
    [255, 209, 102],
    [6, 214, 160],
    [17, 138, 178],
    [7, 59, 76],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display bbox-prefixed marker groups as paper planes in Rerun, together with "
            "their center, local Z axis, and tilt relative to the ground plane."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Prefixed marker CSV")
    parser.add_argument(
        "--opf",
        type=Path,
        required=True,
        help="Path to project.opf or the OPF export root containing project.opf",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.35,
        help="Displayed length of the local Z axis in world units",
    )
    parser.add_argument(
        "--ground-axis-length",
        type=float,
        default=0.25,
        help="Displayed length of the local ground-plane reference axes in world units",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path("modified_images_paper_view"),
        help="Directory for annotated JPEG outputs",
    )
    return parser.parse_args()


def resolve_project_path(opf_path: Path) -> Path:
    if opf_path.is_dir():
        project_path = opf_path / "project.opf"
    else:
        project_path = opf_path
    if not project_path.exists():
        raise FileNotFoundError(f"OPF project file not found: {project_path}")
    return project_path


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def fit_ground_plane(project) -> PlaneFrame:
    points = np.asarray(
        [point.coordinates for point in project.calibration.calibrated_control_points.points],
        dtype=float,
    )
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center)
    x_axis = normalize(vh[0])
    y_axis = normalize(vh[1])
    normal = normalize(vh[2])

    camera_positions = np.asarray(
        [camera.position for camera in project.calibration.calibrated_cameras.cameras],
        dtype=float,
    )
    if np.dot(normal, camera_positions.mean(axis=0) - center) < 0:
        normal = -normal

    y_axis = normalize(np.cross(normal, x_axis))
    x_axis = normalize(np.cross(y_axis, normal))
    return PlaneFrame(center=center, normal=normal, x_axis=x_axis, y_axis=y_axis)


def averaged_marker_points(rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bbox_id = row["bbox_id"]
        if bbox_id == "UNASSIGNED":
            continue
        grouped[bbox_id][row["marker_id"]].append(
            np.array(
                [
                    float(row["world_x"]),
                    float(row["world_y"]),
                    float(row["world_z"]),
                ],
                dtype=float,
            )
        )

    averaged: dict[str, dict[str, np.ndarray]] = {}
    for bbox_id, marker_map in grouped.items():
        averaged[bbox_id] = {
            marker_id: np.mean(points, axis=0) for marker_id, points in marker_map.items()
        }
    return averaged


def fit_paper_frame(marker_points: dict[str, np.ndarray], ground_plane: PlaneFrame) -> PlaneFrame:
    points = np.asarray(list(marker_points.values()), dtype=float)
    if len(points) < 3:
        raise ValueError("Each paper needs at least 3 marker points")

    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center)
    x_axis = normalize(vh[0])
    normal = normalize(vh[2])

    if np.dot(normal, ground_plane.normal) < 0:
        normal = -normal

    # Use a stable in-plane basis from the four marker corners.
    y_axis = normalize(np.cross(normal, x_axis))
    x_axis = normalize(np.cross(y_axis, normal))
    return PlaneFrame(center=center, normal=normal, x_axis=x_axis, y_axis=y_axis)


def ordered_marker_corners(
    frame: PlaneFrame,
    marker_points: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Return the actual four marker points ordered around the paper center.

    The markers are treated as the paper corners directly. We sort them by angle in the
    fitted paper plane so overlays connect the real detected corners instead of a synthetic
    rectangle expanded from axis extents.
    """

    points = np.asarray(list(marker_points.values()), dtype=float)
    marker_ids = list(marker_points.keys())
    rel = points - frame.center
    xs = rel @ frame.x_axis
    ys = rel @ frame.y_axis
    angles = np.arctan2(ys, xs)
    order = np.argsort(angles)
    return points[order], [marker_ids[index] for index in order]


def tilt_deg(frame: PlaneFrame, ground_plane: PlaneFrame) -> float:
    cos_angle = float(np.clip(np.dot(frame.normal, ground_plane.normal), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def tilt_components_deg(frame: PlaneFrame, ground_plane: PlaneFrame) -> tuple[float, float]:
    """Return signed tilt components along the ground-plane X and Y axes.

    The paper normal is expressed in the ground-plane basis:
    - `tilt_x_deg` captures how much the paper tilts in the ground X direction
    - `tilt_y_deg` captures how much the paper tilts in the ground Y direction

    Together with the total tilt magnitude, these let you see whether the paper leans
    more strongly in one direction than the other.
    """

    nx = float(np.dot(frame.normal, ground_plane.x_axis))
    ny = float(np.dot(frame.normal, ground_plane.y_axis))
    nz = float(np.dot(frame.normal, ground_plane.normal))
    tilt_x_deg = float(np.degrees(np.arctan2(nx, nz)))
    tilt_y_deg = float(np.degrees(np.arctan2(ny, nz)))
    return tilt_x_deg, tilt_y_deg


def corner_height_mm(
    marker_points: dict[str, np.ndarray],
    ground_plane: PlaneFrame,
) -> dict[str, float]:
    """Signed height of each corner above the ground plane, in millimeters."""

    heights: dict[str, float] = {}
    for marker_id, point in marker_points.items():
        heights[marker_id] = float(np.dot(point - ground_plane.center, ground_plane.normal) * 1000.0)
    return heights


def build_paper_visualizations(
    rows: list[dict[str, str]],
    ground_plane: PlaneFrame,
    ground_axis_length: float,
) -> list[PaperVisualization]:
    visualizations: list[PaperVisualization] = []
    for bbox_id, marker_map in sorted(averaged_marker_points(rows).items()):
        frame = fit_paper_frame(marker_map, ground_plane)
        tilt_x_deg, tilt_y_deg = tilt_components_deg(frame, ground_plane)
        corners, _ = ordered_marker_corners(frame, marker_map)
        visualizations.append(
            PaperVisualization(
                bbox_id=bbox_id,
                center=frame.center,
                corners=corners,
                normal=frame.normal,
                z_axis=frame.normal,
                ground_ref_x=ground_plane.x_axis * ground_axis_length,
                ground_ref_y=ground_plane.y_axis * ground_axis_length,
                tilt_deg=tilt_deg(frame, ground_plane),
                tilt_x_deg=tilt_x_deg,
                tilt_y_deg=tilt_y_deg,
                corner_height_mm=corner_height_mm(marker_map, ground_plane),
            )
        )
    return visualizations


def rerun_init() -> None:
    try:
        rr.init("Paper Viewer", spawn=True)
    except RuntimeError as exc:
        if "Failed to find Rerun Viewer executable" not in str(exc):
            raise
        rr.init("Paper Viewer", spawn=False)


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


def build_camera_maps(project) -> tuple[dict[str, object], dict[int, object]]:
    camera_by_image_name: dict[str, object] = {}
    calibrated_by_id: dict[int, object] = {}
    for raw_camera, calibrated_camera in zip(
        project.camera_list.cameras,
        project.calibration.calibrated_cameras.cameras,
        strict=False,
    ):
        camera_by_image_name[Path(raw_camera.uri).name] = calibrated_camera
        calibrated_by_id[int(calibrated_camera.id)] = calibrated_camera
    return camera_by_image_name, calibrated_by_id


def project_world_to_image(calibrated_camera, sensor, point_world: np.ndarray) -> np.ndarray | None:
    rotation = rotation_from_opk(np.asarray(calibrated_camera.orientation_deg, dtype=float))
    point_camera = rotation.T @ (point_world - np.asarray(calibrated_camera.position, dtype=float))
    if point_camera[2] >= -1e-6:
        return None

    focal = float(sensor.internals.focal_length_px)
    principal = np.asarray(sensor.internals.principal_point_px, dtype=float)
    x_px = focal * (point_camera[0] / -point_camera[2]) + principal[0]
    y_px = principal[1] - focal * (point_camera[1] / -point_camera[2])
    return np.array([x_px, y_px], dtype=float)


def annotate_images(
    project,
    rows: list[dict[str, str]],
    papers: list[PaperVisualization],
    images_dir: Path,
    output_dir: Path,
    axis_length: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_map = {
        int(sensor.id): sensor for sensor in project.calibration.calibrated_cameras.sensors
    }
    camera_by_image_name, _ = build_camera_maps(project)
    averaged = averaged_marker_points(rows)
    font = ImageFont.load_default(size=22)

    rr.set_time("image", sequence=0)
    for image_index, raw_camera in enumerate(project.camera_list.cameras):
        image_name = Path(raw_camera.uri).name
        if image_name not in camera_by_image_name:
            continue

        calibrated_camera = camera_by_image_name[image_name]
        sensor = sensor_map[int(calibrated_camera.sensor_id)]
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")

        draw = ImageDraw.Draw(annotated)
        rr.set_time("image", sequence=image_index)

        for paper_index, paper in enumerate(papers):
            color = tuple(PAPER_COLORS[paper_index % len(PAPER_COLORS)])

            corners_2d = [
                project_world_to_image(calibrated_camera, sensor, point)
                for point in paper.corners
            ]
            if any(point is None for point in corners_2d):
                continue
            corners = [tuple(point.tolist()) for point in corners_2d if point is not None]

            draw.line((corners[0], corners[1], corners[2], corners[3], corners[0]), fill=color, width=3)

            center_2d = project_world_to_image(calibrated_camera, sensor, paper.center)
            z_tip_2d = project_world_to_image(calibrated_camera, sensor, paper.center + paper.z_axis * axis_length)
            gx_tip_2d = project_world_to_image(calibrated_camera, sensor, paper.center + paper.ground_ref_x)
            gy_tip_2d = project_world_to_image(calibrated_camera, sensor, paper.center + paper.ground_ref_y)
            if center_2d is None:
                continue

            cx, cy = center_2d.tolist()
            if z_tip_2d is not None:
                draw.line(((cx, cy), tuple(z_tip_2d.tolist())), fill=(255, 255, 255), width=4)
                draw.text((z_tip_2d[0] + 4, z_tip_2d[1] + 4), "Z", font=font, fill=(255, 255, 255))
            if gx_tip_2d is not None:
                draw.line(((cx, cy), tuple(gx_tip_2d.tolist())), fill=(0, 255, 255), width=3)
                draw.text((gx_tip_2d[0] + 4, gx_tip_2d[1] + 4), "GX", font=font, fill=(0, 255, 255))
            if gy_tip_2d is not None:
                draw.line(((cx, cy), tuple(gy_tip_2d.tolist())), fill=(255, 255, 0), width=3)
                draw.text((gy_tip_2d[0] + 4, gy_tip_2d[1] + 4), "GY", font=font, fill=(255, 255, 0))

            label = (
                f"{paper.bbox_id} "
                f"tilt={paper.tilt_deg:.2f}deg "
                f"gx={paper.tilt_x_deg:.2f} "
                f"gy={paper.tilt_y_deg:.2f}"
            )
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            draw.rectangle(
                (cx - text_width / 2 - 8, cy - text_height - 24, cx + text_width / 2 + 8, cy - 4),
                fill=(0, 0, 0),
            )
            draw.text((cx - text_width / 2, cy - text_height - 18), label, font=font, fill=color)

            for marker_id, marker_point in sorted(averaged[paper.bbox_id].items()):
                marker_2d = project_world_to_image(calibrated_camera, sensor, marker_point)
                if marker_2d is None:
                    continue
                mx, my = marker_2d.tolist()
                draw.ellipse((mx - 8, my - 8, mx + 8, my + 8), outline=color, width=3)
                draw.text(
                    (mx + 10, my - 12),
                    f"{paper.bbox_id}_{marker_id} {paper.corner_height_mm[marker_id]:.1f}mm",
                    font=font,
                    fill=color,
                )

        output_path = output_dir / image_name
        annotated.save(output_path, quality=50)
        rr.log(f"world/images/{image_name}", rr.Image(np.asarray(annotated)))


def log_scene(
    project,
    rows: list[dict[str, str]],
    papers: list[PaperVisualization],
    ground_plane: PlaneFrame,
    axis_length: float,
) -> None:
    rerun_init()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points = project.point_cloud_objs[0].nodes[0]
    rr.log("world/points", rr.Points3D(points.position, colors=points.color))

    ground_extent = 1.5
    ground_cross = [
        np.array(
            [
                ground_plane.center - ground_plane.x_axis * ground_extent,
                ground_plane.center + ground_plane.x_axis * ground_extent,
            ]
        ),
        np.array(
            [
                ground_plane.center - ground_plane.y_axis * ground_extent,
                ground_plane.center + ground_plane.y_axis * ground_extent,
            ]
        ),
    ]
    rr.log("world/ground_plane_axes", rr.LineStrips3D(ground_cross), static=True)

    outlines = []
    z_axes = []
    centers = []
    labels = []
    colors = []
    marker_points = []
    marker_labels = []
    marker_colors = []

    for index, paper in enumerate(papers):
        color = PAPER_COLORS[index % len(PAPER_COLORS)]
        outline = np.vstack([paper.corners, paper.corners[0]])
        outlines.append(outline)
        z_axes.append(np.array([paper.center, paper.center + paper.z_axis * axis_length]))
        centers.append(paper.center)
        labels.append(
            f"{paper.bbox_id} | tilt={paper.tilt_deg:.2f} deg | "
            f"gx={paper.tilt_x_deg:.2f} deg | gy={paper.tilt_y_deg:.2f} deg"
        )
        colors.append(color)

    averaged = averaged_marker_points(rows)
    for index, paper in enumerate(papers):
        color = PAPER_COLORS[index % len(PAPER_COLORS)]
        for marker_id, point in sorted(averaged[paper.bbox_id].items()):
            marker_points.append(point)
            marker_labels.append(
                f"{paper.bbox_id}_{marker_id} | h={paper.corner_height_mm[marker_id]:.1f} mm"
            )
            marker_colors.append(color)

    if outlines:
        rr.log("world/papers/outlines", rr.LineStrips3D(outlines, colors=colors), static=True)
        rr.log("world/papers/z_axes", rr.LineStrips3D(z_axes, colors=colors), static=True)
        rr.log(
            "world/papers/centers",
            rr.Points3D(centers, colors=colors, labels=labels),
            static=True,
        )
    if marker_points:
        rr.log(
            "world/papers/markers",
            rr.Points3D(marker_points, colors=marker_colors, labels=marker_labels),
            static=True,
        )


def main(args: argparse.Namespace) -> None:
    project_path = resolve_project_path(args.opf)
    project = resolve(load(str(project_path)))
    rows = load_rows(args.csv)
    ground_plane = fit_ground_plane(project)
    papers = build_paper_visualizations(rows, ground_plane, args.ground_axis_length)
    log_scene(project, rows, papers, ground_plane, args.axis_length)
    annotate_images(project, rows, papers, project_path.parent / "images", args.output_images, args.axis_length)
    print(f"Loaded {len(papers)} papers from {args.csv}")
    print("Logged point cloud, paper outlines, local Z axes, tilt labels, and annotated images to Rerun")


if __name__ == "__main__":
    args = parse_args()
    main(args)
