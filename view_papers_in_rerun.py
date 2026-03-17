from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
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
    tilt_deg: float


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


def paper_corners(frame: PlaneFrame, marker_points: dict[str, np.ndarray]) -> np.ndarray:
    points = np.asarray(list(marker_points.values()), dtype=float)
    rel = points - frame.center
    xs = rel @ frame.x_axis
    ys = rel @ frame.y_axis
    x_extent = max(abs(xs.min()), abs(xs.max()))
    y_extent = max(abs(ys.min()), abs(ys.max()))

    local_corners = np.array(
        [
            [-x_extent, -y_extent],
            [x_extent, -y_extent],
            [x_extent, y_extent],
            [-x_extent, y_extent],
        ],
        dtype=float,
    )
    return np.asarray(
        [
            frame.center + corner[0] * frame.x_axis + corner[1] * frame.y_axis
            for corner in local_corners
        ],
        dtype=float,
    )


def tilt_deg(frame: PlaneFrame, ground_plane: PlaneFrame) -> float:
    cos_angle = float(np.clip(np.dot(frame.normal, ground_plane.normal), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def build_paper_visualizations(
    rows: list[dict[str, str]],
    ground_plane: PlaneFrame,
) -> list[PaperVisualization]:
    visualizations: list[PaperVisualization] = []
    for bbox_id, marker_map in sorted(averaged_marker_points(rows).items()):
        frame = fit_paper_frame(marker_map, ground_plane)
        visualizations.append(
            PaperVisualization(
                bbox_id=bbox_id,
                center=frame.center,
                corners=paper_corners(frame, marker_map),
                normal=frame.normal,
                z_axis=frame.normal,
                tilt_deg=tilt_deg(frame, ground_plane),
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
        labels.append(f"{paper.bbox_id} | tilt={paper.tilt_deg:.2f} deg")
        colors.append(color)

    averaged = averaged_marker_points(rows)
    for index, paper in enumerate(papers):
        color = PAPER_COLORS[index % len(PAPER_COLORS)]
        for marker_id, point in sorted(averaged[paper.bbox_id].items()):
            marker_points.append(point)
            marker_labels.append(f"{paper.bbox_id}_{marker_id}")
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
    papers = build_paper_visualizations(rows, ground_plane)
    log_scene(project, rows, papers, ground_plane, args.axis_length)
    print(f"Loaded {len(papers)} papers from {args.csv}")
    print("Logged point cloud, paper outlines, local Z axes, and tilt labels to Rerun")


if __name__ == "__main__":
    args = parse_args()
    main(args)
