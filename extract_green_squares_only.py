#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

import ezdxf


def _key(p: tuple[float, float, float], tol: float) -> tuple[int, int, int]:
    return (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))


def _vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _dot(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _norm(u: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(u, u))


def _add(a: tuple[float, float, float], u: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + u[0], a[1] + u[1], a[2] + u[2])


def _sub(a: tuple[float, float, float], u: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - u[0], a[1] - u[1], a[2] - u[2])


@dataclass(frozen=True)
class LineSeg:
    layer: str
    handle: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


def _iter_green_layers(doc: ezdxf.EzDxf) -> Iterator[str]:
    # AutoCAD Color Index (ACI): 3 == green.
    for layer in doc.layers:
        if layer.dxf.color == 3:
            yield layer.dxf.name


def _load_lines(doc: ezdxf.EzDxf, layers: Iterable[str]) -> list[LineSeg]:
    msp = doc.modelspace()
    out: list[LineSeg] = []
    for layer in layers:
        for e in msp.query(f'LINE[layer=="{layer}"]'):
            out.append(
                LineSeg(
                    layer=layer,
                    handle=str(e.dxf.handle),
                    start=(e.dxf.start.x, e.dxf.start.y, e.dxf.start.z),
                    end=(e.dxf.end.x, e.dxf.end.y, e.dxf.end.z),
                )
            )
    return out


def _detect_square_line_sets(
    lines: list[LineSeg],
    *,
    tol_point: float,
    tol_angle_cos: float,
    tol_len_rel: float,
) -> set[tuple[int, int, int, int]]:
    pt_rep: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    inc: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    edge_to_idx: dict[frozenset[tuple[int, int, int]], int] = {}

    for idx, e in enumerate(lines):
        ka = _key(e.start, tol_point)
        kb = _key(e.end, tol_point)
        pt_rep.setdefault(ka, e.start)
        pt_rep.setdefault(kb, e.end)
        inc[ka].append(idx)
        inc[kb].append(idx)
        edge_to_idx[frozenset((ka, kb))] = idx

    squares: set[tuple[int, int, int, int]] = set()

    for i, e in enumerate(lines):
        ka = _key(e.start, tol_point)
        kb = _key(e.end, tol_point)
        A0 = pt_rep[ka]
        B0 = pt_rep[kb]
        v0 = _vec(A0, B0)
        lv0 = _norm(v0)
        if lv0 == 0.0:
            continue

        # Try both endpoints as the corner (A) to avoid orientation assumptions.
        for kA, kB, A, B, v, lv in (
            (ka, kb, A0, B0, v0, lv0),
            (kb, ka, B0, A0, (-v0[0], -v0[1], -v0[2]), lv0),
        ):
            for j in inc[kA]:
                if j == i:
                    continue
                ej = lines[j]
                kj1 = _key(ej.start, tol_point)
                kj2 = _key(ej.end, tol_point)
                if kj1 == kA:
                    C = pt_rep[kj2]
                    kC = kj2
                elif kj2 == kA:
                    C = pt_rep[kj1]
                    kC = kj1
                else:
                    continue

                w = _vec(A, C)
                lw = _norm(w)
                if lw == 0.0:
                    continue

                # Right-angle check: cos(theta) ~= 0.
                cos_abs = abs(_dot(v, w) / (lv * lw))
                if cos_abs > tol_angle_cos:
                    continue

                # Square check: side lengths approximately equal.
                if abs(lv - lw) / max(lv, lw) > tol_len_rel:
                    continue

                # The 4th corner should be at B +/- w (depends on w direction).
                for D in (_add(B, w), _sub(B, w)):
                    kD = _key(D, tol_point)
                    if kD not in pt_rep:
                        continue

                    if frozenset((kC, kD)) not in edge_to_idx:
                        continue
                    if frozenset((kB, kD)) not in edge_to_idx:
                        continue

                    idxs = sorted(
                        [
                            i,
                            j,
                            edge_to_idx[frozenset((kC, kD))],
                            edge_to_idx[frozenset((kB, kD))],
                        ]
                    )
                    squares.add(tuple(idxs))  # type: ignore[arg-type]

    return squares


def _ordered_square_vertices(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    # Sort by polar angle around centroid; good enough for convex quads.
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.hypot(dx, dy)


def _square_width_from_vertices(ordered: list[tuple[float, float, float]]) -> float | None:
    if len(ordered) != 4:
        return None
    d = [_dist_xy(ordered[i], ordered[(i + 1) % 4]) for i in range(4)]
    if any(x <= 0.0 for x in d):
        return None
    return sum(d) / 4.0


def _bbox_xy(points: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract green (ACI=3) square boxes from a DXF into a new DXF.")
    ap.add_argument("--input", default="sample.dxf", help="Input DXF path.")
    ap.add_argument("--output", default="green_squares_only.dxf", help="Output DXF path.")
    ap.add_argument(
        "--csv-output",
        default=None,
        help="Optional bbox CSV output path. Default: <output>.csv",
    )
    ap.add_argument(
        "--details-csv-output",
        default=None,
        help="Optional detailed CSV (width/area + source entityId). Default: not written.",
    )
    ap.add_argument(
        "--min-width",
        type=float,
        default=None,
        help="Only include squares with width >= this value (drawing units).",
    )
    ap.add_argument(
        "--max-width",
        type=float,
        default=None,
        help="Only include squares with width <= this value (drawing units).",
    )
    ap.add_argument("--tol-point", type=float, default=1e-4, help="Point snapping tolerance used for endpoint matching.")
    ap.add_argument(
        "--tol-angle-cos",
        type=float,
        default=1e-2,
        help="Max |cos(theta)| for a corner to be treated as 90 degrees. Smaller is stricter.",
    )
    ap.add_argument(
        "--tol-len-rel",
        type=float,
        default=2e-2,
        help="Max relative side length mismatch for a square. Smaller is stricter.",
    )
    args = ap.parse_args()

    doc = ezdxf.readfile(args.input)
    green_layers = list(_iter_green_layers(doc))
    if not green_layers:
        raise SystemExit("No green (ACI=3) layers found in DXF.")

    lines = _load_lines(doc, green_layers)
    if not lines:
        raise SystemExit("No LINE entities found on green layers.")

    squares = _detect_square_line_sets(
        lines,
        tol_point=args.tol_point,
        tol_angle_cos=args.tol_angle_cos,
        tol_len_rel=args.tol_len_rel,
    )
    if not squares:
        raise SystemExit("No squares detected (try relaxing tolerances).")

    out = ezdxf.new(doc.dxfversion)
    msp_out = out.modelspace()

    # Create a single green layer for output to keep it clean.
    out_layer = "GREEN_SQUARES_ONLY"
    if out_layer not in out.layers:
        out.layers.new(out_layer, dxfattribs={"color": 3})

    # Emit each square as a closed polyline (more compact than 4 LINEs).
    kept = 0
    bbox_rows: list[dict[str, object]] = []
    details_rows: list[dict[str, object]] = []

    # Ensure stable ordering (sets are unordered).
    squares_sorted = sorted(squares, key=lambda idxs: "_".join(sorted(lines[idx].handle for idx in idxs)))

    for idxs in squares_sorted:
        pt_keys = set()
        pts: list[tuple[float, float, float]] = []
        for idx in idxs:
            a = lines[idx].start
            b = lines[idx].end
            for p in (a, b):
                k = _key(p, args.tol_point)
                if k not in pt_keys:
                    pt_keys.add(k)
                    pts.append(p)

        if len(pts) != 4:
            # Shouldn't happen for clean squares, but skip anything odd.
            continue

        ordered = _ordered_square_vertices(pts)
        width = _square_width_from_vertices(ordered)
        if width is None:
            continue
        if args.min_width is not None and width < args.min_width:
            continue
        if args.max_width is not None and width > args.max_width:
            continue

        bottomX, bottomY, topX, topY = _bbox_xy(ordered)
        area = width * width
        entity_id = "sq_" + "_".join(sorted(lines[idx].handle for idx in idxs))

        # Keep z from points (often constant); ezdxf expects (x, y [,start_width, end_width, bulge])
        msp_out.add_lwpolyline([(p[0], p[1]) for p in ordered], close=True, dxfattribs={"layer": out_layer, "color": 3})
        kept += 1

        bbox_rows.append(
            {
                "id": f"footing_{kept:03d}",
                "bottomX": bottomX,
                "bottomY": bottomY,
                "bottomZ": 0,
                "topX": topX,
                "topY": topY,
                "topZ": 0,
            }
        )

        details_rows.append(
            {
                "entityId": entity_id,
                "topX": topX,
                "topY": topY,
                "bottomX": bottomX,
                "bottomY": bottomY,
                "width": width,
                "area": area,
            }
        )

    out.saveas(args.output)
    csv_path = args.csv_output or f"{args.output}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "bottomX", "bottomY", "bottomZ", "topX", "topY", "topZ"])
        writer.writeheader()
        writer.writerows(bbox_rows)

    if args.details_csv_output:
        with open(args.details_csv_output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["entityId", "topX", "topY", "bottomX", "bottomY", "width", "area"],
            )
            writer.writeheader()
            writer.writerows(details_rows)

    print(f"Wrote {kept} squares to {args.output}")
    print(f"Wrote CSV to {csv_path}")
    if args.details_csv_output:
        print(f"Wrote detailed CSV to {args.details_csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

