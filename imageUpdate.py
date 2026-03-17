from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class MarkerObservation:
    marker_id: str
    x: float
    y: float


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "opf" / "images"
CSV_PATH = PROJECT_ROOT / "opf" / "final.csv"
OUTPUT_DIR = PROJECT_ROOT / "modified_images"

MARKER_RADIUS = 18
LABEL_MARGIN = 12
MARKER_COLOR = (255, 64, 64)
LABEL_BG_COLOR = (0, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 0)


def load_marker_map(csv_path: Path) -> dict[str, list[MarkerObservation]]:
    marker_map: dict[str, list[MarkerObservation]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                raise ValueError(f"Invalid row {row_number} in {csv_path}: {row!r}")

            image_name = Path(row[0].strip()).name
            marker_id = row[1].strip()
            x = float(row[2].strip())
            y = float(row[3].strip())

            marker_map[image_name].append(
                MarkerObservation(marker_id=marker_id, x=x, y=y)
            )

    return dict(marker_map)


def draw_marker(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, marker: MarkerObservation
) -> None:
    x = round(marker.x)
    y = round(marker.y)

    draw.ellipse(
        (x - MARKER_RADIUS, y - MARKER_RADIUS, x + MARKER_RADIUS, y + MARKER_RADIUS),
        outline=MARKER_COLOR,
        width=5,
    )
    draw.line((x - MARKER_RADIUS, y, x + MARKER_RADIUS, y), fill=MARKER_COLOR, width=3)
    draw.line((x, y - MARKER_RADIUS, x, y + MARKER_RADIUS), fill=MARKER_COLOR, width=3)

    text = marker.marker_id
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    label_left = x + LABEL_MARGIN
    label_top = y - text_height - LABEL_MARGIN
    label_box = (
        label_left - 8,
        label_top - 6,
        label_left + text_width + 8,
        label_top + text_height + 6,
    )

    draw.rectangle(label_box, fill=LABEL_BG_COLOR)
    draw.text((label_left, label_top), text, font=font, fill=LABEL_TEXT_COLOR)


def annotate_images() -> tuple[int, int]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input image directory not found: {INPUT_DIR}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Marker CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    marker_map = load_marker_map(CSV_PATH)
    font = ImageFont.load_default(size=24)

    processed_count = 0
    skipped_count = 0

    for image_path in sorted(INPUT_DIR.iterdir()):
        if not image_path.is_file():
            continue

        markers = marker_map.get(image_path.name)
        if not markers:
            skipped_count += 1
            continue

        with Image.open(image_path) as source_image:
            annotated = source_image.convert("RGB")

        draw = ImageDraw.Draw(annotated)
        for marker in markers:
            draw_marker(draw, font, marker)

        annotated.save(OUTPUT_DIR / image_path.name, quality=95)
        processed_count += 1

    return processed_count, skipped_count


def main() -> None:
    processed_count, skipped_count = annotate_images()
    print(f"Annotated {processed_count} images into {OUTPUT_DIR}")
    print(f"Skipped {skipped_count} images with no marker rows")


if __name__ == "__main__":
    main()
