# OPF Marker BBox Tools

This project contains two main scripts:

- [assign_markers_to_bboxes.py](/home/amjed/git/opf-renderer/assign_markers_to_bboxes.py)
- [view_papers_in_rerun.py](/home/amjed/git/opf-renderer/view_papers_in_rerun.py)

The first script assigns raw marker detections to world-space bounding boxes. The second script visualizes the result in Rerun and also exports annotated images.

## Workflow

1. Start with a raw marker CSV such as `final.csv`.
2. Provide a bbox CSV with real-world XY bounds for each footing or paper.
3. Provide OPF metadata so image detections can be projected onto the reconstructed ground plane.
4. Run `assign_markers_to_bboxes.py` to produce a prefixed output CSV.
5. Run `view_papers_in_rerun.py` to inspect the result in 3D and on annotated images.

## Script 1: Assign Markers To BBoxes

[`assign_markers_to_bboxes.py`](/home/amjed/git/opf-renderer/assign_markers_to_bboxes.py) is the production executable.

It does the following:

- loads calibrated camera poses and intrinsics from OPF sidecar JSON
- fits a ground plane from calibrated control points
- projects each 2D marker detection into 3D world space
- checks which bbox contains that projected point in XY
- prefixes the marker ID with the bbox ID
- writes the final output CSV

### Inputs

- `--marker-csv`
  Raw marker detections. Expected rows:
  `image_name,marker_id,x_px,y_px[,score]`

- `--bbox-csv`
  World-space bbox CSV. Required columns:
  `id,bottomX,bottomY,topX,topY`

  Optional columns:
  `bottomZ,topZ`

- OPF metadata, provided either by:
  - `--opf-root <opf_export_dir>`
  - or `--opf-json-dir <opf_export_dir/opf_files>`

### Minimal OPF Files Required

You do not need the full OPF object model for this script. These files are sufficient:

- [camera_list.json](sample/opf_files/camera_list.json)
- [calibrated_cameras.json](sample/opf_files/calibrated_cameras.json)
- [input_cameras.json](sample/opf_files/input_cameras.json)
- [calibrated_control_points.json](sample/opf_files/calibrated_control_points.json)

### Example

```bash
.venv/bin/python assign_markers_to_bboxes.py \
  --marker-csv sample/final.csv \
  --bbox-csv sample/paper_bboxes.csv \
  --opf-root sample \
  --output-csv sample/final_bbox_prefixed.csv
```

### Output

The output CSV contains:

- `image_name`
- `bbox_id`
- `prefixed_marker_id`
- `marker_id`
- `x_px`
- `y_px`
- `score`
- `world_x`
- `world_y`
- `world_z`

Example prefixed marker ID:

```text
paper_01_1x12:015
```

In the real construction workflow, this would become something like:

```text
footing_102_marker_7
```

## Script 2: View Papers In Rerun

[`view_papers_in_rerun.py`](/home/amjed/git/opf-renderer/view_papers_in_rerun.py) visualizes the prefixed CSV together with the OPF point cloud.

It does the following:

- loads the prefixed output CSV
- loads the OPF point cloud and control points
- groups markers by `bbox_id`
- treats the 4 averaged marker points as the actual paper corners
- fits a paper plane from those 4 corner points
- computes paper tilt relative to the ground plane
- logs paper outlines, marker labels, tilt labels, and local Z axes to Rerun
- exports annotated JPEG images at 50% quality

### Inputs

- `--csv`
  Output from `assign_markers_to_bboxes.py`

- `--opf`
  Path to `project.opf` or the OPF export root directory containing it

- `--output-images`
  Directory for annotated image export

### Example

```bash
.venv/bin/python view_papers_in_rerun.py \
  --csv sample/final_bbox_prefixed.csv \
  --opf sample \
  --output-images modified_images_paper_view
```

### What It Shows

In Rerun:

- point cloud
- paper corner polygons
- paper center labels
- local paper `Z` axis
- ground reference axes
- marker labels with per-corner height above ground in mm

On exported images:

- projected corner polygon
- projected `Z`, `GX`, and `GY` reference lines
- marker labels
- tilt annotations

## Accuracy Notes

These tools are useful for inspection, labeling, and relative tilt analysis, but they do not by themselves guarantee mm-level construction accuracy.

For mm-level trust, you still need to validate:

- control point survey accuracy
- camera calibration quality
- reconstruction residuals / bundle adjustment residuals
- plane-fit residuals per paper
- per-corner height consistency
- whether the paper is actually planar in the scene

The viewer currently reports:

- total tilt magnitude
- signed tilt along ground X
- signed tilt along ground Y
- per-corner height relative to the fitted ground plane in mm

## Current Recommended Commands

Assign markers:

```bash
.venv/bin/python assign_markers_to_bboxes.py \
  --marker-csv sample/final.csv \
  --bbox-csv sample/paper_bboxes.csv \
  --opf-root sample \
  --output-csv sample/final_bbox_prefixed.csv
```

Visualize:

```bash
.venv/bin/python view_papers_in_rerun.py \
  --csv sample/final_bbox_prefixed.csv \
  --opf sample \
  --output-images modified_images_paper_view
```
