from __future__ import annotations

import csv
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from PySide6 import QtCore, QtGui, QtWidgets

import assign_markers_to_bboxes_cli as core


def apply_modern_style(app: QtWidgets.QApplication) -> None:
    # Use a consistent base style across platforms.
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    palette = QtGui.QPalette()
    # Dark-ish neutral theme (pleasant for long runs).
    base = QtGui.QColor("#0f1115")
    window = QtGui.QColor("#0f1115")
    alt_base = QtGui.QColor("#141825")
    text = QtGui.QColor("#e8eaf0")
    disabled_text = QtGui.QColor("#7f8796")
    button = QtGui.QColor("#1a2030")
    button_text = QtGui.QColor("#e8eaf0")
    highlight = QtGui.QColor("#2f6fed")
    highlighted_text = QtGui.QColor("#ffffff")

    palette.setColor(QtGui.QPalette.ColorRole.Window, window)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Base, base)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, alt_base)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QtGui.QPalette.ColorRole.Text, text)
    palette.setColor(QtGui.QPalette.ColorRole.Button, button)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, button_text)
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff4d4d"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, highlighted_text)
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#9aa3b2"))

    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled_text)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled_text)
    palette.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.WindowText, disabled_text)

    app.setPalette(palette)

    app.setFont(QtGui.QFont("Segoe UI" if "Segoe UI" in QtGui.QFontDatabase.families() else app.font().family(), 10))

    app.setStyleSheet(
        """
        QWidget { color: #e8eaf0; }

        QGroupBox {
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 10px;
          margin-top: 12px;
          padding: 12px;
          background: rgba(255,255,255,0.02);
        }
        QGroupBox::title {
          subcontrol-origin: margin;
          left: 10px;
          padding: 0 6px;
          color: rgba(232,234,240,0.85);
        }

        QLineEdit, QDoubleSpinBox, QPlainTextEdit {
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 10px;
          padding: 8px 10px;
          background: rgba(255,255,255,0.03);
          selection-background-color: #2f6fed;
        }
        QLineEdit:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
          border: 1px solid rgba(47,111,237,0.9);
          background: rgba(47,111,237,0.08);
        }

        QPushButton {
          border: 1px solid rgba(255,255,255,0.14);
          border-radius: 10px;
          padding: 8px 12px;
          background: rgba(255,255,255,0.04);
        }
        QPushButton:hover { background: rgba(255,255,255,0.07); }
        QPushButton:pressed { background: rgba(255,255,255,0.10); }
        QPushButton:disabled {
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.02);
          color: rgba(232,234,240,0.45);
        }

        QPushButton#PrimaryButton {
          border: 1px solid rgba(47,111,237,0.90);
          background: rgba(47,111,237,0.90);
          color: #ffffff;
          font-weight: 600;
        }
        QPushButton#PrimaryButton:hover { background: rgba(47,111,237,0.98); }
        QPushButton#PrimaryButton:pressed { background: rgba(47,111,237,0.75); }

        QProgressBar {
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 8px;
          background: rgba(255,255,255,0.03);
          text-align: center;
        }
        QProgressBar::chunk { background: rgba(47,111,237,0.85); border-radius: 8px; }

        QRadioButton { spacing: 8px; }
        QRadioButton::indicator {
          width: 16px; height: 16px;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.18);
          background: rgba(255,255,255,0.03);
        }
        QRadioButton::indicator:checked {
          border: 1px solid rgba(47,111,237,0.95);
          background: rgba(47,111,237,0.95);
        }

        QScrollBar:vertical { width: 12px; margin: 2px; background: transparent; }
        QScrollBar::handle:vertical { background: rgba(255,255,255,0.10); border-radius: 6px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: rgba(255,255,255,0.16); }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """
    )


DEFAULT_OUTPUT_COLUMNS: tuple[str, ...] = (
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
)


def write_output_csv_selected_columns(
    rows: list[dict[str, str]],
    output_csv: Path,
    selected_columns: list[str],
    *,
    exclude_unassigned: bool,
) -> None:
    if not selected_columns:
        raise ValueError("Select at least one output column")

    unknown = [c for c in selected_columns if c not in DEFAULT_OUTPUT_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown output columns: {unknown}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_columns)
        writer.writeheader()
        filtered_rows = rows
        if exclude_unassigned:
            filtered_rows = [row for row in rows if row.get("bbox_id") != "UNASSIGNED"]

        for row in sorted(
            filtered_rows,
            key=lambda item: (
                item.get("image_name", ""),
                item.get("bbox_id", ""),
                item.get("marker_id", ""),
                float(item.get("x_px", "0") or 0.0),
                float(item.get("y_px", "0") or 0.0),
            ),
        ):
            writer.writerow({k: row.get(k, "") for k in selected_columns})


@dataclass(frozen=True)
class GuiInputs:
    marker_csv: Path
    bbox_csv: Path
    output_csv: Path
    opf_root: Path | None
    opf_json_dir: Path | None
    bbox_padding_z: float
    selected_output_columns: list[str]
    exclude_unassigned: bool


class Worker(QtCore.QObject):
    log = QtCore.Signal(str)
    finished = QtCore.Signal(int, int, str)  # total, assigned, output
    failed = QtCore.Signal(str)

    def __init__(self, inputs: GuiInputs) -> None:
        super().__init__()
        self._inputs = inputs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            total, assigned = self._run_pipeline(self._inputs)
            self.finished.emit(total, assigned, str(self._inputs.output_csv))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_pipeline(self, inputs: GuiInputs) -> tuple[int, int]:
        args = SimpleNamespace(
            marker_csv=inputs.marker_csv,
            bbox_csv=inputs.bbox_csv,
            output_csv=inputs.output_csv,
            opf_root=inputs.opf_root,
            opf_json_dir=inputs.opf_json_dir,
            camera_list_json=None,
            calibrated_cameras_json=None,
            input_cameras_json=None,
            control_points_json=None,
            bbox_padding_z=inputs.bbox_padding_z,
        )

        self.log.emit("Loading OPF metadata…")
        paths = core.resolve_opf_paths(args)
        camera_models = core.build_camera_models(
            camera_list_json=paths["camera_list"],
            calibrated_cameras_json=paths["calibrated_cameras"],
            input_cameras_json=paths["input_cameras"],
        )
        plane = core.orient_plane_toward_cameras(core.fit_plane(paths["control_points"]), camera_models)

        self.log.emit("Loading marker CSV…")
        marker_rows = core.load_marker_rows(inputs.marker_csv)
        self.log.emit(f"Loaded {len(marker_rows)} marker rows")

        self.log.emit("Projecting markers to ground plane…")
        projected_rows = core.project_marker_rows(marker_rows, camera_models, plane)
        all_z = [float(core.np.asarray(row["world_point"])[2]) for row in projected_rows]

        self.log.emit("Loading bounding boxes…")
        boxes = core.load_bboxes(
            inputs.bbox_csv,
            default_bottom_z=min(all_z) - inputs.bbox_padding_z,
            default_top_z=max(all_z) + inputs.bbox_padding_z,
        )
        bbox_index = core.build_bbox_index(boxes)
        self.log.emit(f"Loaded {len(boxes)} bounding boxes")

        self.log.emit("Assigning markers to boxes…")
        assigned_rows = core.assign_rows(projected_rows, bbox_index)
        write_output_csv_selected_columns(
            rows=assigned_rows,
            output_csv=inputs.output_csv,
            selected_columns=inputs.selected_output_columns,
            exclude_unassigned=inputs.exclude_unassigned,
        )

        assigned_count = sum(1 for row in assigned_rows if row["bbox_id"] != "UNASSIGNED")
        return len(assigned_rows), assigned_count


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Assign Markers to BBoxes")
        self.setMinimumSize(900, 600)

        self._thread: QtCore.QThread | None = None
        self._worker: Worker | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Assign Markers to BBoxes")
        title.setFont(QtGui.QFont(title.font().family(), 16, QtGui.QFont.Weight.DemiBold))
        subtitle = QtWidgets.QLabel("Pick your input CSVs + OPF metadata, then run to generate a bbox-prefixed output CSV.")
        subtitle.setStyleSheet("color: rgba(232,234,240,0.72);")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form_card = QtWidgets.QGroupBox("Inputs")
        form = QtWidgets.QFormLayout(form_card)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)
        layout.addWidget(form_card)

        self.marker_edit = QtWidgets.QLineEdit()
        self.marker_edit.setPlaceholderText("e.g. final.csv")
        self.marker_btn = QtWidgets.QPushButton("Browse…")
        self.marker_btn.clicked.connect(self._browse_marker)
        form.addRow("Marker CSV", self._hbox(self.marker_edit, self.marker_btn))

        self.bbox_edit = QtWidgets.QLineEdit()
        self.bbox_edit.setPlaceholderText("e.g. footing_bboxes.csv")
        self.bbox_btn = QtWidgets.QPushButton("Browse…")
        self.bbox_btn.clicked.connect(self._browse_bbox)
        form.addRow("BBox CSV", self._hbox(self.bbox_edit, self.bbox_btn))

        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("e.g. final_bbox_prefixed.csv")
        self.output_btn = QtWidgets.QPushButton("Save as…")
        self.output_btn.clicked.connect(self._browse_output)
        form.addRow("Output CSV", self._hbox(self.output_edit, self.output_btn))

        opf_group = QtWidgets.QGroupBox("OPF metadata location")
        opf_layout = QtWidgets.QGridLayout(opf_group)
        opf_layout.setHorizontalSpacing(12)
        opf_layout.setVerticalSpacing(10)

        self.opf_root_radio = QtWidgets.QRadioButton("OPF root (contains opf_files/)")
        self.opf_dir_radio = QtWidgets.QRadioButton("opf_files directory")
        self.opf_root_radio.setChecked(True)
        self.opf_root_radio.toggled.connect(self._refresh_opf_mode)
        self.opf_dir_radio.toggled.connect(self._refresh_opf_mode)

        self.opf_root_edit = QtWidgets.QLineEdit()
        self.opf_root_edit.setPlaceholderText("Folder that contains opf_files/")
        self.opf_root_btn = QtWidgets.QPushButton("Browse…")
        self.opf_root_btn.clicked.connect(self._browse_opf_root)

        self.opf_dir_edit = QtWidgets.QLineEdit()
        self.opf_dir_edit.setPlaceholderText("The opf_files folder")
        self.opf_dir_btn = QtWidgets.QPushButton("Browse…")
        self.opf_dir_btn.clicked.connect(self._browse_opf_dir)

        opf_layout.addWidget(self.opf_root_radio, 0, 0)
        opf_layout.addWidget(self.opf_root_edit, 0, 1)
        opf_layout.addWidget(self.opf_root_btn, 0, 2)
        opf_layout.addWidget(self.opf_dir_radio, 1, 0)
        opf_layout.addWidget(self.opf_dir_edit, 1, 1)
        opf_layout.addWidget(self.opf_dir_btn, 1, 2)

        layout.addWidget(opf_group)

        settings = QtWidgets.QHBoxLayout()
        layout.addLayout(settings)
        settings_label = QtWidgets.QLabel("BBox padding Z (m)")
        settings_label.setStyleSheet("color: rgba(232,234,240,0.80);")
        settings.addWidget(settings_label)
        self.padding_edit = QtWidgets.QDoubleSpinBox()
        self.padding_edit.setDecimals(4)
        self.padding_edit.setRange(0.0, 1000.0)
        self.padding_edit.setSingleStep(0.01)
        self.padding_edit.setValue(0.05)
        settings.addWidget(self.padding_edit)
        hint = QtWidgets.QLabel("Used only when bottomZ/topZ are missing in the bbox CSV.")
        hint.setStyleSheet("color: rgba(232,234,240,0.62);")
        settings.addWidget(hint)
        settings.addStretch(1)

        columns_group = QtWidgets.QGroupBox("Output columns")
        columns_layout = QtWidgets.QVBoxLayout(columns_group)
        columns_layout.setContentsMargins(12, 12, 12, 12)
        columns_layout.setSpacing(10)

        presets = QtWidgets.QHBoxLayout()
        columns_layout.addLayout(presets)

        self.columns_minimal_btn = QtWidgets.QPushButton("Minimal")
        self.columns_minimal_btn.clicked.connect(self._preset_minimal_columns)
        presets.addWidget(self.columns_minimal_btn)

        self.columns_full_btn = QtWidgets.QPushButton("Full")
        self.columns_full_btn.clicked.connect(self._preset_full_columns)
        presets.addWidget(self.columns_full_btn)

        presets.addStretch(1)

        self.columns_select_all_btn = QtWidgets.QPushButton("Select all")
        self.columns_select_all_btn.clicked.connect(lambda: self._set_all_columns(True))
        presets.addWidget(self.columns_select_all_btn)

        self.columns_select_none_btn = QtWidgets.QPushButton("Select none")
        self.columns_select_none_btn.clicked.connect(lambda: self._set_all_columns(False))
        presets.addWidget(self.columns_select_none_btn)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        columns_layout.addLayout(grid)

        self.column_checks: dict[str, QtWidgets.QCheckBox] = {}
        for idx, col in enumerate(DEFAULT_OUTPUT_COLUMNS):
            cb = QtWidgets.QCheckBox(col)
            self.column_checks[col] = cb
            grid.addWidget(cb, idx // 3, idx % 3)

        columns_help = QtWidgets.QLabel(
            "Tip: world_x/world_y/world_z are calculated internally for matching, "
            "but you can hide them from the output CSV."
        )
        columns_help.setStyleSheet("color: rgba(232,234,240,0.62);")
        columns_help.setWordWrap(True)
        columns_layout.addWidget(columns_help)

        self.exclude_unassigned_cb = QtWidgets.QCheckBox("Exclude UNASSIGNED rows (only keep assigned markers)")
        self.exclude_unassigned_cb.setChecked(False)
        columns_layout.addWidget(self.exclude_unassigned_cb)

        layout.addWidget(columns_group)

        run_row = QtWidgets.QHBoxLayout()
        layout.addLayout(run_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        run_row.addWidget(self.progress, 1)

        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)

        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(max(9, mono.pointSize()))
        self.log_edit.setFont(mono)
        self.log_edit.setPlaceholderText("Status output will appear here…")
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group, 1)

        self._preset_minimal_columns()
        self._refresh_opf_mode()

    def _hbox(self, edit: QtWidgets.QWidget, btn: QtWidgets.QWidget) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(edit, 1)
        l.addWidget(btn)
        return w

    def _append_log(self, msg: str) -> None:
        self.log_edit.appendPlainText(msg)

    def _browse_marker(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select marker CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.marker_edit.setText(path)
            if not self.output_edit.text().strip():
                p = Path(path)
                self.output_edit.setText(str(p.with_name(p.stem + "_bbox_prefixed.csv")))

    def _browse_bbox(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select bbox CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.bbox_edit.setText(path)

    def _browse_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self.output_edit.setText(path)

    def _browse_opf_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select OPF export root")
        if path:
            self.opf_root_edit.setText(path)

    def _browse_opf_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select opf_files directory")
        if path:
            self.opf_dir_edit.setText(path)

    def _refresh_opf_mode(self) -> None:
        root_mode = self.opf_root_radio.isChecked()
        self.opf_root_edit.setEnabled(root_mode)
        self.opf_root_btn.setEnabled(root_mode)
        self.opf_dir_edit.setEnabled(not root_mode)
        self.opf_dir_btn.setEnabled(not root_mode)

    def _set_all_columns(self, checked: bool) -> None:
        for cb in self.column_checks.values():
            cb.setChecked(checked)

    def _preset_full_columns(self) -> None:
        self._set_all_columns(True)

    def _preset_minimal_columns(self) -> None:
        minimal = {
            "image_name",
            "prefixed_marker_id",
            "x_px",
            "y_px",
            "score",
        }
        for name, cb in self.column_checks.items():
            cb.setChecked(name in minimal)

    def _validate(self) -> GuiInputs:
        marker = self.marker_edit.text().strip()
        bbox = self.bbox_edit.text().strip()
        output = self.output_edit.text().strip()
        if not marker:
            raise ValueError("Marker CSV is required")
        if not bbox:
            raise ValueError("BBox CSV is required")
        if not output:
            raise ValueError("Output CSV is required")

        if self.opf_root_radio.isChecked():
            opf_root = self.opf_root_edit.text().strip()
            if not opf_root:
                raise ValueError("OPF root is required (or switch to opf_files directory mode)")
            opf_root_path: Path | None = Path(opf_root)
            opf_dir_path: Path | None = None
        else:
            opf_dir = self.opf_dir_edit.text().strip()
            if not opf_dir:
                raise ValueError("opf_files directory is required (or switch to OPF root mode)")
            opf_root_path = None
            opf_dir_path = Path(opf_dir)

        selected_cols = [name for name, cb in self.column_checks.items() if cb.isChecked()]
        if not selected_cols:
            raise ValueError("Select at least one output column")

        return GuiInputs(
            marker_csv=Path(marker),
            bbox_csv=Path(bbox),
            output_csv=Path(output),
            opf_root=opf_root_path,
            opf_json_dir=opf_dir_path,
            bbox_padding_z=float(self.padding_edit.value()),
            selected_output_columns=selected_cols,
            exclude_unassigned=bool(self.exclude_unassigned_cb.isChecked()),
        )

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.progress.setVisible(running)
        self.marker_btn.setEnabled(not running)
        self.bbox_btn.setEnabled(not running)
        self.output_btn.setEnabled(not running)
        self.opf_root_btn.setEnabled(not running and self.opf_root_radio.isChecked())
        self.opf_dir_btn.setEnabled(not running and self.opf_dir_radio.isChecked())
        self.opf_root_radio.setEnabled(not running)
        self.opf_dir_radio.setEnabled(not running)
        self.columns_minimal_btn.setEnabled(not running)
        self.columns_full_btn.setEnabled(not running)
        self.columns_select_all_btn.setEnabled(not running)
        self.columns_select_none_btn.setEnabled(not running)
        self.exclude_unassigned_cb.setEnabled(not running)
        for cb in self.column_checks.values():
            cb.setEnabled(not running)

    def _run(self) -> None:
        if self._thread is not None:
            return

        try:
            inputs = self._validate()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Invalid inputs", str(exc))
            return

        self._append_log("Starting…")
        self._set_running(True)

        thread = QtCore.QThread(self)
        worker = Worker(inputs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        thread.start()

    @QtCore.Slot(int, int, str)
    def _on_finished(self, total: int, assigned: int, output: str) -> None:
        self._append_log(f"Done. Output: {output}")
        QtWidgets.QMessageBox.information(
            self,
            "Done",
            f"Wrote {total} rows\nAssigned {assigned} markers\n\nOutput:\n{output}",
        )

    @QtCore.Slot(str)
    def _on_failed(self, details: str) -> None:
        self._append_log(details)
        QtWidgets.QMessageBox.critical(self, "Failed", "Assignment failed. See log for details.")

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running(False)


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setApplicationName("Assign Markers to BBoxes")
    apply_modern_style(app)

    win = MainWindow()
    win.show()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

