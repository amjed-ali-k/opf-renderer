# Build Windows `.exe` (GUI)

This repo includes a small Qt (PySide6) GUI for `assign_markers_to_bboxes`:

- GUI entrypoint: `assign_markers_to_bboxes_qt.py` (Qt / PySide6)
- CLI copy: `assign_markers_to_bboxes_cli.py`
- PyInstaller spec: `packaging/assign_markers_to_bboxes_gui.spec`

## Build steps (Windows)

1. Install Python (recommended: 64-bit Python 3.11+).
2. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy
pip install pyside6
pip install pyinstaller
```

If you need the same dependency set as the repo uses, install via `uv` or install from your own lockfile process. For the GUI `.exe`, `numpy` + `pyside6` are required and the rest are not needed.

1. Build the exe:

```powershell
pyinstaller --clean --noconfirm packaging\assign_markers_to_bboxes_gui.spec
```

1. Output:

- `dist\assign_markers_to_bboxes\assign_markers_to_bboxes.exe`

You can zip the whole `dist\assign_markers_to_bboxes\` folder and share it with users.