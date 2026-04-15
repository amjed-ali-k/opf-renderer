from __future__ import annotations

# PyInstaller spec for building a standalone Windows GUI executable.
#
# Build on Windows:
#   pyinstaller --clean --noconfirm packaging/assign_markers_to_bboxes_gui.spec

import sys
from pathlib import Path

block_cipher = None

project_root = Path(__file__).resolve().parents[1]
entry = project_root / "assign_markers_to_bboxes_qt.py"

a = Analysis(
    [str(entry)],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        # NumPy sometimes needs explicit collection on Windows.
        "numpy",
        # Qt/PySide6 is usually auto-detected, but we keep it explicit.
        "PySide6",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="assign_markers_to_bboxes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

