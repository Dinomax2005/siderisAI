# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for Sideris Desktop.

Build with:   pyinstaller --noconfirm Luno.spec
Output:       dist/Sideris/            (onedir app, Sideris + SiderisCLI)
              dist/Sideris.app         (macOS bundle)

The file is still named Luno.spec because the 5.0 line started as "Luno
Desktop" and build scripts reference it; the product and every visible
name is Sideris.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve()  # noqa: F821 - provided by PyInstaller
VERSION = "5.0.0"

datas = []
binaries = []
hiddenimports = [
    "sideris_desktop",
    "sideris_desktop.app",
    "sideris_desktop.cli",
    "sideris_desktop.server",
    "sideris_desktop.installer",
    "sideris_desktop.service",
    "sideris_desktop.training",
    "sideris_desktop.train_worker",
    "sideris_desktop.core",
    "sideris_desktop.inference",
    "sideris_desktop.downloads",
    "sideris_desktop.models",
    "sideris_desktop.config",
    "tkinter",
    "tkinter.ttk",
    "tkinter.font",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "http.server",
    "urllib.request",
    "plistlib",
]

for name in ("logo.png", "logo-256.png", "sideris.ico", "sideris.icns"):
    path = ROOT / "assets" / name
    if path.is_file():
        datas.append((str(path), "assets"))

for path in sorted((ROOT / "sideris_desktop").glob("*.py")):
    datas.append((str(path), "sideris_source/sideris_desktop"))
if (ROOT / "sideris_launcher.py").is_file():
    datas.append((str(ROOT / "sideris_launcher.py"), "sideris_source"))
for extra in ("pyproject.toml", "README.md", "LICENSE"):
    path = ROOT / extra
    if path.is_file():
        datas.append((str(path), "sideris_source"))

# The inference runtime is bundled into the installer so the shipped app
# works offline out of the box.  Training (torch/transformers/peft) is
# deliberately NOT bundled: it is installed into a private venv only when
# the training permission is granted.
try:
    lc_datas, lc_binaries, lc_hidden = collect_all("llama_cpp")
    datas += lc_datas
    binaries += lc_binaries
    hiddenimports += lc_hidden
    # llama_cpp imports these at runtime; name them explicitly so a late
    # import can never fail inside the frozen app.
    hiddenimports += ["diskcache", "jinja2", "numpy", "typing_extensions"]
except Exception as exc:  # pragma: no cover - build environment dependent
    print(f"[Luno.spec] llama_cpp not bundled: {exc}")

excludes = [
    "torch", "torchvision", "torchaudio", "transformers", "peft", "datasets",
    "accelerate", "tensorflow", "keras", "jax", "flax", "sklearn", "scipy",
    "pandas", "matplotlib", "IPython", "notebook", "jupyter", "PyQt5",
    "PyQt6", "PySide2", "PySide6", "wx", "cv2", "numba", "onnxruntime",
    "sentencepiece", "bitsandbytes", "deepspeed",
]

icon = str(ROOT / "assets" / ("sideris.ico" if sys.platform == "win32" else "sideris.icns"))
if sys.platform.startswith("linux"):
    icon = str(ROOT / "assets" / "logo-256.png")

# ``sideris_launcher.py`` imports the package instead of executing
# ``sideris_desktop/__main__.py`` as a bare script, which would break its
# relative imports inside the frozen app.
a = Analysis(
    [str(ROOT / "sideris_launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sideris",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

cli = None
if sys.platform != "darwin":
    # A terminal binary is genuinely useful on Windows/Linux.  The macOS
    # .app already contains the GUI and a second executable inside the
    # bundle adds signing complexity for no benefit.
    cli = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="SiderisCLI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon,
    )

collect_targets = [exe] + ([cli] if cli is not None else [])

coll = COLLECT(
    *collect_targets,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Sideris",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Sideris.app",
        icon=str(ROOT / "assets" / "sideris.icns"),
        bundle_identifier="ai.sideris.desktop",
        version=VERSION,
        info_plist={
            "CFBundleName": "Sideris",
            "CFBundleDisplayName": "Sideris",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "Sideris Desktop 5.0.0 - Apache-2.0",
            "LSApplicationCategoryType": "public.app-category.productivity",
        },
    )
