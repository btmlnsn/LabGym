# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

ROOT     = os.path.abspath(os.getcwd())
PKG_DIR  = os.path.join(ROOT, "LabGym")
SPEC_DIR = os.path.join(PKG_DIR, "pyinstaller")

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

hiddenimports = []
binaries = []

# Torch + torchvision: submodules and native libs
for mod in ("torch", "torchvision"):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# If upstream detectron2 is imported anywhere, include it too
for mod in ("detectron2",):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Your own package
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass

# tomli for py<3.11 fallback (your logging uses tomllib/tomli)
try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

datas = [
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]

log_cfg = os.path.join(ROOT, "logging.yaml")
if os.path.exists(log_cfg):
    datas.append((log_cfg, "LabGym"))

# Ship vendored detectron2 sources as *real files* (TorchScript needs them).
D2_VENDOR = os.path.join(PKG_DIR, "detectron2")
if os.path.isdir(D2_VENDOR):
    datas.append((D2_VENDOR, "LabGym/detectron2"))

a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    # Ensure .py sources are written to disk, not just into a .pyz
    noarchive=True,
    # Also force PyInstaller to emit .py files for these modules
    module_collection_mode={
        "LabGym.detectron2": "pyz+py",
        "detectron2": "pyz+py",   # harmless if not present
    },
)

pyz = PYZ(a.pure, a.zipped_data)

# Build an onedir bundle (loose files), not a one-file stub.
exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON if os.path.exists(ICON) else None,
    console=False,
)

# Gather into a folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

# Wrap the folder into a proper macOS .app bundle
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    bundle_identifier=BUNDLE_ID,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)
