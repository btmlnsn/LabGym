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

# Collect torch & torchvision binaries + submodules
for mod in ("torch", "torchvision"):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# If you import upstream detectron2 as well, include it too
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

# tomli for py<3.11 (your mylogging/config uses tomllib/tomli)
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

# IMPORTANT: ship vendored detectron2 sources as real files
# (TorchScript needs them at runtime)
D2_VENDOR = os.path.join(PKG_DIR, "detectron2")
if os.path.isdir(D2_VENDOR):
    datas.append((D2_VENDOR, "LabGym/detectron2"))

a = Analysis(
    # keep your real entry point — no env hacks (Torch stays ON)
    [os.path.join(PKG_DIR, "__main__.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    # Force PyInstaller to *also* write .py files on disk
    #   for the modules that TorchScript inspects.
    # Your app uses the vendored detectron2; include upstream if used.
    module_collection_mode={
        "LabGym.detectron2": "pyz+py",
        "detectron2": "pyz+py",        # keep if you import upstream too
    },
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON if os.path.exists(ICON) else None,
    console=False,
)

app = BUNDLE(
    exe,
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
