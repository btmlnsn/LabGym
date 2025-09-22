# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# Use repo root as CWD on CI and locally
ROOT    = os.path.abspath(os.getcwd())
PKG_DIR = os.path.join(ROOT, "LabGym")
SPEC_DIR = os.path.join(PKG_DIR, "pyinstaller")

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

hiddenimports, binaries = [], []

# Big libs that use dynamic import / native libs
for mod in ["torch", "torchvision", "detectron2", "tensorflow"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Your package
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass

# Ensure tomli is included when running on Python < 3.11
try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

# ---- Datas ---------------------------------------------------------------
datas = [
    # app assets
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]

# ship logging.yaml if present
log_cfg = os.path.join(ROOT, "logging.yaml")
if os.path.exists(log_cfg):
    datas.append((log_cfg, "LabGym"))

# *** CRUCIAL ***
# Ship the vendored detectron2 sources as real files so TorchScript can read them
D2_SRC = os.path.join(PKG_DIR, "detectron2")
if os.path.isdir(D2_SRC):
    datas.append((D2_SRC, "LabGym/detectron2"))

a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],
    pathex=[ROOT],                 # make "LabGym" importable
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    # keep .py *and* .pyz for vendored detectron2 so sources exist at runtime
    module_collection_mode={
        "LabGym.detectron2": "pyz+py",
        # If you also import upstream detectron2 from pip/conda anywhere, uncomment:
        # "detectron2": "pyz+py",
    },
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON if os.path.exists(ICON) else None,
    console=False,  # change to True if you want a console during local debugging
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
