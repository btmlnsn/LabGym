# -*- mode: python ; coding: utf-8 -*-
import os
from glob import glob
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

# Resolve paths from repo root (runner CWD)
ROOT      = os.path.abspath(os.getcwd())
PKG_DIR   = os.path.join(ROOT, "LabGym")
HOOKS_DIR = os.path.join(PKG_DIR, "pyinstaller", "hooks")

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

# ---------------- Hidden imports / binaries ----------------
hiddenimports, binaries = [], []

# Your package (belt & suspenders)
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass

# Vendor ML stacks (safe to try even if not present)
for mod in ["torch", "torchvision"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Common detectron2 deps that are sometimes imported dynamically
for mod in ["fvcore", "iopath", "yacs"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass

# Py3.10 TOML fallback (your code uses tomli when tomllib is missing)
try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

# ---------------- Data files ----------------
datas = [
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]

# Logging config if present at repo root
if os.path.exists(os.path.join(ROOT, "logging.yaml")):
    datas.append((os.path.join(ROOT, "logging.yaml"), "LabGym"))

# Include any top-level YAML/TOML in LabGym/ used at runtime
for y in glob(os.path.join(PKG_DIR, "*.y*ml")):
    datas.append((y, "LabGym"))
for t in glob(os.path.join(PKG_DIR, "*.toml")):
    datas.append((t, "LabGym"))

# 🔑 Ship LabGym.detectron2 as real .py sources (not frozen/zipped)
#    This makes TorchScript/JIT happy when it calls inspect.getsource().
datas += collect_data_files("LabGym.detectron2", include_py_files=True)

# Only pass hookspath if the folder exists
hookspath = [HOOKS_DIR] if os.path.isdir(HOOKS_DIR) else []

# ---------------- Build graph ----------------
# Prefer collecting LabGym.detectron2 as source files:
MODULE_COLLECTION_MODE = {
    # Put all submodules of LabGym.detectron2 on disk as .py files
    "LabGym.detectron2": "py",
}

a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],  # entry script
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    noarchive=False,
    module_collection_mode=MODULE_COLLECTION_MODE,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON,
    console=False,  # windowed app
)

GIT_SHA = os.getenv("GITHUB_SHA", "dev")[:7]

app = BUNDLE(
    exe,
    name=f"{APP_NAME}.app",
    bundle_identifier=BUNDLE_ID,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": f"0.1.0+{GIT_SHA}",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)
