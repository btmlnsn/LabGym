# -*- mode: python ; coding: utf-8 -*-
import os
from glob import glob
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

# Resolve from repo root (runner CWD)
ROOT       = os.path.abspath(os.getcwd())
PKG_DIR    = os.path.join(ROOT, "LabGym")
HOOKS_DIR  = os.path.join(PKG_DIR, "pyinstaller", "hooks")

APP_NAME   = "LabGym"
BUNDLE_ID  = "yelab.LabGym"
ICON       = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

# ---------------- Hidden imports / binaries ----------------
hiddenimports, binaries = [], []

# Your package
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass

# Torch stacks (safe to try even if not present)
for mod in ("torch", "torchvision"):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Common detectron2 deps that can be imported dynamically
for mod in ("fvcore", "iopath", "yacs", "omegaconf"):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass

# Py3.10 TOML fallback (your code imports tomli if tomllib missing)
try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

# ---------------- Data files ----------------
datas = [
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]

# Optional logging config at repo root
log_cfg = os.path.join(ROOT, "logging.yaml")
if os.path.exists(log_cfg):
    datas.append((log_cfg, "LabGym"))

# Include any top-level YAML/TOML under LabGym used at runtime
for y in glob(os.path.join(PKG_DIR, "*.y*ml")):
    datas.append((y, "LabGym"))
for t in glob(os.path.join(PKG_DIR, "*.toml")):
    datas.append((t, "LabGym"))

# ---- 🔑 Detectron2 sources on disk (TorchScript needs real .py) ----
EXCLUDES = []

# Case A: vendored detectron2 lives at LabGym/detectron2
VENDORED_D2 = os.path.join(PKG_DIR, "detectron2")
if os.path.isdir(VENDORED_D2):
    # copy the whole tree into the app at LabGym/detectron2
    datas.append((VENDORED_D2, "LabGym/detectron2"))
    # ensure PyInstaller does NOT freeze this package
    EXCLUDES.append("LabGym.detectron2")
else:
    # Case B: using upstream 'detectron2' package
    try:
        # copy its package files (including .py) into the app at top-level 'detectron2'
        datas += collect_data_files("detectron2", include_py_files=True)
        EXCLUDES.append("detectron2")
    except Exception:
        # if not installed, nothing to do (build will still succeed without D2)
        pass

# Only pass hookspath if the folder exists (for our runtime hook below)
hookspath = [HOOKS_DIR] if os.path.isdir(HOOKS_DIR) else []

# ---------------- Build graph ----------------
a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],  # entry
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    excludes=EXCLUDES,        # 👈 critical: import real .py files we copied
    noarchive=False,          # default is fine since D2 is excluded from freeze
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON if os.path.exists(ICON) else None,
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
