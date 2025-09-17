# -*- mode: python ; coding: utf-8 -*-
import os
from glob import glob
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# Resolve paths from the workflow's CWD (repo root)
ROOT     = os.path.abspath(os.getcwd())
PKG_DIR  = os.path.join(ROOT, "LabGym")
HOOKS_DIR = os.path.join(PKG_DIR, "pyinstaller", "hooks")

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

# ---- Hidden imports / binaries ------------------------------------------------
hiddenimports, binaries = [], []

# Your own package (belt & suspenders)
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass
# If you use detectron2 under LabGym (matches John's --collect-all LabGym.detectron2)
try:
    hiddenimports += collect_submodules("LabGym.detectron2")
except Exception:
    pass

# ML stacks (optional; safe if not installed)
for mod in ["torch", "torchvision", "tensorflow"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Py3.10 TOML fallback — make sure tomli is bundled even if import is guarded
try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

# ---- Data files ---------------------------------------------------------------
datas = [
    # package assets (icons, etc.)
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]

# Logging config (equivalent to John's --add-data=../logging.yaml:LabGym)
if os.path.exists(os.path.join(ROOT, "logging.yaml")):
    datas.append((os.path.join(ROOT, "logging.yaml"), "LabGym"))

# Optional: include any top-level YAML/TOML in LabGym/ used at runtime
for y in glob(os.path.join(PKG_DIR, "*.y*ml")):
    datas.append((y, "LabGym"))
for t in glob(os.path.join(PKG_DIR, "*.toml")):
    datas.append((t, "LabGym"))

# Only pass hookspath if the folder exists (avoids path errors)
hookspath = [HOOKS_DIR] if os.path.isdir(HOOKS_DIR) else []

# ---- Build graph --------------------------------------------------------------
a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],  # app entry
    pathex=[ROOT],                           # make 'LabGym' importable during analysis
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON,
    console=False,   # windowed app
)

# Optional: embed the CI commit SHA in the bundle version
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
