# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# Use the workflow's CWD (repo root)
ROOT    = os.path.abspath(os.getcwd())
PKG_DIR = os.path.join(ROOT, "LabGym")
SPEC_DIR = os.path.join(PKG_DIR, "pyinstaller")

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join(PKG_DIR, "assets", "icons", "labgym.icns")

hiddenimports, binaries = [], []
for mod in ["torch", "torchvision", "detectron2", "tensorflow"]:
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Pull in your own package and assets
try:
    hiddenimports += collect_submodules("LabGym")
except Exception:
    pass

try:
    hiddenimports += collect_submodules("tomli")
except Exception:
    hiddenimports += ["tomli"]

datas = [
    (os.path.join(PKG_DIR, "assets"), "LabGym/assets"),
]
if os.path.exists(os.path.join(ROOT, "logging.yaml")):
    datas.append((os.path.join(ROOT, "logging.yaml"), "LabGym"))

a = Analysis(
    [os.path.join(PKG_DIR, "__main__.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, "hooks")],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON,
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
