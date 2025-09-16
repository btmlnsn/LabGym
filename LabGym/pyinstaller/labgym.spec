# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

APP_NAME  = "LabGym"
BUNDLE_ID = "yelab.LabGym"
ICON      = os.path.join("LabGym", "assets", "icons", "labgym.icns")  # <— uses your existing icon

# If you don't need TF/detectron2 at runtime, remove them to shrink size.
hiddenimports, binaries = [], []
for mod in ["torch", "torchvision", "detectron2", "tensorflow"]:
    try: hiddenimports += collect_submodules(mod)
    except Exception: pass
    try: binaries += collect_dynamic_libs(mod)
    except Exception: pass

datas = []
if os.path.exists("logging.yaml"):
    datas.append(("logging.yaml", "LabGym"))

a = Analysis(
    ['myapp.py'],
    pathex=['pyinstaller'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join('pyinstaller','hooks')],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name=APP_NAME,
    icon=ICON,        # <— macOS wants .icns
    console=False,    # windowed
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
