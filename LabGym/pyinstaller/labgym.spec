# LabGym/pyinstaller/labgym.spec  (Python 3.10)

from pathlib import Path
import os
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE, COLLECT
from PyInstaller.building.datas import Tree

project_root = Path.cwd()
labgym_root  = project_root / 'LabGym'
assert labgym_root.exists()

entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
assert entry_script.exists()

vendored_detectron2 = labgym_root / 'detectron2'
assert vendored_detectron2.exists()

# guarantee an on-disk copy in the sidecar folder:
datas = []
# Use Tree to preserve the directory structure safely (avoids __init__.py dir/file collisions)
datas += Tree(str(vendored_detectron2), prefix='LabGym/detectron2').toc

log_yaml = labgym_root / 'logging.yaml'
if log_yaml.exists():
    datas.append((str(log_yaml), 'LabGym'))

runtime_hook = labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py'
assert runtime_hook.exists()

hiddenimports = ['torch', 'torchvision', 'tomli']
module_collection_mode = {'torchvision': 'py'}

a = Analysis(
    scripts=[str(entry_script)],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,                # sidecar copy ends up in dist/LabGym/...
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(runtime_hook)],
    excludes=['detectron2'],    # ensure only vendored is used
    noarchive=True,             # keep modules unpacked
    module_collection_mode=module_collection_mode,
)

pyz = PYZ(a.pure)

icon_file = labgym_root / 'pyinstaller' / 'sunflower.png'
icon_arg  = str(icon_file) if icon_file.exists() else None

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='LabGym',
    console=False,
    icon=icon_arg,
)

# 1) build a **one-dir sidecar** that contains our datas as real files:
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='LabGym_sidecar'   # → dist/LabGym/ (DIRECTORY) guaranteed
)

# 2) also emit the .app bundle (UI entry). resources here are optional; the hook can load from sidecar.
app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={'NSHighResolutionCapable': True},
    # If you *also* want a copy inside the .app, uncomment the line below:
    # resources=[(str(vendored_detectron2), 'LabGym/detectron2')]  # Contents/Resources/...
)
