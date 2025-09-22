# labgym.spec

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.datastruct import Tree

# Use CWD as repo root (Actions runs pyinstaller from repo root)
project_root = Path.cwd()
labgym_root = project_root / 'LabGym'
assert labgym_root.exists(), f"Expected {labgym_root} to exist; run PyInstaller from repo root."

pathex = [str(project_root)]

# ABSOLUTE path to the entry script (prevents the doubled path issue)
entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
assert entry_script.exists(), f"Entry script not found: {entry_script}"

# Ship vendored detectron2 as *real .py files on disk* inside the .app
vendored_detectron2 = labgym_root / 'detectron2'
assert vendored_detectron2.exists(), f"Vendored detectron2 not found at: {vendored_detectron2}"

datas = []
# Copy the entire tree into Contents/Resources/LabGym/detectron2/...
datas += Tree(str(vendored_detectron2), prefix='LabGym/detectron2')
# also include logging.yaml (adjust if your path differs)
log_yaml = labgym_root / 'logging.yaml'
if log_yaml.exists():
    datas += [(str(log_yaml), 'LabGym')]

# Optional icon (if present)
icon_file = labgym_root / 'pyinstaller' / 'sunflower.png'
icon_arg = str(icon_file) if icon_file.exists() else None

# Keep runtime hook that ensures imports prefer on-disk sources
runtime_hooks = [str(labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py')]

# Min hidden imports (avoid importing vendored package at spec time)
hiddenimports = [
    'torch',
    'torchvision',
    'tomli',
] + collect_submodules('torch')[:0]  # no-op; placeholder if you later need it

# Prefer shipping torchvision sources as .py too (helps with inspect/JIT edge cases)
module_collection_mode = {
    'torchvision': 'py',
    # if you *also* rely on external 'detectron2' anywhere, uncomment:
    # 'detectron2': 'py',
}

a = Analysis(
    scripts=[str(entry_script)],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=True,  # don't zip into PYZ: keep modules unpacked
    module_collection_mode=module_collection_mode,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='LabGym',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon_arg,
)

app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={'NSHighResolutionCapable': True},
)
