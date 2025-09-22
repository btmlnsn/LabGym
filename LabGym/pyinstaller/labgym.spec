# LabGym/pyinstaller/labgym.spec  (Python 3.10)

from pathlib import Path
import os

from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE
from PyInstaller.utils.hooks import collect_submodules

# ---- Path setup (PyInstaller runs spec without __file__) ----
project_root = Path.cwd()
labgym_root = project_root / 'LabGym'
assert labgym_root.exists(), f"Expected {labgym_root} to exist; run PyInstaller from repo root."
pathex = [str(project_root)]

# ---- Entry script (absolute) ----
entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
assert entry_script.exists(), f"Entry script not found: {entry_script}"

# ---- Vendored detectron2 paths ----
vendored_detectron2 = labgym_root / 'detectron2'
assert vendored_detectron2.exists(), f"Vendored detectron2 not found at: {vendored_detectron2}"

# ---- Optional logging.yaml if you ship it ----
log_yaml = labgym_root / 'logging.yaml'
has_log_yaml = log_yaml.exists()

# ---- Runtime hook to prefer on-disk sources (we updated this earlier) ----
runtime_hook = labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py'
assert runtime_hook.exists(), f"Missing runtime hook: {runtime_hook}"

# ---- Hidden imports kept minimal (avoid importing vendored pkg at spec time) ----
hiddenimports = [
    'torch',
    'torchvision',
    'tomli',
]

# ---- Prefer torchvision as .py too (helps inspect/JIT sometimes) ----
module_collection_mode = {
    'torchvision': 'py',
    # If any external 'detectron2' sneaks in via deps, you can also force:
    # 'detectron2': 'py',
}

# ---- Build graph ----
a = Analysis(
    scripts=[str(entry_script)],
    pathex=pathex,
    binaries=[],
    datas=[],                       # <- leave empty here; we’ll place files via BUNDLE.resources
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(runtime_hook)],
    excludes=['detectron2'],        # ensure only vendored is used
    noarchive=True,                 # keep modules unpacked (helps inspect.getsource)
    module_collection_mode=module_collection_mode,
)

pyz = PYZ(a.pure)

icon_file = labgym_root / 'pyinstaller' / 'sunflower.png'
icon_arg = str(icon_file) if icon_file.exists() else None

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

# ---- Put vendored sources *inside* .app/Contents/Resources ----
bundle_resources = [
    # Copy the entire vendored detectron2 tree into Resources/LabGym/detectron2
    (str(vendored_detectron2), 'LabGym/detectron2'),
]
if has_log_yaml:
    bundle_resources.append((str(log_yaml), 'LabGym'))

app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={'NSHighResolutionCapable': True},
    resources=bundle_resources,     # <—— KEY: ensure physical files inside Resources
)
