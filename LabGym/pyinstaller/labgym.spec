# LabGym/pyinstaller/labgym.spec  (Python 3.10)

from pathlib import Path
import os

from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE
from PyInstaller.utils.hooks import collect_submodules

# ---- Paths ----
project_root = Path.cwd()
labgym_root = project_root / 'LabGym'
assert labgym_root.exists(), f"Expected {labgym_root} to exist; run PyInstaller from repo root."
pathex = [str(project_root)]

# ---- Entrypoint ----
entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
assert entry_script.exists(), f"Entry script not found: {entry_script}"

# ---- Vendored detectron2 ----
vendored_detectron2 = labgym_root / 'detectron2'
assert vendored_detectron2.exists(), f"Vendored detectron2 not found at: {vendored_detectron2}"

def walk_as_datas(src_dir: Path, dest_prefix: str):
    out = []
    src_dir = src_dir.resolve()
    for root, _, files in os.walk(src_dir):
        root_p = Path(root)
        for fn in files:
            src = root_p / fn
            rel = src.relative_to(src_dir)
            dest = Path(dest_prefix) / rel
            out.append((str(src), str(dest)))
    return out

# ---- Sidecar datas (guaranteed copy next to the .app) ----
datas = []
datas += walk_as_datas(vendored_detectron2, 'LabGym/detectron2')

# Optional logging.yaml
log_yaml = labgym_root / 'logging.yaml'
if log_yaml.exists():
    datas.append((str(log_yaml), 'LabGym'))

# ---- Runtime hook ----
runtime_hook = labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py'
assert runtime_hook.exists(), f"Missing runtime hook: {runtime_hook}"

# ---- Hidden imports ----
hiddenimports = [
    'torch',
    'torchvision',
    'tomli',
]

# ---- Prefer torchvision as .py (helps inspect/JIT sometimes) ----
module_collection_mode = {
    'torchvision': 'py',
}

a = Analysis(
    scripts=[str(entry_script)],
    pathex=pathex,
    binaries=[],
    datas=datas,                    # ensure sidecar copy exists for JIT
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

# ---- ALSO request a copy inside .app/Contents/Resources ----
bundle_resources = [
    (str(vendored_detectron2), 'LabGym/detectron2'),
]
if log_yaml.exists():
    bundle_resources.append((str(log_yaml), 'LabGym'))

app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={'NSHighResolutionCapable': True},
    resources=bundle_resources,     # if honored, places files into Contents/Resources
)
