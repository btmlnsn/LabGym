# LabGym/pyinstaller/labgym.spec
# Python 3.10 only

import os
from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE
from PyInstaller.utils.hooks import collect_submodules

# Use CWD as repo root (Actions invokes pyinstaller from repo root)
project_root = Path.cwd()
labgym_root = project_root / 'LabGym'
assert labgym_root.exists(), f"Expected {labgym_root} to exist; run PyInstaller from repo root."

pathex = [str(project_root)]

# Absolute entry script path (prevents doubled path issue)
entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
assert entry_script.exists(), f"Entry script not found: {entry_script}"

# --- Ship vendored detectron2 as real .py files on disk ---
vendored_detectron2 = labgym_root / 'detectron2'
assert vendored_detectron2.exists(), f"Vendored detectron2 not found at: {vendored_detectron2}"

def walk_as_datas(src_dir: Path, dest_prefix: str):
    """Return list of (src, dest) tuples for all files under src_dir."""
    out = []
    src_dir = src_dir.resolve()
    for root, _, files in os.walk(src_dir):
        root_p = Path(root)
        for fn in files:
            src = root_p / fn
            rel = src.relative_to(src_dir)  # path inside the package
            dest = Path(dest_prefix) / rel
            out.append((str(src), str(dest)))
    return out

datas = []
# copy all vendored detectron2 files to Resources/LabGym/detectron2/...
datas += walk_as_datas(vendored_detectron2, 'LabGym/detectron2')

# include logging.yaml if present (adjust if your path differs)
log_yaml = labgym_root / 'logging.yaml'
if log_yaml.exists():
    datas.append((str(log_yaml), 'LabGym'))

# Optional app icon
icon_file = labgym_root / 'pyinstaller' / 'sunflower.png'
icon_arg = str(icon_file) if icon_file.exists() else None

# Runtime hook ensures imports prefer on-disk sources, not FrozenImporter
runtime_hooks = [str(labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py')]
assert Path(runtime_hooks[0]).exists(), f"Missing runtime hook: {runtime_hooks[0]}"

# Keep hiddenimports minimal; avoid importing vendored detectron2 at spec time
hiddenimports = [
    'torch',
    'torchvision',
    'tomli',
]

# Prefer torchvision as .py too (avoids occasional inspect/JIT edge cases)
module_collection_mode = {
    'torchvision': 'py',
    # If external 'detectron2' ever appears in deps, you can also force:
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
    noarchive=True,                # keep modules unpacked (no PYZ zip archive)
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
    console=False,                 # windowed app
    icon=icon_arg,
)

app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={'NSHighResolutionCapable': True},
)
