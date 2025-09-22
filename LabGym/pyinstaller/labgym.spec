# labgym.spec
# Python 3.10 only

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

spec_file = Path(os.path.abspath(__file__))
project_root = spec_file.parent.parent.parent  # .../LabGym/pyinstaller -> repo root
pathex = [str(project_root)]

# --- Force detectron2 (vendored) to be shipped as real .py files on disk ---
# We do two things:
#   (A) Tell PyInstaller to collect the package in "py" mode (no frozen loader).
#   (B) Also include the whole tree as data, so files physically exist at runtime.
module_collection_mode = {
    # Our vendored package
    'LabGym.detectron2': 'py',
    # If any external 'detectron2' sneaks in, also force py files
    'detectron2': 'py',
    # (Optional) torchvision source kept on disk too, avoids rare JIT/inspect issues
    'torchvision': 'py',
}

# Collect additional bits PyInstaller could miss
d2_datas, d2_bins, d2_hidden = collect_all('LabGym.detectron2')

# Explicitly include the vendored source tree as data, to guarantee file presence.
# This mirrors what `--collect-all LabGym.detectron2` does in John's script.
vendored_detectron2_src = project_root / 'LabGym' / 'detectron2'
if not vendored_detectron2_src.exists():
    raise SystemExit(f"Vendored detectron2 not found at: {vendored_detectron2_src}")

extra_datas = [
    # (src, dest) -> lands in .app/Contents/Resources/LabGym/detectron2/...
    (str(vendored_detectron2_src), 'LabGym/detectron2'),
    # ensure logging.yaml (adjust path if you keep it elsewhere)
    (str(project_root / 'LabGym' / 'logging.yaml'), 'LabGym'),
]

# App icon (optional, adjust if you have one)
icon_file = str((project_root / 'LabGym' / 'pyinstaller' / 'sunflower.png'))

# Runtime hook guarantees sys.path prefers the on-disk copy of LabGym.detectron2
runtime_hooks = [str(project_root / 'LabGym' / 'pyinstaller' / 'rthook_detectron2_source.py')]

a = Analysis(
    ['LabGym/pyinstaller/myapp.py'],   # same entrypoint as John’s script
    pathex=pathex,
    binaries=d2_bins,
    datas=extra_datas + d2_datas,
    hiddenimports=[
        # torch/vision sometimes dynamic import bits
        'torch',
        'torchvision',
        # defend against dynamic imports inside both detectron2 names
        'LabGym.detectron2',
        'detectron2',
        # tomli still used by some stacks on py3.10
        'tomli',
    ] + d2_hidden,
    hookspath=[],
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=True,                    # keep modules unpacked (no PYZ zip)
    module_collection_mode=module_collection_mode,
)

pyz = PYZ(a.pure)  # still created but unused for our forced 'py' packages

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
    console=False,        # windowed app
    icon=icon_file if os.path.exists(icon_file) else None,
)

# On macOS, produce the .app bundle
app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_file if os.path.exists(icon_file) else None,
    info_plist={
        'NSHighResolutionCapable': True,
    },
)
