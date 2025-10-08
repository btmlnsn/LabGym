from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE, BUNDLE, COLLECT

project_root = Path.cwd()
labgym_root  = project_root / 'LabGym'
entry_script = labgym_root / 'pyinstaller' / 'myapp.py'
vendored_detectron2 = labgym_root / 'detectron2'
icon_file = labgym_root / 'assets' / 'icons' / 'labgym.icns'
icon_arg  = str(icon_file) if icon_file.exists() else None

datas = [(str(vendored_detectron2), 'LabGym/detectron2')]
log_yaml = labgym_root / 'logging.yaml'
if log_yaml.exists():
    datas.append((str(log_yaml), 'LabGym'))

runtime_hook = labgym_root / 'pyinstaller' / 'rthook_detectron2_source.py'

a = Analysis(
    scripts=[str(entry_script)],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=['torch', 'torchvision', 'tomli'],
    hookspath=[],
    runtime_hooks=[str(runtime_hook)],
    excludes=['detectron2'],
    noarchive=True,
    module_collection_mode={'torchvision': 'py'},
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name='LabGym',
    console=False,
    icon=icon_arg,
)

# Sidecar folder (real files)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name='LabGym_sidecar'
)

# .app bundle
app = BUNDLE(
    exe,
    name='LabGym.app',
    bundle_identifier='yelab.LabGym',
    icon=icon_arg,
    info_plist={
        'NSHighResolutionCapable': True,
        'CFBundleIconFile': 'labgym.icns',
        'CFBundleName': 'LabGym',
        'CFBundleDisplayName': 'LabGym',
    },
    resources=[(str(icon_file), '')],  # ensure ICNS in Contents/Resources
)
