# rthook_detectron2_source.py
# Make sure imports resolve to the on-disk copy of LabGym.detectron2
# instead of frozen modules, so inspect.getsource() works.

import os
import sys
from pathlib import Path

def _insert_vendor_parent_on_sys_path():
    # In a PyInstaller macOS .app:
    #   This file ends up in .../Contents/Resources/
    #   Our data tree is .../Contents/Resources/LabGym/detectron2
    base = None

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # onefile / MEIPASS case (not used here, but safe)
        base = Path(sys._MEIPASS)
    elif getattr(sys, 'frozen', False):
        # onefolder .app: runtime hook is in Contents/Resources
        # __file__ -> .../Contents/Resources/rthook_detectron2_source.py
        base = Path(__file__).resolve().parent
    else:
        # dev runs
        base = Path(__file__).resolve().parent

    # We want to add the parent of 'LabGym' so that 'import LabGym.detectron2' finds the on-disk package
    labgym_dir = base / 'LabGym'
    detectron2_dir = labgym_dir / 'detectron2'

    if (labgym_dir / '__init__.py').exists() and detectron2_dir.exists():
        parent = str(labgym_dir.parent)
        # Put it at the very front to beat the FrozenImporter
        if parent not in sys.path:
            sys.path.insert(0, parent)

_insert_vendor_parent_on_sys_path()
