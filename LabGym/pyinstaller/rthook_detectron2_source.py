# Make imports prefer the on-disk copy of LabGym.detectron2 inside the app bundle,
# so inspect.getsource() and TorchScript JIT can see real .py files.

import sys
import os
from pathlib import Path

def _find_labgym_root_candidates():
    """
    Return candidate directories that might directly contain 'LabGym/'.
    For a macOS .app one-folder build, data files typically land under:
        .../LabGym.app/Contents/MacOS/
    Some projects choose Resources; we’ll check both to be robust.
    """
    cands = []

    if getattr(sys, "frozen", False):
        # In a one-folder .app, sys.executable is:
        #   .../LabGym.app/Contents/MacOS/LabGym
        exe_dir = Path(sys.executable).resolve().parent
        # Most common: datas are placed *next to* the executable
        cands.append(exe_dir)
        # Also try Resources sibling, in case packaging/layout changes later
        res_dir = exe_dir.parent / "Resources"
        cands.append(res_dir)
    else:
        # Dev runs: use current file’s parent
        cands.append(Path(__file__).resolve().parent)

    # De-dup while preserving order
    seen = set()
    uniq = []
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def _prefer_on_disk_labgym():
    for base in _find_labgym_root_candidates():
        labgym_dir = base / "LabGym"
        detectron2_dir = labgym_dir / "detectron2"
        if labgym_dir.is_dir() and detectron2_dir.is_dir() and (labgym_dir / "__init__.py").exists():
            parent = str(base)  # we want 'base' on sys.path so 'import LabGym...' works
            if parent not in sys.path:
                sys.path.insert(0, parent)
            # Once we’ve inserted a good candidate, we’re done.
            return

_prefer_on_disk_labgym()
