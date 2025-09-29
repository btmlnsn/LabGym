# LabGym/pyinstaller/rthook_detectron2_source.py

import sys
from pathlib import Path

def _candidate_bases():
    bases = []
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        macos = exe.parent                                   # .../LabGym.app/Contents/MacOS
        bases.append(macos)                                  # inside .app (MacOS)
        bases.append(macos.parent / "Resources")             # inside .app (Resources)

        # dist/
        dist_dir = macos.parent.parent.parent                # .../dist

        # PyInstaller 6.10 one-dir sidecar puts real files under _internal
        sidecar_root = dist_dir / "LabGym_sidecar"
        bases.append(sidecar_root / "_internal")             # preferred: dist/LabGym_sidecar/_internal
        bases.append(sidecar_root)                           # fallback:  dist/LabGym_sidecar
    else:
        bases.append(Path(__file__).resolve().parent)
    # de-dupe while preserving order
    seen, uniq = set(), []
    for b in bases:
        if b not in seen:
            uniq.append(b)
            seen.add(b)
    return uniq

def _prefer_on_disk_labgym():
    for base in _candidate_bases():
        labgym = base / "LabGym"
        d2 = labgym / "detectron2"
        if labgym.is_dir() and d2.is_dir() and (labgym / "__init__.py").exists():
            p = str(base)   # put the directory that contains 'LabGym' on sys.path
            if p not in sys.path:
                sys.path.insert(0, p)
            return

_prefer_on_disk_labgym()
