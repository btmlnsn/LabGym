import sys
from pathlib import Path

def _candidate_bases():
    bases = []
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        macos = exe.parent                                   # .../LabGym.app/Contents/MacOS
        bases.append(macos)                                   # inside .app (MacOS)
        bases.append(macos.parent / "Resources")              # inside .app (Resources)

        # sidecar (one-folder layout): dist/LabGym (sibling to .app)
        # .../dist/LabGym.app/Contents/MacOS/LabGym
        # sidecar likely at .../dist/LabGym
        dist_dir = macos.parent.parent.parent                 # .../dist
        bases.append(dist_dir / "LabGym")                     # .../dist/LabGym
    else:
        bases.append(Path(__file__).resolve().parent)
    seen = set()
    uniq = []
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
