# Ensure the on-disk LabGym/ (and top-level detectron2/, if bundled) is importable at runtime.
import os, sys
from pathlib import Path

meipass = getattr(sys, "_MEIPASS", None) or os.environ.get("_MEIPASS")
if not meipass:
    raise SystemExit(0)  # not frozen

root = Path(meipass)

labgym_pkg = root / "LabGym"
if labgym_pkg.exists():
    sys.path.insert(0, str(labgym_pkg))

d2_top = root / "detectron2"
if d2_top.exists():
    sys.path.insert(0, str(d2_top))
