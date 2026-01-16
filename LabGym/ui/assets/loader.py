"""
LabGym.ui.assets.loader
"""

from __future__ import annotations

# Standard library imports
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    # Python >= 3.9
    from importlib import resources as ir

except ImportError:
    import importlib_resources as ir  # type: ignore

_PACKAGE = "LabGym.ui.assets.icons"

@contextmanager
def resource_tmp(name: str) -> Iterator[Path]:
    """"
    Yield a real filesystem path to <name>.
    Works even if LabGym is embedded inside a zipped wheel or PyInstaller
    one-file bundle - ir.as_file() copies toa  temp dir and auto-cleans.
    """
    
    ref = ir.files(_PACKAGE) / name
    
    with ir.as_file(ref) as tmp:
        yield tmp


# public helpers
def icon_path(name: str = "labgym.ico") -> Path:
    """
    Returns a pathlib.Path to the icon - for APIs that insist on a path.
    May point to a temp file; valid only inside the context manager!
    """

    if sys.meta_path and any(m.__class__.__name__ == "FrozenImporter" for m in sys.meta_path):
        # PyInstaller one-file: always use the context manager
        return resource_tmp(name).__enter__()  # caller must close

    else:
        # Most installs: direct path is safe
        return ir.files(_PACKAGE).joinpath(name)


def icon_bytes(name: str = "labgym.ico") -> bytes:
    """
    Raw bytes -  nice when the framework can accept a memory buffer.
    """
    return ir.read_binary(_PACKAGE, name)