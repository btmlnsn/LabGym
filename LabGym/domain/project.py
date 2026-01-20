"""
Canonical folder locations *with user-override support*.

SAFE TO IMPORT from any layer – no GUI / heavy deps.
"""
from __future__ import annotations

import sys
from pathlib import Path
from LabGym.config import core as _cfg_mod    # avoid name clash


class ProjectPaths:
    """
    Resolution order (highest → lowest):
        1. LABGYM_DETECTORS / LABGYM_MODELS env-vars
        2. values in config.toml / .yaml / .ini / --configfile
        3. hard-wired package defaults  …/LabGym/{detectors,models}
    """

    _pkg_root   = Path(__file__).resolve().parent.parent
    _defaults   = {
        "detectors": _pkg_root / "detectors",
        "models":    _pkg_root / "models",
    }

    # these get populated *after* class definition
    detectors: Path
    models:    Path

    # ---------- helpers ----------

    @staticmethod
    def _safe_cfg() -> dict:
        """
        Run config.get_config() but ignore SystemExit raised by our CLI parser
        when pytest injects flags like -q, -k, -s, etc.
        """
        try:
            return _cfg_mod.get_config("detectors", "models")
        except SystemExit as exc:
            if exc.code and "unrecognized option" in str(exc.code):
                return {}
            raise  # propagate genuine exits

    @classmethod
    def ensure(cls) -> None:
        cls.detectors.mkdir(parents=True, exist_ok=True)
        cls.models.mkdir(parents=True, exist_ok=True)


# ---------- compute final paths now that the class exists ----------
_cfg = ProjectPaths._safe_cfg()
ProjectPaths.detectors = Path(_cfg.get("detectors", ProjectPaths._defaults["detectors"])).expanduser()
ProjectPaths.models    = Path(_cfg.get("models",    ProjectPaths._defaults["models"])).expanduser()

# auto-create folders
ProjectPaths.ensure()



__all__ = ["ProjectPaths"]
