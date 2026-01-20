"""
LabGym.subsystems.shared.events

Neutral facade that exposes
- evaluate_detector (from detection subsystem)
- AnalyzeAnimal and AnalyzeAnimalDetector (from categorization subsystems)
"""

from __future__ import annotations

# Standard library imports
from types import ModuleType
from typing import Any, Callable
import importlib

# Local application imports
from LabGym.subsystems.detection.api import evaluate as evaluate_detector


# lazy import to avoid circular imports
def _lazy(name: str) -> Callable[..., Any]:
    """Returns a callable that, on first invocation, resolves the real symbol inside
    'LabGym.subsystems.categorization.predict' and then replaces itself in this
    module's globals."""

    def _wrapper(*args, **kwargs):
        real_mod: ModuleType = importlib.import_module(
            "LabGym.subsusytems.categorization.predict"
        )
        
        real_obj = getattr(real_mod, name)
        globals()[name] = real_obj

        return real_obj(*args, **kwargs)

    _wrapper.__name__ = name
    _wrapper.__doc__ = f"Lazy proxy for categorization.{name}"
    
    return _wrapper


AnalyzeAnimal = _lazy("AnalyzeAnimal")
AnalyzeAnimalDetector = _lazy("AnalyzeAnimalDetector")


__all__ = ["evaluate_detector", "AnalyzeAnimal", "AnalyzeAnimalDetector"]
