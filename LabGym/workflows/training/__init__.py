"""
LabGym.workflows.training

This parent package will group the two independent training workflows:

detector: image-example generation and Mask-RCNN training (detectron2)
categorizer: behavior example generation and network training

Higher-level code can 'import LabGym.workflows.training as wf_train' and then 
reach sub-modules via attributes once they are populated.
"""

from __future__ import annotations

__all__: list[str] = []
