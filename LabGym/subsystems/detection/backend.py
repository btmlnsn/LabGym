"""
LabGym.subsystems.detection.backend
"""

from LabGym.detectron2 import model_zoo
from LabGym.detectron2.checkpoint import DetectionCheckpointer
from LabGym.detectron2.config import get_cfg
from LabGym.detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from LabGym.detectron2.data.datasets import register_coco_instances
from LabGym.detectron2.engine import DefaultTrainer, DefaultPredictor
from LabGym.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from LabGym.detectron2.modeling import build_model
from LabGym.detectron2.utils.visualizer import Visualizer


__all__ = [
    "model_zoo",
    "DetectionCheckpointer",
    "get_cfg",
    "MetadataCatalog",
    "DatasetCatalog",
    "build_detection_test_loader",
    "register_coco_instances",
    "DefaultTrainer",
    "DefaultPredictor",
    "COCOEvaluator",
    "inference_on_dataset",
    "build_model",
    "Visualizer",
]
