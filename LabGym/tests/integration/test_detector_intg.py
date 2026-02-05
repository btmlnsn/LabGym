"""
/LabGym.tests.integration.test_detector_intg

Integration tests for LabGym.detector (real training, slow)
"""

"""
Integration tests for LabGym.detector (real training, slow).
"""

# Standard library imports
import json
from pathlib import Path

# Related third party imports
import pytest
import torch

# Local application imports
from LabGym.detector import Detector


pytestmark = [
    pytest.mark.slow, 
    pytest.mark.integration,
    
    
    # PyTorch emits a deprecation warning when the bundled Detectron2 code calls
    # torch.meshgrid() without the new `indexing` argument. That call is in
    # third-party code we don't maintain; the warning is not a bug in our tests or logic.
    
    # It is filtered here so that test output stays clean; if PyTorch later
    # enforces the change, any break will surface as a real runtime failure
    pytest.mark.filterwarnings("ignore:torch.meshgrid.*indexing:UserWarning"),
]


def test_train_creates_model_parameters_and_config(
    minimal_coco_annotation,
    minimal_training_images,
    tmp_path,
):
    out = str(tmp_path / "detector")
    Detector().train(
        minimal_coco_annotation,
        minimal_training_images,
        out,
        iteration_num=10,
        inference_size=100,
    )
    base = Path(out)
    assert (base / "model_parameters.txt").exists()
    assert (base / "config.yaml").exists()


def test_train_saves_valid_model_parameters(
    minimal_coco_annotation,
    minimal_training_images,
    tmp_path,
):
    out = str(tmp_path / "detector")
    Detector().train(
        minimal_coco_annotation,
        minimal_training_images,
        out,
        iteration_num=10,
        inference_size=100,
    )
    with open(Path(out) / "model_parameters.txt") as f:
        params = json.load(f)
    assert "animal_names" in params
    assert "animal_mapping" in params
    assert params["inferencing_framesize"] == 100
    assert "animal" in params["animal_names"]


def test_load_after_train(
    minimal_coco_annotation,
    minimal_training_images,
    tmp_path,
):
    out = str(tmp_path / "detector")
    Detector().train(
        minimal_coco_annotation,
        minimal_training_images,
        out,
        iteration_num=10,
        inference_size=100,
    )
    d = Detector()
    d.load(out, ["animal"])
    assert d.animal_mapping is not None
    assert d.current_detector is not None

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA")
def test_train_uses_cuda(
    minimal_coco_annotation,
    minimal_training_images,
    tmp_path,
):
    d = Detector()
    assert d.device == "cuda"
    out = str(tmp_path / "detector_gpu")
    d.train(
        minimal_coco_annotation,
        minimal_training_images,
        out,
        iteration_num=10,
        inference_size=100,
    )

