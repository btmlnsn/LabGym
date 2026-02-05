"""
/LabGym.tests.unit.test_detector

Unit tests for LabGym.detector (Detectron2 wrapper)
"""


# Standard library imports
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Related third party imports
import pytest
import torch

# Local application imports
from LabGym.detector import Detector


# Initialization
class TestDetectorInit:
    def test_init_cpu_when_no_cuda(self):
        with patch("LabGym.detector.torch.cuda.is_available", return_value=False):
            d = Detector()
            assert d.device == "cpu"
            assert d.animal_mapping is None
            assert d.current_detector is None

    def test_init_cuda_when_available(self):
        with patch("LabGym.detector.torch.cuda.is_available", return_value=True):
            d = Detector()
            assert d.device == "cuda"
            assert d.animal_mapping is None
            assert d.current_detector is None


# Training (mocked)
class TestDetectorTrainUnit:
    @patch("LabGym.detector.DefaultTrainer")
    @patch("LabGym.detector.DefaultPredictor")
    @patch("LabGym.detector.DetectionCheckpointer")
    @patch("LabGym.detector.get_cfg")
    @patch("LabGym.detector.model_zoo")
    @patch("LabGym.detector.register_coco_instances")
    @patch("LabGym.detector.DatasetCatalog")
    @patch("LabGym.detector.MetadataCatalog")
    def test_train_registers_dataset(
        self,
        mock_meta,
        mock_ds,
        mock_register,
        mock_zoo,
        mock_get_cfg,
        mock_ckpt,
        mock_pred,
        mock_trainer,
        minimal_coco_annotation,
        minimal_training_images,
        tmp_path,
    ):
        mock_ds.list.return_value = []
        mock_meta.get.return_value = MagicMock(thing_classes=["__background__", "animal"])
        mock_cfg = MagicMock()
        mock_cfg.OUTPUT_DIR = str(tmp_path / "out")
        mock_cfg.dump.return_value = "config"
        mock_get_cfg.return_value = mock_cfg
        mock_zoo.get_config_file.return_value = "config.yaml"
        mock_zoo.get_checkpoint_url.return_value = "checkpoint.pth"
        mock_trainer.return_value = MagicMock()
        mock_pred.return_value = MagicMock(model=MagicMock())

        out = str(tmp_path / "detector")
        Detector().train(
            minimal_coco_annotation,
            minimal_training_images,
            out,
            iteration_num=10,
            inference_size=100,
        )
        mock_register.assert_called_once()
        assert mock_register.call_args[0][0] == "LabGym_detector_train"

    @patch("LabGym.detector.DefaultTrainer")
    @patch("LabGym.detector.DefaultPredictor")
    @patch("LabGym.detector.DetectionCheckpointer")
    @patch("LabGym.detector.get_cfg")
    @patch("LabGym.detector.model_zoo")
    @patch("LabGym.detector.register_coco_instances")
    @patch("LabGym.detector.DatasetCatalog")
    @patch("LabGym.detector.MetadataCatalog")
    def test_train_removes_existing_dataset(
        self,
        mock_meta,
        mock_ds,
        mock_register,
        mock_zoo,
        mock_get_cfg,
        mock_ckpt,
        mock_pred,
        mock_trainer,
        minimal_coco_annotation,
        minimal_training_images,
        tmp_path,
    ):
        mock_ds.list.return_value = ["LabGym_detector_train"]
        mock_meta.get.return_value = MagicMock(thing_classes=["__background__", "animal"])
        mock_cfg = MagicMock()
        mock_cfg.OUTPUT_DIR = str(tmp_path / "out")
        mock_cfg.dump.return_value = "config"
        mock_get_cfg.return_value = mock_cfg
        mock_zoo.get_config_file.return_value = "config.yaml"
        mock_zoo.get_checkpoint_url.return_value = "checkpoint.pth"
        mock_trainer.return_value = MagicMock()
        mock_pred.return_value = MagicMock(model=MagicMock())

        out = str(tmp_path / "detector")
        Detector().train(
            minimal_coco_annotation,
            minimal_training_images,
            out,
            iteration_num=10,
            inference_size=100,
        )
        mock_ds.remove.assert_called_once_with("LabGym_detector_train")
        mock_meta.remove.assert_called_once_with("LabGym_detector_train")

    @patch("LabGym.detector.DefaultTrainer")
    @patch("LabGym.detector.DefaultPredictor")
    @patch("LabGym.detector.DetectionCheckpointer")
    @patch("LabGym.detector.get_cfg")
    @patch("LabGym.detector.model_zoo")
    @patch("LabGym.detector.register_coco_instances")
    @patch("LabGym.detector.DatasetCatalog")
    @patch("LabGym.detector.MetadataCatalog")
    def test_train_writes_model_parameters_from_annotation(
        self,
        mock_meta,
        mock_ds,
        mock_register,
        mock_zoo,
        mock_get_cfg,
        mock_ckpt,
        mock_pred,
        mock_trainer,
        minimal_coco_annotation,
        minimal_training_images,
        tmp_path,
    ):
        """Written model_parameters.txt has animal_names from categories id>0, animal_mapping, inferencing_framesize."""
        mock_ds.list.return_value = []
        mock_meta.get.return_value = MagicMock(thing_classes=["__background__", "animal"])
        out_dir = tmp_path / "detector"
        mock_cfg = MagicMock()
        mock_cfg.OUTPUT_DIR = str(out_dir)
        mock_cfg.dump.return_value = "config_yaml_content"
        mock_get_cfg.return_value = mock_cfg
        mock_zoo.get_config_file.return_value = "config.yaml"
        mock_zoo.get_checkpoint_url.return_value = "checkpoint.pth"
        mock_trainer.return_value = MagicMock()
        mock_pred.return_value = MagicMock(model=MagicMock())

        Detector().train(
            minimal_coco_annotation,
            minimal_training_images,
            str(out_dir),
            iteration_num=10,
            inference_size=100,
        )

        params_path = out_dir / "model_parameters.txt"
        assert params_path.exists()
        with open(params_path) as f:
            params = json.load(f)
        assert params["animal_names"] == ["animal"]
        assert params["inferencing_framesize"] == 100
        assert params["animal_mapping"] == {"0": "__background__", "1": "animal"}


# Testing
class TestDetectorTestUnit:
    @patch("LabGym.detector.inference_on_dataset")
    @patch("LabGym.detector.build_detection_test_loader")
    @patch("LabGym.detector.COCOEvaluator")
    @patch("LabGym.detector.cv2.imwrite")
    @patch("LabGym.detector.cv2.imread")
    @patch("LabGym.detector.DefaultPredictor")
    @patch("LabGym.detector.get_cfg")
    @patch("LabGym.detector.DatasetCatalog")
    @patch("LabGym.detector.register_coco_instances")
    def test_test_registers_dataset_and_reads_from_detector_dir(
        self,
        mock_register,
        mock_ds,
        mock_get_cfg,
        mock_predictor,
        mock_imread,
        mock_imwrite,
        mock_eval_cls,
        mock_loader,
        mock_inference_on,
        tmp_path,
    ):
        """test() registers 'LabGym_detector_test' and reads model_parameters from path_to_detector."""
        mock_ds.list.return_value = []
        mock_ds.get.return_value = [{"file_name": str(tmp_path / "img.jpg")}]
        mock_imread.return_value = __import__("numpy").zeros((100, 100, 3), dtype="uint8")

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        (detector_dir / "model_parameters.txt").write_text(
            json.dumps({
                "animal_names": ["animal"],
                "inferencing_framesize": 100,
            })
        )
        (detector_dir / "config.yaml").write_text("config")
        (detector_dir / "model_final.pth").write_bytes(b"x")

        mock_cfg = MagicMock()
        mock_get_cfg.return_value = mock_cfg
        mock_predictor.return_value = MagicMock()
        mock_eval_inst = MagicMock()
        mock_eval_inst._results = {"bbox": {"AP": 0.5}}
        mock_eval_cls.return_value = mock_eval_inst
        mock_loader.return_value = []

        Detector().test(
            str(tmp_path / "ann.json"),
            str(tmp_path / "images"),
            str(detector_dir),
            str(tmp_path / "out"),
        )

        mock_register.assert_called_once()
        assert mock_register.call_args[0][0] == "LabGym_detector_test"
        mock_ds.get.assert_called_with("LabGym_detector_test")


# Loading
class TestDetectorLoadUnit:
    @patch("LabGym.detector.build_model")
    @patch("LabGym.detector.DetectionCheckpointer")
    @patch("LabGym.detector.get_cfg")
    def test_load_sets_animal_mapping_and_current_detector(
        self, mock_get_cfg, mock_ckpt_cls, mock_build, tmp_path
    ):
        mock_get_cfg.return_value = MagicMock()
        mock_ckpt_cls.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        (detector_dir / "config.yaml").write_text("dummy")
        (detector_dir / "model_final.pth").write_bytes(b"dummy")
        params = {
            "animal_names": ["animal"],
            "animal_mapping": {0: "__background__", 1: "animal"},
            "inferencing_framesize": 100,
        }
        (detector_dir / "model_parameters.txt").write_text(json.dumps(params))

        d = Detector()
        d.load(str(detector_dir), ["animal"])

        assert d.animal_mapping == {"0": "__background__", "1": "animal"}
        assert d.current_detector is not None

    def test_load_raises_file_not_found_when_dir_missing(self):
        d = Detector()
        with pytest.raises(FileNotFoundError):
            d.load("/nonexistent/detector/path", ["animal"])

    def test_load_raises_file_not_found_when_model_parameters_missing(self, tmp_path):
        (tmp_path / "empty").mkdir()
        d = Detector()
        with pytest.raises(FileNotFoundError):
            d.load(str(tmp_path / "empty"), ["animal"])

    def test_load_raises_json_decode_error_when_model_parameters_invalid(self, tmp_path):
        detector_dir = tmp_path / "detector"
        detector_dir.mkdir()
        (detector_dir / "config.yaml").write_text("dummy")
        (detector_dir / "model_final.pth").write_bytes(b"dummy")
        (detector_dir / "model_parameters.txt").write_text("not valid json")

        d = Detector()
        with pytest.raises(json.JSONDecodeError):
            d.load(str(detector_dir), ["animal"])


# Inference
class TestDetectorInference:
    def test_inference_calls_model_returns_outputs(self):
        d = Detector()
        d.current_detector = MagicMock()
        out = MagicMock()
        d.current_detector.return_value = out
        inp = MagicMock()
        assert d.inference(inp) is out
        d.current_detector.assert_called_once_with(inp)

    def test_inference_fails_if_not_loaded(self):
        d = Detector()
        d.current_detector = None
        with pytest.raises(TypeError):
            d.inference(MagicMock())

    @patch("LabGym.detector.torch.no_grad")
    def test_inference_uses_no_grad(self, mock_no_grad):
        mock_no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=None)
        d = Detector()
        d.current_detector = MagicMock(return_value=MagicMock())
        d.inference(MagicMock())
        mock_no_grad.assert_called_once()

