"""
LabGym.tests.unit.analysis.test_prepare_analysis

Tests for AnalyzeAnimal.prepare_analysis() method in analyzebehavior.py
"""

# Standard library imports
import os
from unittest.mock import patch, MagicMock

# Related third party imports
import numpy as np
import pytest

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestPrepareAnalysis:
    """Tests for prepare_analysis() method."""
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_initializes_video_properties(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should extract FPS and frame dimensions from video."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0  # FPS
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, test_frame)
        mock_video_capture_class.return_value = mock_capture
        
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            categorize_behavior=False,
        )
        
        # Assert
        assert analyzer.fps == 30
        assert analyzer.path_to_video == synthetic_video_file
        assert os.path.exists(analyzer.results_path)
        assert analyzer.basename == "test_video.avi"
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_creates_results_directory(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should create results directory if it doesn't exist."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        results_dir = tmp_path / "test_results"
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(results_dir),
            animal_number=1,
            categorize_behavior=False,
        )
        
        # Assert
        assert results_dir.exists()
        assert analyzer.results_path == str(results_dir / "test_video")
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_calls_estimate_constants(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should call estimate_constants to get background and animal area."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            delta=1.2,
            categorize_behavior=False,
        )
        
        # Assert
        mock_estimate_constants.assert_called_once()
        assert analyzer.background is not None
        assert analyzer.background_low is not None
        assert analyzer.background_high is not None
        assert analyzer.animal_area == 700
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_calculates_kernel_size(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should calculate kernel size based on framesize/animal_number."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        # 100x100 frame, 1 animal = 100 pixels per animal
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            categorize_behavior=False,
        )
        
        # Assert
        # framesize = min(100, 100) = 100
        # framesize/animal_number = 100/1 = 100
        # 100 < 250, so kernel should be 3
        assert analyzer.kernel == 3
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_initializes_behavior_parameters_when_categorizing(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result,
        sample_behavior_names_and_colors
    ):
        """Should initialize behavior parameters when categorize_behavior=True."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            categorize_behavior=True,
            names_and_colors=sample_behavior_names_and_colors,
        )
        
        # Assert
        assert 'walking' in analyzer.all_behavior_parameters
        assert 'resting' in analyzer.all_behavior_parameters
        assert analyzer.all_behavior_parameters['walking']['color'] == sample_behavior_names_and_colors['walking']
        assert 'probability' in analyzer.all_behavior_parameters['walking']
        assert 'count' in analyzer.all_behavior_parameters['walking']
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_initializes_tracking_structures(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should initialize tracking data structures for each animal."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        animal_number = 2
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=animal_number,
            categorize_behavior=False,
        )
        
        # Assert
        assert len(analyzer.animal_contours) == animal_number
        assert len(analyzer.animal_centers) == animal_number
        assert len(analyzer.animal_heights) == animal_number
        assert 0 in analyzer.animal_contours
        assert 1 in analyzer.animal_contours
        assert analyzer.animal_existingcenters[0] == (-10000, -10000)
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_resizes_frame_when_framewidth_specified(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should resize frame when framewidth is specified."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        original_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mock_capture.read.return_value = (True, original_frame)
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        framewidth = 100
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            framewidth=framewidth,
            categorize_behavior=False,
        )
        
        # Assert
        assert analyzer.framewidth == framewidth
        assert analyzer.frameheight == 100  # Maintains aspect ratio
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_logs_preparation_steps(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should log preparation steps."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            categorize_behavior=False,
        )
        
        # Assert
        assert len(analyzer.log) > 0
        assert 'Preparation started' in analyzer.log[0]
        assert 'Preparation completed' in analyzer.log[-1]
    
    @patch('LabGym.analyzebehavior.estimate_constants')
    @patch('cv2.VideoCapture')
    def test_calculates_total_framecount(
        self,
        mock_video_capture_class,
        mock_estimate_constants,
        synthetic_video_file,
        tmp_path,
        mock_estimate_constants_result
    ):
        """Should calculate total analysis framecount based on duration."""
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0  # 30 fps
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_video_capture_class.return_value = mock_capture
        mock_estimate_constants.return_value = mock_estimate_constants_result
        
        analyzer = AnalyzeAnimal()
        duration = 5  # seconds
        
        # Act
        analyzer.prepare_analysis(
            path_to_video=synthetic_video_file,
            results_path=str(tmp_path),
            animal_number=1,
            duration=duration,
            categorize_behavior=False,
        )
        
        # Assert
        # duration * fps + 1 = 5 * 30 + 1 = 151
        assert analyzer.total_analysis_framecount == 151