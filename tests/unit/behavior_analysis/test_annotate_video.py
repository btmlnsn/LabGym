"""
LabGym.tests.unit.behavior_analysis.test_annotate_video

Tests for AnalyzeAnimal.annotate_video() method.
"""

# Standard library imports
import os
from unittest.mock import patch, MagicMock

# Related third party imports
import numpy as np
import pytest
import cv2

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestAnnotateVideo:
    """Tests for annotate_video() method."""
    
    @pytest.fixture
    def analyzer_setup(self, sample_background_frame, sample_video_frames_bgr):
        """Create analyzer with data for annotation."""
        
        analyzer = AnalyzeAnimal()
        analyzer.path_to_video = "test_video.avi"
        analyzer.fps = 30
        analyzer.t = 0.0
        analyzer.duration = 2
        analyzer.length = 15
        analyzer.framewidth = None
        analyzer.background = sample_background_frame
        analyzer.categorize_behavior = True
        
        # Set up tracking data
        analyzer.all_time = [0.0, 0.033, 0.067]
        analyzer.skipped_frames = []
        
        # Create contours and centers
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour = np.array([
            [[int(50 + 15 * np.cos(a)), int(50 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        
        analyzer.animal_contours = {
            0: [contour, contour, contour]
        }
        analyzer.animal_centers = {
            0: [(50, 50), (51, 51), (52, 52)]
        }
        analyzer.event_probability = {
            0: [['walking', 0.9], ['walking', 0.8], ['resting', 0.7]]
        }
        analyzer.all_behavior_parameters = {
            'walking': {'color': ('Walking', '#FF0000')},
            'resting': {'color': ('Resting', '#00FF00')},
        }
        
        return analyzer
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_creates_annotated_video(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should create annotated video file."""
       
       # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:5])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer = analyzer_setup
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=False
        )
        
        # Assert
        mock_writer.write.assert_called()
        mock_writer.release.assert_called_once()
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_saves_trajectory_image(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should save trajectory image."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer = analyzer_setup
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=False
        )
        
        # Assert
        trajectory_file = os.path.join(analyzer.results_path, 'Trajectory.jpg')
        assert os.path.exists(trajectory_file)
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_exports_centers_excel(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should export animal centers to Excel."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer = analyzer_setup
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=False
        )
        
        # Assert
        centers_file = os.path.join(analyzer.results_path, 'all_centers.xlsx')
        assert os.path.exists(centers_file)
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_draws_contours_with_behavior_colors(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should draw contours with behavior-specific colors."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer = analyzer_setup
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=False
        )
        
        # Assert
        # Video writer should be called (contours drawn)
        assert mock_writer.write.called
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_shows_legend_when_enabled(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should show legend when show_legend=True."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer = analyzer_setup
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=True
        )
        
        # Assert
        # Legend should be drawn (video writer called)
        assert mock_writer.write.called
    

    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_handles_skipped_frames(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should handle frames where no animals were detected."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.skipped_frames = [1]  # Frame 1 was skipped
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=['walking', 'resting'],
            show_legend=False
        )
        

        # Assert
        # Should not crash on skipped frames
        assert mock_writer.write.called
    
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_works_without_behavior_categorization(
        self,
        mock_video_capture_class,
        mock_video_writer_class,
        analyzer_setup,
        sample_video_frames_bgr,
        tmp_path
    ):
        """Should work when categorize_behavior=False."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.categorize_behavior = False
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:3])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_writer = MagicMock()
        mock_video_writer_class.return_value = mock_writer
        
        analyzer.results_path = str(tmp_path / "results")
        os.makedirs(analyzer.results_path, exist_ok=True)
        
        # Act
        analyzer.annotate_video(
            ID_colors=[(255, 0, 0)],
            behavior_to_include=[],
            show_legend=False
        )
        

        # Assert
        # Should work without behavior colors
        assert mock_writer.write.called



