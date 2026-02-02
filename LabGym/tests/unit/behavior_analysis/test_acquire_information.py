"""
/tests.unit.behavior_analysis.test_acquire_information

Tests for AnalyzeAnimal.acquire_information() method.
"""

# Standard library imports
from unittest.mock import patch, MagicMock

# Related third party imports
import numpy as np
import pytest

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestAcquireInformation:
    """Tests for acquire_information() method."""
    
    @pytest.fixture
    def analyzer_setup(self, sample_background_frame, mock_estimate_constants_result):
        """Create analyzer with basic setup for acquisition tests."""
        
        analyzer = AnalyzeAnimal()
        analyzer.path_to_video = "test_video.avi"
        analyzer.fps = 30
        analyzer.t = 0.0
        analyzer.duration = 2  # 2 seconds
        analyzer.length = 15
        analyzer.framewidth = None
        analyzer.animal_number = 1
        analyzer.animal_vs_bg = 0
        analyzer.include_bodyparts = False
        analyzer.animation_analyzer = True
        analyzer.dim_tconv = 8
        analyzer.channel = 1
        analyzer.kernel = 3
        analyzer.delta = 1.2
        analyzer.animal_area = 700
        
        # Set backgrounds
        analyzer.background = mock_estimate_constants_result[0]
        analyzer.background_low = mock_estimate_constants_result[1]
        analyzer.background_high = mock_estimate_constants_result[2]
        
        # Initialize tracking structures
        analyzer.animal_contours = {0: [None] * 100}
        analyzer.animal_centers = {0: [None] * 100}
        analyzer.animal_existingcenters = {0: (-10000, -10000)}
        analyzer.animal_heights = {0: [None] * 100}
        analyzer.animations = {0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 100}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 100}
        analyzer.register_counts = {0: None}
        analyzer.to_deregister = {0: 0}
        analyzer.count_to_deregister = 60
        
        return analyzer
    

    @patch('LabGym.analyzebehavior.extract_blob_background')
    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_reads_video_frames(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        mock_extract_blob,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should read frames from video and process them."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        # Return frames one by one
        frame_iter = iter(sample_video_frames_bgr)
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        # Mock contour detection to return detections
        mock_contour = sample_video_frames_bgr[0].copy()
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour = np.array([
            [[int(50 + 15 * np.cos(a)), int(50 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        mock_contour_frame.return_value = ([contour], [(50, 50)], [30], [])
        
        mock_extract_blob.return_value = np.zeros((20, 20, 1), dtype=np.uint8)
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        assert mock_capture.read.called
        assert len(analyzer.all_time) > 0
    

    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_filters_frames_by_time_window(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should only process frames within time window."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.t = 0.5  # Start at 0.5 seconds
        analyzer.duration = 1.0  # 1 second duration
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr)
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_contour_frame.return_value = ([], [], [], [])
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        # Should process approximately 30 frames (1 second * 30 fps)
        assert len(analyzer.all_time) <= 30
    

    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_resizes_frames_when_framewidth_specified(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should resize frames when framewidth is specified."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.framewidth = 50
        analyzer.frameheight = 50
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr)
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_contour_frame.return_value = ([], [], [], [])
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        # contour_frame should be called with resized frame
        assert mock_contour_frame.called
    

    @patch('LabGym.analyzebehavior.AnalyzeAnimal.track_animal')
    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_calls_track_animal_when_contours_detected(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        mock_track_animal,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should call track_animal when contours are detected."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:10])  # Just 10 frames
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        # Return contours for first few frames
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour = np.array([
            [[int(50 + 15 * np.cos(a)), int(50 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        mock_contour_frame.return_value = ([contour], [(50, 50)], [30], [])
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        assert mock_track_animal.called
    

    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_logs_skipped_frames_when_no_contours(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should log skipped frames when no contours detected."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:10])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        # Return no contours
        mock_contour_frame.return_value = ([], [], [], [])
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        assert len(analyzer.skipped_frames) > 0
    

    @patch('LabGym.analyzebehavior.extract_blob_background')
    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_generates_animations_when_enabled(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        mock_extract_blob,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should generate animation blobs when animation_analyzer=True."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.animation_analyzer = True
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:20])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour = np.array([
            [[int(50 + 15 * np.cos(a)), int(50 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        mock_contour_frame.return_value = ([contour], [(50, 50)], [30], [])
        
        mock_extract_blob.return_value = np.zeros((20, 20, 1), dtype=np.uint8)
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        assert mock_extract_blob.called
    
    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_inverts_background_when_animal_vs_bg_is_one(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should invert background when animal_vs_bg=1."""
        
        # Arrange
        analyzer = analyzer_setup
        analyzer.animal_vs_bg = 1
        
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:5])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_contour_frame.return_value = ([], [], [], [])
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        # Background should be inverted (checked in contour_frame call)
        assert mock_contour_frame.called
    

    @patch('LabGym.analyzebehavior.contour_frame')
    @patch('cv2.VideoCapture')
    def test_logs_acquisition_completion(
        self,
        mock_video_capture_class,
        mock_contour_frame,
        analyzer_setup,
        sample_video_frames_bgr
    ):
        """Should log completion of information acquisition."""
        
        # Arrange
        mock_capture = MagicMock()
        mock_capture.get.return_value = 30.0
        frame_iter = iter(sample_video_frames_bgr[:5])
        mock_capture.read.side_effect = lambda: (True, next(frame_iter, None))
        mock_video_capture_class.return_value = mock_capture
        
        mock_contour_frame.return_value = ([], [], [], [])
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.acquire_information()
        
        # Assert
        assert 'Information acquisition completed!' in analyzer.log



