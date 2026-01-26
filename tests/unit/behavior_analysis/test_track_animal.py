"""
/tests.unit.analysis.test_track_animal

Tests for AnalyzeAnimal.track_animal() method in analyzebehavior.py
"""

# Standard library imports
from unittest.mock import patch, MagicMock

# Related third party imports
import numpy as np
import pytest
import cv2
from scipy.spatial import distance

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestTrackAnimal:
    """Tests for track_animal() method."""
    
    @pytest.fixture
    def analyzer_setup(self, sample_background_frame):
        """Create analyzer with basic setup for tracking tests."""
        
        analyzer = AnalyzeAnimal()
        analyzer.animal_number = 2
        analyzer.length = 15
        analyzer.dim_conv = 8
        analyzer.background = sample_background_frame
        analyzer.include_bodyparts = False
        analyzer.std = 0
        
        # Initialize tracking structures
        analyzer.to_deregister = {0: 0, 1: 0}
        analyzer.register_counts = {0: None, 1: None}
        analyzer.animal_contours = {0: [None] * 100, 1: [None] * 100}
        analyzer.animal_centers = {0: [None] * 100, 1: [None] * 100}
        analyzer.animal_existingcenters = {0: (50, 50), 1: (80, 80)}
        analyzer.animal_heights = {0: [None] * 100, 1: [None] * 100}
        analyzer.pattern_images = {
            0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 100,
            1: [np.zeros((8, 8, 3), dtype=np.uint8)] * 100
        }
        analyzer.count_to_deregister = 60
        
        return analyzer
    

    def test_registers_new_animal_on_first_detection(self, analyzer_setup, sample_contours_centers_heights):
        """Should register animal when first detected."""
        
        analyzer = analyzer_setup
        analyzer.register_counts[0] = None
        
        contours, centers, heights, inners = sample_contours_centers_heights
        frame_count = 0
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        assert analyzer.register_counts[0] == 0
        assert analyzer.animal_centers[0][0] == centers[0]
        assert analyzer.animal_contours[0][0] is not None
    

    def test_matches_animals_by_distance(self, analyzer_setup, sample_contours_centers_heights):
        """Should match detected animals to existing tracks by minimum distance."""
        
        analyzer = analyzer_setup
        
        # Create contours at known positions
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour1 = np.array([
            [[int(52 + 15 * np.cos(a)), int(52 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        contour2 = np.array([
            [[int(82 + 15 * np.cos(a)), int(82 + 15 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        
        contours = [contour1, contour2]
        centers = [(52, 52), (82, 82)]
        heights = [30, 30]
        
        frame_count = 10
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights)
        
        # Assert
        # Should match to existing centers at (50, 50) and (80, 80)
        assert analyzer.animal_centers[0][frame_count] == centers[0]
        assert analyzer.animal_centers[1][frame_count] == centers[1]
    

    def test_updates_existing_centers(self, analyzer_setup, sample_contours_centers_heights):
        """Should update animal_existingcenters with new positions."""
        
        analyzer = analyzer_setup
        original_center = analyzer.animal_existingcenters[0]
        
        contours, centers, heights, inners = sample_contours_centers_heights
        new_center = (55, 55)
        centers[0] = new_center
        frame_count = 5
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        assert analyzer.animal_existingcenters[0] == new_center
        assert analyzer.animal_existingcenters[0] != original_center
    

    def test_stores_contours_and_heights(self, analyzer_setup, sample_contours_centers_heights):
        """Should store contours and heights in tracking arrays."""
        
        analyzer = analyzer_setup
        contours, centers, heights, inners = sample_contours_centers_heights
        frame_count = 20
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        assert analyzer.animal_contours[0][frame_count] is not None
        assert analyzer.animal_heights[0][frame_count] == heights[0]
    

    @patch('LabGym.analyzebehavior.generate_patternimage')
    def test_generates_pattern_image(
        self,
        mock_generate_patternimage,
        analyzer_setup,
        sample_contours_centers_heights
    ):
        """Should generate and resize pattern image for each tracked animal."""
        
        # Arrange
        mock_pattern = np.zeros((16, 16, 3), dtype=np.uint8)
        mock_generate_patternimage.return_value = mock_pattern
        
        analyzer = analyzer_setup
        contours, centers, heights, inners = sample_contours_centers_heights
        frame_count = 10
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        mock_generate_patternimage.assert_called()
        # Pattern image should be resized to dim_conv x dim_conv
        assert analyzer.pattern_images[0][frame_count].shape == (8, 8, 3)
    

    @patch('LabGym.analyzebehavior.distance.cdist')
    def test_increments_deregister_counter_for_missing_animals(self, mock_cdist, analyzer_setup):
        """Should increment deregister counter when animal not detected."""
        
        # Arrange
        # Mock cdist to return empty array when centers is empty
        mock_cdist.return_value = np.array([]).reshape(0, 0)
        
        analyzer = analyzer_setup
        analyzer.to_deregister[0] = 10
        
        # No contours detected
        contours = []
        centers = []
        heights = []
        frame_count = 15
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights)
        
        # Assert
        assert analyzer.to_deregister[0] == 11
    

    @patch('LabGym.analyzebehavior.distance.cdist')
    def test_deregisters_animal_after_threshold(self, mock_cdist, analyzer_setup):
        """Should deregister animal after count_to_deregister threshold."""
        
        # Arrange
        # Mock cdist to return empty array when centers is empty
        mock_cdist.return_value = np.array([]).reshape(0, 0)
        
        analyzer = analyzer_setup
        analyzer.to_deregister[0] = analyzer.count_to_deregister + 1
        analyzer.animal_existingcenters[0] = (50, 50)
        
        # No contours detected
        contours = []
        centers = []
        heights = []
        frame_count = 20
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights)
        
        # Assert
        assert analyzer.animal_existingcenters[0] == (-10000, -10000)
    

    def test_resets_deregister_counter_on_detection(self, analyzer_setup, sample_contours_centers_heights):
        """Should reset deregister counter when animal is detected again."""
        
        analyzer = analyzer_setup
        analyzer.to_deregister[0] = 50  # High count
        
        contours, centers, heights, inners = sample_contours_centers_heights
        frame_count = 25
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        assert analyzer.to_deregister[0] == 0
    

    @patch('LabGym.analyzebehavior.generate_patternimage')
    def test_handles_bodyparts_when_included(
        self,
        mock_generate_patternimage,
        analyzer_setup,
        sample_contours_centers_heights
    ):
        """Should handle body parts when include_bodyparts=True."""
        
        # Arrange
        mock_pattern = np.zeros((16, 16, 3), dtype=np.uint8)
        mock_generate_patternimage.return_value = mock_pattern
        
        analyzer = analyzer_setup
        analyzer.include_bodyparts = True
        # Initialize as deque with maxlen (as done in prepare_analysis)
        from collections import deque
        analyzer.animal_inners = {0: deque(maxlen=analyzer.length), 1: deque(maxlen=analyzer.length)}
        
        contours, centers, heights, _ = sample_contours_centers_heights
        # inners should be a list with same length as contours
        # Each element should be a list of inner contours (result of get_inner)
        # For testing, we'll use a simple list with one inner contour per detected animal
        inners = [[contours[0], contours[0]]]  # Mock: list of inner contours for first animal
        
        frame_count = 30
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights, inners)
        
        # Assert
        assert len(analyzer.animal_inners[0]) > 0
        mock_generate_patternimage.assert_called()
    

    def test_handles_multiple_animals(self, analyzer_setup):
        """Should track multiple animals simultaneously."""
        
        analyzer = analyzer_setup
        
        # Create two distinct contours
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        contour1 = np.array([
            [[int(30 + 10 * np.cos(a)), int(30 + 10 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        contour2 = np.array([
            [[int(70 + 10 * np.cos(a)), int(70 + 10 * np.sin(a))]]
            for a in angles
        ], dtype=np.int32)
        
        contours = [contour1, contour2]
        centers = [(30, 30), (70, 70)]
        heights = [20, 20]
        frame_count = 5
        
        # Act
        analyzer.track_animal(frame_count, contours, centers, heights)
        
        # Assert
        assert analyzer.animal_centers[0][frame_count] is not None
        assert analyzer.animal_centers[1][frame_count] is not None



        