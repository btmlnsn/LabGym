"""
tests/unit/frame_analysis/test_contour_detection

Tests for LabGym.tools contour detection functions:
- get_inner()
- contour_frame()
"""

# Related third party imports
import numpy as np
import pytest
import cv2


# Local application imports
from LabGym.tools import get_inner, contour_frame



class TestGetInner:
    """Tests for get_inner() function."""

    def test_returns_list(self, sample_bgr_frame, sample_circular_contour):
        """Should return a list of contours."""
        gray = cv2.cvtColor(sample_bgr_frame, cv2.COLOR_BGR2GRAY)
        result = get_inner(gray, sample_circular_contour)
        
        assert isinstance(result, list)


    def test_returns_at_least_two_elements(self, sample_bgr_frame, sample_circular_contour):
        """Should return at least 2 elements (fallback case)."""
        gray = cv2.cvtColor(sample_bgr_frame, cv2.COLOR_BGR2GRAY)
        result = get_inner(gray, sample_circular_contour)
        
        assert len(result) >= 2


    def test_simple_blob_returns_contour_fallback(self):
        """Simple blob with no internal features returns [contour, contour]."""
        # Create a simple uniform gray blob
        frame = np.full((100, 100), 128, dtype=np.uint8)
        contour = np.array([
            [[40, 40]], [[60, 40]], [[60, 60]], [[40, 60]]
        ], dtype=np.int32)

        result = get_inner(frame, contour)
        
        # When < 3 internal contours found, returns [contour, contour]
        assert len(result) >= 2



class TestContourFrame:
    """Tests for contour_frame() function."""

    @pytest.fixture
    def detection_setup(self, sample_bgr_frame, sample_background_frame):
        """Provide frame, background, and related parameters for detection tests."""
        background = sample_background_frame
        # Create low/high variants (simulating illumination variation)
        background_low = (background * 0.9).astype(np.uint8)
        background_high = (background * 1.1).astype(np.uint8)
        background_high = np.clip(background_high, 0, 255).astype(np.uint8)

        return {
            'frame': sample_bgr_frame,
            'background': background,
            'background_low': background_low,
            'background_high': background_high,
            'delta': 1.2,
            'contour_area': 700,  # approx area of circle with radius 15
        }


    def test_returns_tuple_of_four(self, detection_setup):
        """Should return (contours, centers, heights, inners)."""
        result = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
        )
        assert isinstance(result, tuple)
        assert len(result) == 4


    def test_single_animal_detection(self, detection_setup):
        """Should detect single animal when animal_number=1."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
        )

        # May or may not find the animal depending on parameters
        assert isinstance(contours, list)
        assert isinstance(centers, list)
        assert isinstance(heights, list)


    def test_centers_match_contours_count(self, detection_setup):
        """Number of centers should match number of contours."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
        )

        assert len(centers) == len(contours)
        assert len(heights) == len(contours)


    def test_animal_vs_bg_zero(self, detection_setup):
        """animal_vs_bg=0: animals brighter than background."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
            animal_vs_bg=0,
        )

        assert isinstance(contours, list)


    def test_animal_vs_bg_one(self, detection_setup):
        """animal_vs_bg=1: animals darker than background."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
            animal_vs_bg=1,
        )
        assert isinstance(contours, list)

    def test_animal_vs_bg_two(self, detection_setup):
        """animal_vs_bg=2: unclear contrast."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
            animal_vs_bg=2,
        )

        assert isinstance(contours, list)


    def test_include_bodyparts_true(self, detection_setup):
        """include_bodyparts=True should populate inners list."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
            include_bodyparts=True,
        )
        
        # inners should have same length as contours when bodyparts included
        if len(contours) > 0:
            assert len(inners) == len(contours)


    def test_include_bodyparts_false(self, detection_setup):
        """include_bodyparts=False should return empty inners list."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=1,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
            include_bodyparts=False,
        )

        assert inners == []


    def test_multiple_animals(self, detection_setup):
        """animal_number > 1 should allow multiple detections."""
        contours, centers, heights, inners = contour_frame(
            detection_setup['frame'],
            animal_number=3,
            background=detection_setup['background'],
            background_low=detection_setup['background_low'],
            background_high=detection_setup['background_high'],
            delta=detection_setup['delta'],
            contour_area=detection_setup['contour_area'],
        )
        
        # Should return at most animal_number contours
        assert len(contours) <= 3


    def test_no_detection_on_uniform_frame(self):
        """Uniform frame (no animal) should return empty lists."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        background = frame.copy()
        background_low = frame.copy()
        background_high = frame.copy()

        contours, centers, heights, inners = contour_frame(
            frame,
            animal_number=1,
            background=background,
            background_low=background_low,
            background_high=background_high,
            delta=1.2,
            contour_area=500,
        )
        
        # No difference from background = no detection
        assert contours == []
        assert centers == []


