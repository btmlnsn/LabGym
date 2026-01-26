"""
/tests.unit.behavior_analysis.test_analyze_animal_init

Tests for AnalyzeAnimal.__init__() method in analyzebehavior.py
"""

# Related third party imports
import pytest

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal


class TestAnalyzeAnimalInit:
    """Tests for AnalyzeAnimal.__init__() method."""
    
    def test_initializes_all_attributes_to_defaults(self):
        """Should initialize all attributes to expected default values."""
        analyzer = AnalyzeAnimal()
        
        # Basic attributes
        assert analyzer.path_to_video is None
        assert analyzer.basename is None
        assert analyzer.fps is None
        assert analyzer.framewidth is None
        assert analyzer.frameheight is None
        assert analyzer.kernel == 3
        assert analyzer.results_path is None
        
        # Analysis parameters
        assert analyzer.dim_tconv == 8
        assert analyzer.dim_conv == 8
        assert analyzer.channel == 1
        assert analyzer.include_bodyparts is False
        assert analyzer.std == 0
        assert analyzer.categorize_behavior is False
        assert analyzer.animation_analyzer is True
        
        # Time and duration
        assert analyzer.delta is None
        assert analyzer.animal_number is None
        assert analyzer.autofind_t is False
        assert analyzer.t == 0
        assert analyzer.duration == 5
        assert analyzer.length is None
        
        # Background and detection
        assert analyzer.animal_area is None
        assert analyzer.animal_vs_bg is None
        assert analyzer.background is None
        assert analyzer.background_low is None
        assert analyzer.background_high is None
        
        # Tracking data structures
        assert analyzer.skipped_frames == []
        assert analyzer.all_time == []
        assert analyzer.total_analysis_framecount is None
        assert analyzer.to_deregister == {}
        assert analyzer.count_to_deregister is None
        assert analyzer.register_counts == {}
        assert analyzer.animal_contours == {}
        assert analyzer.animal_centers == {}
        assert analyzer.animal_existingcenters == {}
        assert analyzer.animal_heights == {}
        assert analyzer.animal_inners == {}
        assert analyzer.animal_blobs == {}
        assert analyzer.animations == {}
        assert analyzer.pattern_images == {}
        assert analyzer.event_probability == {}
        assert analyzer.all_behavior_parameters == {}
        assert analyzer.log == []
    

    def test_creates_empty_dictionaries(self):
        """Should create empty dictionaries for tracking data."""
        analyzer = AnalyzeAnimal()
        
        assert isinstance(analyzer.to_deregister, dict)
        assert isinstance(analyzer.register_counts, dict)
        assert isinstance(analyzer.animal_contours, dict)
        assert isinstance(analyzer.animal_centers, dict)
        assert isinstance(analyzer.animal_existingcenters, dict)
        assert isinstance(analyzer.animal_heights, dict)
        assert isinstance(analyzer.animal_inners, dict)
        assert isinstance(analyzer.animal_blobs, dict)
        assert isinstance(analyzer.animations, dict)
        assert isinstance(analyzer.pattern_images, dict)
        assert isinstance(analyzer.event_probability, dict)
        assert isinstance(analyzer.all_behavior_parameters, dict)
    

    def test_creates_empty_lists(self):
        """Should create empty lists for time series data."""
        analyzer = AnalyzeAnimal()
        
        assert isinstance(analyzer.skipped_frames, list)
        assert isinstance(analyzer.all_time, list)
        assert isinstance(analyzer.log, list)



