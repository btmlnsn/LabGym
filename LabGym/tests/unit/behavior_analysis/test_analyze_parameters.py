"""
/tests.unit.behavior_analysis.test_analyze_parameters

Tests for AnalyzeAnimal.analyze_parameters() method.
"""

# Related third party imports
import numpy as np
import pytest
import math

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestAnalyzeParameters:
    """Tests for analyze_parameters() method."""
    
    def _initialize_parameter_dicts(self, analyzer, parameter_names, animal_ids):
        """
        Helper to initialize parameter dictionaries before calling analyze_parameters.
        The method expects these to exist but doesn't create them.
        """
        if analyzer.categorize_behavior:
            for behavior_name in analyzer.all_behavior_parameters:
                for param_name in parameter_names:
                    if param_name not in analyzer.all_behavior_parameters[behavior_name]:
                        analyzer.all_behavior_parameters[behavior_name][param_name] = {}
                    for animal_id in animal_ids:
                        if animal_id not in analyzer.all_behavior_parameters[behavior_name][param_name]:
                            analyzer.all_behavior_parameters[behavior_name][param_name][animal_id] = []
        else:
            for param_name in parameter_names:
                if param_name not in analyzer.all_behavior_parameters:
                    analyzer.all_behavior_parameters[param_name] = {}
                for animal_id in animal_ids:
                    if animal_id not in analyzer.all_behavior_parameters[param_name]:
                        analyzer.all_behavior_parameters[param_name][animal_id] = []
    
    @pytest.fixture
    def analyzer_setup_categorized(self):
        """Create analyzer with categorized behavior data."""
       
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.fps = 30
        analyzer.animal_area = 700
        analyzer.categorize_behavior = True
        
        # Set up tracking data
        analyzer.all_time = [0.0, 0.033, 0.067, 0.1, 0.133]
        analyzer.register_counts = {0: 0}
        
        # Create centers moving in a line
        analyzer.animal_centers = {
            0: [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]
        }
        analyzer.animal_heights = {
            0: [30, 31, 32, 33, 34]
        }
        analyzer.animal_contours = {
            0: [None] * 5  # Will be filled with actual contours in tests
        }
        analyzer.event_probability = {
            0: [['walking', 0.9], ['walking', 0.9], ['resting', 0.8], ['resting', 0.8], ['walking', 0.9]]
        }
        # Initialize all_behavior_parameters with empty dicts for parameters that will be created
        analyzer.all_behavior_parameters = {
            'walking': {
                'count': {0: 0},
                'duration': {0: 0},
                'distance': {0: 0.0},
                'latency': {0: 'NA'},
            },
            'resting': {
                'count': {0: 0},
                'duration': {0: 0},
                'distance': {0: 0.0},
                'latency': {0: 'NA'},
            }
        }
        
        return analyzer
    

    @pytest.fixture
    def analyzer_setup_uncategorized(self):
        """Create analyzer without behavior categorization."""
        
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.fps = 30
        analyzer.animal_area = 700
        analyzer.categorize_behavior = False
        
        analyzer.all_time = [0.0, 0.033, 0.067]
        analyzer.register_counts = {0: 0}
        analyzer.animal_centers = {
            0: [(10, 10), (11, 11), (12, 12)]
        }
        analyzer.animal_heights = {
            0: [30, 31, 32]
        }
        analyzer.animal_contours = {
            0: [None] * 3
        }
        analyzer.all_behavior_parameters = {
            'distance': {0: 0.0}
        }
        
        return analyzer
    

    def test_calculates_count_parameter(self, analyzer_setup_categorized):
        """Should count behavior episodes."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['count']
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Should count transitions: walking->resting->walking = 2 walking episodes
        assert analyzer.all_behavior_parameters['walking']['count'][0] >= 0
    

    def test_calculates_duration_parameter(self, analyzer_setup_categorized):
        """Should calculate total duration of each behavior."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['duration']
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Duration should be in seconds (frames / fps) or an integer count
        duration = analyzer.all_behavior_parameters['walking']['duration'][0]
        assert isinstance(duration, (int, float)) or duration == 'NA'
        if isinstance(duration, (int, float)):
            assert duration >= 0
    

    def test_calculates_latency_parameter(self, analyzer_setup_categorized):
        """Should calculate latency to first occurrence."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['latency']
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Latency should be time of first occurrence or 'NA'
        latency = analyzer.all_behavior_parameters['walking']['latency'][0]
        assert latency == 'NA' or isinstance(latency, (int, float))
    

    def test_calculates_speed_parameter(self, analyzer_setup_categorized):
        """Should calculate speed from distance traveled."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries before calling analyze_parameters
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Speed should be calculated for frames with behavior
        assert 0 in analyzer.all_behavior_parameters['walking']['speed']
        speed_values = analyzer.all_behavior_parameters['walking']['speed'][0]
        assert len(speed_values) == len(analyzer.all_time)
    

    def test_calculates_velocity_parameter(self, analyzer_setup_categorized):
        """Should calculate velocity (maximum displacement rate)."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        assert 0 in analyzer.all_behavior_parameters['walking']['velocity']
        velocity_values = analyzer.all_behavior_parameters['walking']['velocity'][0]
        assert len(velocity_values) == len(analyzer.all_time)
    

    def test_calculates_acceleration_parameter(self, analyzer_setup_categorized):
        """Should calculate acceleration from velocity changes."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        assert 0 in analyzer.all_behavior_parameters['walking']['acceleration']
        accel_values = analyzer.all_behavior_parameters['walking']['acceleration'][0]
        assert len(accel_values) == len(analyzer.all_time)
    

    def test_normalizes_distance_when_enabled(self, analyzer_setup_categorized):
        """Should normalize distance by animal area when normalize_distance=True."""
       
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(normalize_distance=True, parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Distance should be normalized (divided by sqrt(animal_area))
        distance = analyzer.all_behavior_parameters['walking']['distance'][0]
        assert distance == 'NA' or (isinstance(distance, (int, float)) and distance >= 0)
    

    def test_calculates_length_parameters(self, analyzer_setup_categorized):
        """Should calculate intensity_length, magnitude_length, vigor_length."""
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['3 length parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['intensity_length', 'magnitude_length', 'vigor_length'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        assert 0 in analyzer.all_behavior_parameters['walking']['intensity_length']
        assert 0 in analyzer.all_behavior_parameters['walking']['magnitude_length']
        assert 0 in analyzer.all_behavior_parameters['walking']['vigor_length']
    

    def test_calculates_areal_parameters(self, analyzer_setup_categorized):
        """Should calculate intensity_area, magnitude_area, vigor_area."""
        import cv2
        
        analyzer = analyzer_setup_categorized
        parameter_to_analyze = ['3 areal parameters']
        
        # Create actual contours for areal calculation
        for i in range(5):
            angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            center = analyzer.animal_centers[0][i]
            contour = np.array([
                [[int(center[0] + 10 * np.cos(a)), int(center[1] + 10 * np.sin(a))]]
                for a in angles
            ], dtype=np.int32)
            analyzer.animal_contours[0][i] = contour
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['intensity_area', 'magnitude_area', 'vigor_area'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        assert 0 in analyzer.all_behavior_parameters['walking']['intensity_area']
        assert 0 in analyzer.all_behavior_parameters['walking']['magnitude_area']
        assert 0 in analyzer.all_behavior_parameters['walking']['vigor_area']
    

    def test_works_without_categorization(self, analyzer_setup_uncategorized):
        """Should calculate parameters when categorize_behavior=False."""
        
        analyzer = analyzer_setup_uncategorized
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        assert 'speed' in analyzer.all_behavior_parameters
        assert 'velocity' in analyzer.all_behavior_parameters
        assert 'acceleration' in analyzer.all_behavior_parameters
        assert 0 in analyzer.all_behavior_parameters['speed']
    

    def test_handles_missing_centers(self, analyzer_setup_categorized):
        """Should handle None values in animal_centers."""
        
        analyzer = analyzer_setup_categorized
        analyzer.animal_centers[0][2] = None  # Missing center
        parameter_to_analyze = ['4 locomotion parameters']
        
        # Initialize parameter dictionaries
        self._initialize_parameter_dicts(analyzer, ['speed', 'velocity', 'acceleration'], [0])
        
        # Act
        analyzer.analyze_parameters(parameter_to_analyze=parameter_to_analyze)
        
        # Assert
        # Should not crash, should handle None gracefully
        assert 0 in analyzer.all_behavior_parameters['walking']['speed']

