"""
LabGym.tests.unit.behavior_analysis.test_craft_data

Tests for AnalyzeAnimal.craft_data() method.
"""

# Related third party imports
import numpy as np
import pytest

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestCraftData:
    """Tests for craft_data() method."""
    
    @pytest.fixture
    def analyzer_setup(self):
        """Create analyzer with tracking data for crafting tests."""
        
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.include_bodyparts = False
        analyzer.animation_analyzer = True
        
        # Create data for 3 animals
        analyzer.animal_centers = {
            0: [(10, 10), (11, 11), (12, 12), None, None],  # 3 valid
            1: [None, None, None, None, None],  # 0 valid - should be deleted
            2: [(20, 20), (21, 21), None, None, None],  # 2 valid
        }
        analyzer.animal_heights = {
            0: [30, 31, 32, None, None],
            1: [None, None, None, None, None],
            2: [40, 41, None, None, None],
        }
        analyzer.animal_contours = {
            0: [None] * 5,
            1: [None] * 5,
            2: [None] * 5,
        }
        analyzer.animal_existingcenters = {
            0: (10, 10),
            1: (-10000, -10000),
            2: (20, 20),
        }
        analyzer.register_counts = {
            0: 0,  # Registered
            1: None,  # Never registered - should be deleted
            2: 0,  # Registered
        }
        analyzer.animations = {
            0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 5,
            1: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 5,
            2: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 5,
        }
        analyzer.animal_blobs = {  # ADD THIS - required when animation_analyzer=True
            0: [],
            1: [],
            2: [],
        }
        analyzer.pattern_images = {
            0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 5,
            1: [np.zeros((8, 8, 3), dtype=np.uint8)] * 5,
            2: [np.zeros((8, 8, 3), dtype=np.uint8)] * 5,
        }
        analyzer.to_deregister = {0: 0, 1: 0, 2: 0}
        analyzer.all_time = [0.0, 0.033, 0.067, 0.1, 0.133]
        
        return analyzer
    

    def test_removes_animals_with_no_detections(self, analyzer_setup):
        """Should remove animals that were never registered."""
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.craft_data()
        
        # Assert
        # Animal 1 should be removed (never registered)
        assert 1 not in analyzer.animal_centers
        assert 1 not in analyzer.animal_heights
        assert 1 not in analyzer.animal_contours
    

    def test_keeps_at_least_one_animal(self, analyzer_setup):
        """Should keep at least one animal even if all have issues."""
        
        analyzer = analyzer_setup
        # Make all animals unregistered
        analyzer.register_counts = {0: None, 1: None, 2: None}
        
        # Act
        analyzer.craft_data()
        
        # Assert
        # Should keep at least one (the one with most detections)
        assert len(analyzer.animal_centers) >= 1
    

    def test_trims_data_to_actual_length(self, analyzer_setup):
        """Should trim all data arrays to match all_time length."""
        
        analyzer = analyzer_setup
        analyzer.all_time = [0.0, 0.033, 0.067]  # Only 3 time points
        
        # Act
        analyzer.craft_data()
        
        # Assert
        assert len(analyzer.animal_centers[0]) == 3
        assert len(analyzer.animal_contours[0]) == 3
        assert len(analyzer.animal_heights[0]) == 3
        assert len(analyzer.animations[0]) == 3
        assert len(analyzer.pattern_images[0]) == 3
    

    def test_cleans_up_tracking_structures(self, analyzer_setup):
        """Should clean up tracking structures for removed animals."""
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.craft_data()
        
        # Assert
        # Animal 1 should be removed from all structures
        assert 1 not in analyzer.to_deregister
        assert 1 not in analyzer.register_counts
    

    def test_preserves_valid_animals(self, analyzer_setup):
        """Should preserve animals with valid detections."""
        
        analyzer = analyzer_setup
        
        # Act
        analyzer.craft_data()
        
        # Assert
        # Animals 0 and 2 should be preserved
        assert 0 in analyzer.animal_centers
        assert 2 in analyzer.animal_centers
    

    def test_handles_bodyparts_cleanup(self):
        """Should clean up animal_inners when include_bodyparts=True."""
        
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.include_bodyparts = True
        analyzer.animation_analyzer = False
        
        # Animal 1 must be in animal_centers for craft_data to delete it
        analyzer.animal_centers = {0: [(10, 10), (11, 11)], 1: [None, None]}
        analyzer.animal_heights = {0: [30, 31], 1: [None, None]}
        analyzer.animal_contours = {0: [None] * 2, 1: [None] * 2}
        analyzer.animal_inners = {0: [[], []], 1: [[], []]}
        analyzer.animal_existingcenters = {0: (10, 10), 1: (-10000, -10000)}
        analyzer.register_counts = {0: 0, 1: None}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2, 1: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2}
        analyzer.to_deregister = {0: 0, 1: 0}
        analyzer.all_time = [0.0, 0.033]
        
        # Act
        analyzer.craft_data()
        
        # Assert
        assert 1 not in analyzer.animal_inners
    

    def test_handles_animation_analyzer_cleanup(self):
        """Should clean up animations when animation_analyzer=True."""
        
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.include_bodyparts = False
        analyzer.animation_analyzer = True
        
        # Animal 1 must be in animal_centers for craft_data to delete it
        analyzer.animal_centers = {0: [(10, 10)], 1: [None]}
        analyzer.animal_heights = {0: [30], 1: [None]}
        analyzer.animal_contours = {0: [None], 1: [None]}
        analyzer.animal_existingcenters = {0: (10, 10), 1: (-10000, -10000)}
        analyzer.animations = {0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)], 1: [np.zeros((15, 8, 8, 1), dtype=np.uint8)]}
        analyzer.animal_blobs = {0: [], 1: []}  # ADD THIS
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)], 1: [np.zeros((8, 8, 3), dtype=np.uint8)]}
        analyzer.register_counts = {0: 0, 1: None}
        analyzer.to_deregister = {0: 0, 1: 0}
        analyzer.all_time = [0.0]
        
        # Act
        analyzer.craft_data()
        
        # Assert
        assert 1 not in analyzer.animations


