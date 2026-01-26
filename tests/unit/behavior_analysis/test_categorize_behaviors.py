"""
LabGym.tests.unit.behavior_analysis.test_categorize_behaviors

Tests for AnalyzeAnimal.categorize_behaviors() method.
"""

# Standard library imports
from unittest.mock import patch, MagicMock

# Related third party imports
import numpy as np
import pytest

# Local application imports
from LabGym.analyzebehavior import AnalyzeAnimal



class TestCategorizeBehaviors:
    """Tests for categorize_behaviors() method."""
    
    @pytest.fixture
    def analyzer_setup(self):
        """Create analyzer with pattern images and animations."""
        
        analyzer = AnalyzeAnimal()
        analyzer.length = 15
        analyzer.animation_analyzer = True
        analyzer.dim_tconv = 8
        analyzer.channel = 1
        
        # Set up data for categorization
        analyzer.all_time = [0.0, 0.033, 0.067, 0.1]
        analyzer.register_counts = {0: 0}
        analyzer.animal_contours = {
            0: [None] * 4  # 4 frames
        }
        analyzer.pattern_images = {
            0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 4
        }
        analyzer.animations = {
            0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 4
        }
        analyzer.all_behavior_parameters = {
            'walking': {
                'color': ('Walking', '#FF0000'),
                'probability': {0: []},
            },
            'resting': {
                'color': ('Resting', '#00FF00'),
                'probability': {0: []},
            }
        }
        
        return analyzer
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_loads_keras_model(
        self,
        mock_load_model,
        analyzer_setup,
        mock_keras_model,
        tmp_path
    ):
        """Should load Keras model from specified path."""
        
        # Arrange
        mock_load_model.return_value = mock_keras_model
        analyzer = analyzer_setup
        model_path = str(tmp_path / "model.h5")
        
        # Act
        analyzer.categorize_behaviors(model_path)
        
        # Assert
        mock_load_model.assert_called_once_with(model_path)
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_concatenates_pattern_images(
        self,
        mock_load_model,
        analyzer_setup,
        mock_keras_model
    ):
        """Should concatenate pattern images from all animals."""
        
        # Arrange
        mock_load_model.return_value = mock_keras_model
        analyzer = analyzer_setup
        analyzer.pattern_images = {
            0: [np.ones((8, 8, 3), dtype=np.uint8)] * 2,
            1: [np.ones((8, 8, 3), dtype=np.uint8)] * 2
        }
        analyzer.animations = {
            0: [np.ones((15, 8, 8, 1), dtype=np.uint8)] * 2,
            1: [np.ones((15, 8, 8, 1), dtype=np.uint8)] * 2
        }
        analyzer.register_counts = {0: 0, 1: 0}
        analyzer.animal_contours = {0: [None] * 2, 1: [None] * 2}
        analyzer.all_time = [0.0, 0.033]
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5")
        
        # Assert
        # Model should be called with concatenated data
        assert mock_keras_model.predict.called
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_initializes_probability_arrays(
        self,
        mock_load_model,
        analyzer_setup,
        mock_keras_model
    ):
        """Should initialize probability arrays for each behavior."""
        
        # Arrange
        mock_load_model.return_value = mock_keras_model
        analyzer = analyzer_setup
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5")
        
        # Assert
        assert len(analyzer.all_behavior_parameters['walking']['probability'][0]) == len(analyzer.all_time)
        assert len(analyzer.all_behavior_parameters['resting']['probability'][0]) == len(analyzer.all_time)
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_assigns_event_probabilities(
        self,
        mock_load_model,
        analyzer_setup,
        mock_keras_model
    ):
        """Should assign event probabilities based on predictions."""
        
        # Arrange
        # Mock predictions: high probability for first behavior
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.8], [0.9], [0.2], [0.85]], dtype=np.float32)
        mock_load_model.return_value = mock_model
        
        analyzer = analyzer_setup
        analyzer.event_probability = {0: []}
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5", uncertain=0.1)
        
        # Assert
        # Should have event probabilities assigned
        assert len(analyzer.event_probability[0]) == len(analyzer.all_time)
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_applies_uncertainty_threshold(
        self,
        mock_load_model,
        analyzer_setup
    ):
        """Should apply uncertainty threshold to filter predictions."""
        
        # Arrange
        mock_model = MagicMock()
        # Predictions with small difference (uncertain)
        mock_model.predict.return_value = np.array([[0.52], [0.51]], dtype=np.float32)
        mock_load_model.return_value = mock_model
        
        analyzer = analyzer_setup
        analyzer.animal_contours = {0: [None] * 2}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2}
        analyzer.animations = {0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 2}
        analyzer.all_time = [0.0, 0.033]
        analyzer.event_probability = {0: []}
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5", uncertain=0.1)
        
        # Assert
        # With uncertainty threshold, should mark as 'NA' when difference is small
        # (0.52 - 0.48 = 0.04 < 0.1 threshold)
        assert len(analyzer.event_probability[0]) == len(analyzer.all_time)
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_filters_by_min_length(
        self,
        mock_load_model,
        analyzer_setup
    ):
        """Should filter brief behaviors using min_length parameter."""
        
        # Arrange
        mock_model = MagicMock()
        # Alternating predictions
        mock_model.predict.return_value = np.array([
            [0.8], [0.8], [0.2], [0.2], [0.8], [0.2]
        ], dtype=np.float32)
        mock_load_model.return_value = mock_model
        
        analyzer = analyzer_setup
        analyzer.animal_contours = {0: [None] * 6}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 6}
        analyzer.animations = {0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 6}
        analyzer.all_time = [0.0, 0.033, 0.067, 0.1, 0.133, 0.167]
        analyzer.event_probability = {0: []}
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5", min_length=3)
        
        # Assert
        # Brief behaviors (< 3 frames) should be filtered to 'NA'
        assert len(analyzer.event_probability[0]) == len(analyzer.all_time)
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_handles_binary_classification(
        self,
        mock_load_model,
        analyzer_setup
    ):
        """Should handle binary classification (2 behaviors)."""
        
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.7], [0.3]], dtype=np.float32)
        mock_load_model.return_value = mock_model
        
        analyzer = analyzer_setup
        analyzer.animal_contours = {0: [None] * 2}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2}
        analyzer.animations = {0: [np.zeros((15, 8, 8, 1), dtype=np.uint8)] * 2}
        analyzer.all_time = [0.0, 0.033]
        analyzer.event_probability = {0: []}
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5")
        
        # Assert
        # For binary: prob[0] = 1 - prediction, prob[1] = prediction
        assert len(analyzer.all_behavior_parameters['walking']['probability'][0]) == 2
        assert len(analyzer.all_behavior_parameters['resting']['probability'][0]) == 2
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_works_without_animation_analyzer(
        self,
        mock_load_model,
        analyzer_setup
    ):
        """Should work when animation_analyzer=False."""
       
        # Arrange
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.7], [0.3]], dtype=np.float32)
        mock_load_model.return_value = mock_model
        
        analyzer = analyzer_setup
        analyzer.animation_analyzer = False
        analyzer.animal_contours = {0: [None] * 2}
        analyzer.pattern_images = {0: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2}
        analyzer.all_time = [0.0, 0.033]
        analyzer.event_probability = {0: []}
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5")
        
        # Assert
        # Should work with only pattern images
        assert mock_model.predict.called
    

    @patch('LabGym.analyzebehavior.load_model')
    def test_cleans_up_memory(
        self,
        mock_load_model,
        analyzer_setup,
        mock_keras_model
    ):
        """Should delete animations and pattern_images to free memory."""
        
        # Arrange
        mock_load_model.return_value = mock_keras_model
        analyzer = analyzer_setup
        
        # Act
        analyzer.categorize_behaviors("dummy_path.h5")
        
        # Assert
        # These should be deleted after categorization
        assert not hasattr(analyzer, 'animations') or analyzer.animations == {}
        assert not hasattr(analyzer, 'pattern_images') or analyzer.pattern_images == {}



        