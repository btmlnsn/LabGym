"""
/tests.unit.categorizer.test_model_architecture

Tests for LabGym categorizer model architecture construction:
- simple_vgg, simple_tvgg (VGG-style CNNs)
- simple_resnet, simple_tresnet (ResNet architectures)
- res_block, tres_block (Residual blocks)
- combined_network (Animation Analyzer + Pattern Recognizer)
"""

# Related third party imports
import numpy as np
import pytest
import tensorflow as tf
from keras import Input

# Local application imports
from LabGym.categorizer import Categorizers


class TestSimpleVGG:
    """Tests for simple_vgg() model architecture."""

    def test_builds_vgg_without_classifier(self):
        """Should build VGG feature extractor without classifier."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        output = categorizer.simple_vgg(
            inputs, filters=8, classes=3, level=2, with_classifier=False
        )
        assert output is not None
        assert hasattr(output, 'shape')

    def test_builds_vgg_with_classifier_binary(self):
        """Should build VGG model with binary classifier (sigmoid)."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        model = categorizer.simple_vgg(
            inputs, filters=8, classes=2, level=2, with_classifier=True
        )
        assert model is not None
        assert model.output_shape[-1] == 1

    def test_builds_vgg_with_classifier_multiclass(self):
        """Should build VGG model with multiclass classifier (softmax)."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        model = categorizer.simple_vgg(
            inputs, filters=8, classes=5, level=2, with_classifier=True
        )
        assert model is not None
        assert model.output_shape[-1] == 5

    def test_vgg_level_affects_depth(self):
        """Should create deeper networks with higher level."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        model_low = categorizer.simple_vgg(
            inputs, filters=8, classes=3, level=2, with_classifier=True
        )
        model_high = categorizer.simple_vgg(
            inputs, filters=8, classes=3, level=4, with_classifier=True
        )
        assert len(model_high.layers) > len(model_low.layers)


class TestSimpleTVGG:
    """Tests for simple_tvgg() (Time-distributed VGG)."""

    def test_builds_tvgg_without_classifier(self):
        """Should build Time-distributed VGG feature extractor."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        output = categorizer.simple_tvgg(
            inputs, filters=8, classes=3, level=2, with_classifier=False
        )
        assert output is not None
        assert hasattr(output, 'shape')

    def test_builds_tvgg_with_classifier(self):
        """Should build Time-distributed VGG with classifier."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        model = categorizer.simple_tvgg(
            inputs, filters=8, classes=3, level=2, with_classifier=True
        )
        assert model is not None
        assert model.output_shape[-1] == 3

    def test_tvgg_includes_lstm_layer(self):
        """Should include LSTM layer in Time-distributed VGG."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        model = categorizer.simple_tvgg(
            inputs, filters=8, classes=3, level=2, with_classifier=True
        )
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        assert 'LSTM' in layer_names


class TestResBlock:
    """Tests for res_block() residual block construction."""

    def test_builds_basic_res_block(self):
        """Should build basic residual block."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)
        output = categorizer.res_block(
            x, filters=16, strides=2, block=False, basic=True
        )
        assert output is not None
        assert hasattr(output, 'shape')

    def test_builds_bottleneck_res_block(self):
        """Should build bottleneck residual block."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)
        output = categorizer.res_block(
            x, filters=16, strides=2, block=False, basic=False
        )
        assert output is not None


class TestTResBlock:
    """Tests for tres_block() (Time-distributed residual block)."""

    def test_builds_time_distributed_res_block(self):
        """Should build time-distributed residual block."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(8, (3, 3), padding='same')
        )(inputs)
        output = categorizer.tres_block(
            x, filters=16, strides=2, block=False, basic=True
        )
        assert output is not None


class TestSimpleResNet:
    """Tests for simple_resnet() model architecture."""

    def test_builds_resnet_without_classifier(self):
        """Should build ResNet feature extractor."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        output = categorizer.simple_resnet(
            inputs, filters=8, classes=3, level=5, with_classifier=False
        )
        assert output is not None

    def test_builds_resnet_with_classifier(self):
        """Should build ResNet with classifier."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        model = categorizer.simple_resnet(
            inputs, filters=8, classes=3, level=5, with_classifier=True
        )
        assert model is not None
        assert model.output_shape[-1] == 3

    def test_resnet_level_affects_architecture(self):
        """Should use different architectures based on level."""
        categorizer = Categorizers()
        inputs = Input(shape=(32, 32, 3))
        model_low = categorizer.simple_resnet(
            inputs, filters=8, classes=3, level=5, with_classifier=True
        )
        model_high = categorizer.simple_resnet(
            inputs, filters=8, classes=3, level=7, with_classifier=True
        )
        assert len(model_high.layers) != len(model_low.layers)


class TestSimpleTResNet:
    """Tests for simple_tresnet() (Time-distributed ResNet)."""

    def test_builds_tresnet_without_classifier(self):
        """Should build Time-distributed ResNet feature extractor."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        output = categorizer.simple_tresnet(
            inputs, filters=8, classes=3, level=5, with_classifier=False
        )
        assert output is not None

    def test_builds_tresnet_with_classifier(self):
        """Should build Time-distributed ResNet with classifier."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        model = categorizer.simple_tresnet(
            inputs, filters=8, classes=3, level=5, with_classifier=True
        )
        assert model is not None
        assert model.output_shape[-1] == 3

    def test_tresnet_includes_lstm(self):
        """Should include LSTM layer in Time-distributed ResNet."""
        categorizer = Categorizers()
        inputs = Input(shape=(15, 16, 16, 1))
        model = categorizer.simple_tresnet(
            inputs, filters=8, classes=3, level=5, with_classifier=True
        )
        layer_names = [layer.__class__.__name__ for layer in model.layers]
        assert 'LSTM' in layer_names


class TestCombinedNetwork:
    """Tests for combined_network() (Animation Analyzer + Pattern Recognizer)."""

    def test_builds_combined_network(self):
        """Should build combined network with both inputs."""
        categorizer = Categorizers()
        model = categorizer.combined_network(
            time_step=15,
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            classes=3,
            level_tconv=1,
            level_conv=2,
        )
        assert model is not None
        assert len(model.inputs) == 2
        assert model.output_shape[-1] == 3

    def test_combined_network_accepts_two_inputs(self):
        """Should accept both animation and pattern image inputs."""
        categorizer = Categorizers()
        model = categorizer.combined_network(
            time_step=15,
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            classes=2,
            level_tconv=1,
            level_conv=2,
        )
        animation_input = np.random.random((1, 15, 16, 16, 1)).astype('float32')
        pattern_input = np.random.random((1, 32, 32, 3)).astype('float32')
        output = model.predict([animation_input, pattern_input], verbose=0)
        assert output.shape == (1, 1)
        assert 0.0 <= output[0, 0] <= 1.0

    def test_combined_network_binary_classification(self):
        """Should use sigmoid for binary classification."""
        categorizer = Categorizers()
        model = categorizer.combined_network(
            time_step=15,
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            classes=2,
            level_tconv=1,
            level_conv=2,
        )
        assert model.output_shape[-1] == 1

    def test_combined_network_multiclass_classification(self):
        """Should use softmax for multiclass classification."""
        categorizer = Categorizers()
        model = categorizer.combined_network(
            time_step=15,
            dim_tconv=16,
            dim_conv=32,
            channel=1,
            classes=5,
            level_tconv=1,
            level_conv=2,
        )
        assert model.output_shape[-1] == 5