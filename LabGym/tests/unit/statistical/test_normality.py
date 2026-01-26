"""
/tests.unit.statistical.test_normality

Tests for the data_mining.normal() method
This method performs Shapiro-Wilk normality testing on datasets
"""

# Related third party imports
import numpy as np
import pandas as pd
import pytest 

# Local application imports
from LabGym.minedata import data_mining


# Helper to create a data_mining instance
def create_data_mining_instance(tmp_path, pval = 0.05):
    """
    Create a data_mining instance with minimal required parameters
    The normal() method only needs self.pval, but __init__ requires result_path
    """

    # Minimal data structure required by __init__
    dummy_data = [{'dummy': {'param': pd.Series([1, 2, 3])}}]

    return data_mining(
        data_in = dummy_data,
        control_in = None,
        paired_in = False,
        result_path_in = str(tmp_path),
        pval_in = pval,
        file_names_in = ['dummy']
    )


# TestNormalMethodBasic includes core functionality tests for the normal() method
class TestNormalMethodBasic:
    """Test basic functionality of the normal() method"""

    def test_normal_distribution_returns_true(self, tmp_path, normal_series_large):
        """
        Data drawn from normal distribution should return True.
        Using large sample (n = 50) for reliable Shapiro-Wilk results
        """

        dm = create_data_mining_instance(tmp_path)
        dataset = [normal_series_large]

        result = dm.normal(dataset)

        assert result is True


    def test_non_normal_distribution_returns_false(self, tmp_path, non_normal_series):
        """
        Data drawn from exponential distribution should return False.
        Exponential distribution is clearly non-normal
        NOTE: Requires 3+ series due to suspected bug in normal() method
        """
        np.random.seed(42)
        
        dm = create_data_mining_instance(tmp_path)
        
        dataset = [
            pd.Series(np.random.exponential(10, 50)),
            pd.Series(np.random.exponential(10, 50)),
            pd.Series(np.random.exponential(10, 50)),
        ]

        result = dm.normal(dataset)

        assert result is False


    def test_multiple_normal_series_returns_true(self, tmp_path):
        """
        Multiple series all from normal distributions should return True
        """
        np.random.seed(42)

        dm = create_data_mining_instance(tmp_path)

        dataset = [
            pd.Series(np.random.normal(100, 10, 50)),
            pd.Series(np.random.normal(50, 5, 50)),
            pd.Series(np.random.normal(200, 20, 50)),
        ]

        result = dm.normal(dataset)

        assert result is True


    def test_mixed_distributions_returns_false(self, tmp_path, normal_series_large, non_normal_series):
        """
        If ANY series is non-normal, overall result should be False.
        This tests the "all must be normal" logic.
        NOTE: Requires 3+ series due to suspectedbug in normal() method
        """
        np.random.seed(42)
        
        dm = create_data_mining_instance(tmp_path)
        
        dataset = [
            pd.Series(np.random.normal(100, 10, 50)),
            pd.Series(np.random.normal(100, 10, 50)),
            pd.Series(np.random.exponential(10, 50)),  # Non-normal
        ]

        result = dm.normal(dataset)

        assert result is False



# TestNormalMethodPvalThreshold includes p-value threshold tests
class TestNormalMethodPvalThreshold:
    """Test that the p-value threshold is correctly applied."""

    def test_stricter_pval_threshold(self, tmp_path):
        """
        With stricter p-value (0.01), borderline data may be classified differently
        """
        np.random.seed(42)

        dm_strict = create_data_mining_instance(tmp_path, pval = 0.01)
        dm_lenient = create_data_mining_instance(tmp_path, pval = 0.10)

        # Creates data that's somewhat normal, but not perfectly
        dataset = [pd.Series(np.random.normal(100, 10, 30))]

        # Both should work (results may vary, but ideally no errors)
        result_strict = dm_strict.normal(dataset)
        result_lenient = dm_lenient.normal(dataset)

        assert isinstance(result_strict, bool)
        assert isinstance(result_lenient, bool)


    def test_default_pval_is_005(self, tmp_path):
        """Verify default p-value threshold is 0.05"""

        dm = create_data_mining_instance(tmp_path)

        assert dm.pval == 0.05

    

# TestNormalMethodEdgeCases includes tests for edge cases and boundary conditions within the normal() method
class TestNormalMethodEdgeCases:
    """Test edge cases for the normal() method"""

    def test_empty_dataset_returns_true(self, tmp_path):
        """
        Empty dataset (no series) should return True.
        The loop doesn't execute, so normal stays True.
        """

        dm = create_data_mining_instance(tmp_path)
        dataset = []

        result = dm.normal(dataset)

        assert result is True

    
    def test_small_sample_size_behavior(self, tmp_path):
        """
        Test behavior with small samples (n < 3)

        NOTE: There may be a bug in the original code - it checks len(dataset) >= 3,
        instead of len(i) >= 3. This test documets current behavior.

        With this potential bug, samples < 3 elements are still tested if dataset has >=3 series,
        which will cause scipy.stats.shapiro to raise an error or warning.
        """

        dm = create_data_mining_instance(tmp_path)

        # Single series with only 2 elements
        # Current code checks len(dataset) and not len(series)
        tiny_series = pd.Series([1.0, 2.0])
        dataset = [tiny_series]

        # With len(dataset)=1 < 3, the shapiro test is skipped entirely
        # SO this returns True (the default)
        result = dm.normal(dataset)
        
        assert result is True


    def test_exactly_three_samples(self, tmp_path):
        """
        Shapiro-Wilk requires at least 3 samples.
        Test with exactly 3 samples per series.
        """
        np.random.seed(42)
        
        dm = create_data_mining_instance(tmp_path)

        # 3 series (len(dataset) = 3), each with 3 values
        dataset = [
            pd.Series(np.random.normal(10, 1, 3)),
            pd.Series(np.random.normal(10, 1, 3)),
            pd.Series(np.random.normal(10, 1, 3)),
        ]

        # Should execute without error
        result = dm.normal(dataset)

        assert isinstance(result, bool)


    def test_identical_values_handling(self, tmp_path, identical_values_series):
        """
        Series with all identical values is technically "perfectly normal"
        but scipy.stats.shapiro may return nan or warning.

        Tests that this edge case is handled
        """

        dm = create_data_mining_instance(tmp_path)

        # Need dataset length >= 3 for the check to run
        dataset = [
            identical_values_series,
            identical_values_series,
            identical_values_series,
        ]

        # This may generate a warning from scipy about zero range
        # scipy.stats.shapiro returns (nan, 1.0) for constant data
        with pytest.warns(UserWarning, match="Input data has range zero"):
            result = dm.normal(dataset)

        #With p-value = 1.0, this should return True (1.0 >= 0.05)
        assert result is True

    
    def test_series_with_nans_not_preprocessed(self, tmp_path, series_with_nans):
        """
        The normal() method does NOT call dropna() - it receives raw series.
        Series with NaNs may cause issues with shapiro test.

        Note: the two_groups() and multiple_groups() methods call dropna()
        before passing to normal(), so this edge case may very well not occur in practice.
        """

        dm = create_data_mining_instance(tmp_path)

        # 3 series to trigger the shapiro check
        dataset = [series_with_nans, series_with_nans, series_with_nans]

        # scipy.stats.shapiro handles NaNs by ignoring them (as of recent versions)
        # or may raise an error in older versions

        result = dm.normal(dataset)
        
        assert isinstance(result, bool)



# TestingNormalMethodDocumentedBug - known potential issue mentioned above
class TestNormalMethodDocumentedBug:
    """
    Document the potential bug in normal() method

    Bug: Line 46 checks `if len(dataset) >= 3` but should check `if len(i) >= 3`

    Current behavior: Shapiro-Wilk is only run if there are 3+ series in dataset,
    regardless of how many samples are in each series.

    Expected behavior (believed to be ideal behavior): Shapiro-Wilk should be run if each series has 3+ samples.
    """

    def test_bug_skips_check_with_few_series(self, tmp_path):
        """
        With only 2 series (even if each has many samples),
        the normality check is skipped entirely.
        """
        np.random.seed(42)

        dm = create_data_mining_instance(tmp_path)

        # 2 series with 100 samples each - clearly enough for Shapiro-Wilk
        # But with len(dataset)=2 < 3, the check is skipped
        dataset = [
            pd.Series(np.random.exponential(10, 100)),  # Non-normal!
            pd.Series(np.random.exponential(10, 100)),  # Non-normal!
        ]

        result = dm.normal(dataset)

        # BUG: Returns True because the check was skipped, not because data is normal
        assert result is True  # This documents the suspected bug


    def test_bug_runs_check_with_tiny_samples_if_many_series(self, tmp_path):
        """
        With 3+ series but only 2 samples each, Shapiro-Wilk issues a SmallSampleWarning.
        The check runs because len(dataset) >= 3, even though samples are too few.
        """

        dm = create_data_mining_instance(tmp_path)

        # 3 series but only 2 samples each - too few for Shapiro-Wilk
        dataset = [
            pd.Series([1.0, 2.0]),
            pd.Series([3.0, 4.0]),
            pd.Series([5.0, 6.0]),
        ]

        # scipy returns NaN p-values with SmallSampleWarning (not UserWarning)
        # NaN < 0.05 is False, so normal stays True
        with pytest.warns(Warning):
            result = dm.normal(dataset)

        # Returns True because NaN comparison returns False
        assert result is True



@pytest.mark.xfail(
    reason=(
        "Spec (extended guide §2.3.2) implies Shapiro should assess normality "
        "even for two groups, but normal() only runs Shapiro when len(dataset) >= 3, "
        "so a single non-normal series returns True."
    )
)
def test_single_non_normal_series_should_be_detected_spec(tmp_path, non_normal_series):
    """
    SPEC TEST: what would be ideal behavior once normal() is fixed.

    For a single clearly non-normal series, normal() should return False.
    Current implementation returns True because Shapiro is never run when
    len(dataset) < 3.
    """

    dm = create_data_mining_instance(tmp_path)
    dataset = [non_normal_series]

    result = dm.normal(dataset)

    assert result is False
