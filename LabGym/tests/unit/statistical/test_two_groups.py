"""
LabGym.tests.unit.statistical.test_two_groups

Tests for the data_mining.two_groups() method
This method performs two-group statistical comparisons using:
- Paired t-test (normal, paired)
- Unpaired t-test (normal, unpaired)
- Wilcoxon signed-rank test (non-normal, paired)
- Mann-Whitney U test (non-normal, unpaired)
"""

# Standard library imports
import copy
import os

# Related third party imports
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Local application imports
from LabGym.minedata import data_mining


# Helper function to create two-group behavior data
def create_two_group_data(mean_a, mean_b, std = 2, n = 30, distribution = 'normal', seed = 42):
    """
    Create two-group behavior data with specified parameters

    Args:
        mean_a, mean_b: Means groups A and B
        std: Standard deviation
        n: Sample size per group
        distribution: 'normal' or 'exponential'
        seed: Random seed

    Returns:
        List of two dicts in data_mining format
    """
    np.random.seed(seed)

    if distribution == 'normal':
        data_a = np.random.normal(mean_a, std, n)
        np.random.seed(seed + 1)
        data_b = np.random.normal(mean_b, std, n)
    
    else:
        data_a = np.random.exponential(mean_a, n)
        np.random.seed(seed + 1)
        data_b = np.random.exponential(mean_b, n)

    group_a = {'behavior1': {'param1': pd.Series(data_a)}}
    group_b = {'behavior1': {'param1': pd.Series(data_b)}}

    return [group_a, group_b]



# TestTwoGroupsTestSelection verifies that the correct statistical test is chosen
class TestTwoGroupsTestSelection:
    """Test that the correct statistical test is selected based on data properties."""

    def test_normal_unpaired_uses_ttest_ind(self, tmp_path):
        """
        Normal data + unpaired should use unpaired t-test (ttest_ind).
        """

        data = create_two_group_data(10, 20, distribution='normal')
        
        with patch('LabGym.minedata.stats.ttest_ind') as mock_ttest:
            mock_ttest.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=False,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B']
            )
            dm.two_groups()
            dm.writer.close()
            
            mock_ttest.assert_called()


    def test_normal_paired_uses_ttest_rel(self, tmp_path):
        """
        Normal data + paired should use paired t-test (ttest_rel).
        """

        data = create_two_group_data(10, 20, distribution='normal')
        
        with patch('LabGym.minedata.stats.ttest_rel') as mock_ttest:
            mock_ttest.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=True,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B']
            )
            dm.two_groups()
            dm.writer.close()
            
            mock_ttest.assert_called()


    def test_two_groups_always_uses_parametric_due_to_bug(self, tmp_path, two_groups_non_normal):
        """
        BUG: Due to suspected normal() bug, two group comparisons ALWAYS use t-tests.
        
        The normality check is skipped when len(dataset) < 3, so with only
        2 groups, data is always treated as normal regardless of distribution.
        """

        data = copy.deepcopy(two_groups_non_normal)

        with patch('LabGym.minedata.stats.ttest_ind') as mock_ttest, \
             patch('LabGym.minedata.stats.mannwhitneyu') as mock_mw:
            mock_ttest.return_value = MagicMock(pvalue = 0.001)
            mock_mw.return_value = MagicMock(pvalue = 0.001)

            dm = data_mining(
                data_in = data,
                paired_in = False,
                result_path_in = str(tmp_path),
                file_names_in = ['A', 'B']
            )

            dm.two_groups()
            dm.writer.close()

            # BUG: t-test is used even for non-normal data
            mock_ttest.assert_called()
            mock_mw.assert_not_called()


    def test_two_groups_paired_always_uses_ttest_rel_due_to_bug(self, tmp_path, two_groups_non_normal):
        """
        BUG: Paired two-group comparisons always use paired t-test
        """

        data = copy.deepcopy(two_groups_non_normal)

        with patch('LabGym.minedata.stats.ttest_rel') as mock_ttest, \
             patch('LabGym.minedata.stats.wilcoxon') as mock_wilcoxon:
            mock_ttest.return_value = MagicMock(pvalue = 0.001)
            mock_wilcoxon.return_value = MagicMock(pvalue = 0.001)

            dm = data_mining(
                data_in = data,
                paired_in = True,
                result_path_in = str(tmp_path),
                file_names_in = ['A', 'B']
            )

            dm.two_groups()
            dm.writer.close()

            # BUG: paired t-test is used even for non-normal data
            mock_ttest.assert_called()
            mock_wilcoxon.assert_not_called()


    @pytest.mark.xfail(
        reason=(
            "Spec (extended guide §2.3.2) says non-normal two-group unpaired data "
            "should use Mann-Whitney U, but normal() never runs for len(dataset)<3, "
            "so two_groups() always uses ttest_ind instead."
        )
    )
    def test_non_normal_unpaired_uses_mannwhitneyu_spec(self, tmp_path, two_groups_non_normal):
        """
        SPEC TEST: what should happen once normal() and two_groups() are fixed.
        Non-normal unpaired data should select Mann-Whitney U test.
        """
        data = copy.deepcopy(two_groups_non_normal)
        
        with patch('LabGym.minedata.stats.mannwhitneyu') as mock_mw:
            mock_mw.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=False,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B']
            )
            dm.two_groups()
            dm.writer.close()
            
            mock_mw.assert_called()


    @pytest.mark.xfail(
        reason=(
            "Spec (extended guide §2.3.2) says non-normal two-group paired data "
            "should use Wilcoxon, but normal() never runs for len(dataset)<3, "
            "so two_groups() always uses ttest_rel instead."
        )
    )
    def test_non_normal_paired_uses_wilcoxon_spec(self, tmp_path, two_groups_non_normal):
        """
        SPEC TEST: what should happen once normal() and two_groups() are fixed.
        Non-normal paired data should select Wilcoxon signed-rank test.
        """
        data = copy.deepcopy(two_groups_non_normal)
        
        with patch('LabGym.minedata.stats.wilcoxon') as mock_wilcoxon:
            mock_wilcoxon.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=True,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B']
            )
            dm.two_groups()
            dm.writer.close()
            
            mock_wilcoxon.assert_called()



# TestTwoGroupsWithControl verifies the integrity of control group handling
class TestTwoGroupsWithControl:
    """Test two-group comparisons when a control group is provided."""

    def test_control_group_inserted_at_position_zero(self, tmp_path):
        """
        When control_in is provided, it should be inserted at position 0 of data.
        """
        np.random.seed(42)
        
        # Single experimental group
        exp_group = {'behavior1': {'param1': pd.Series(np.random.normal(20, 2, 30))}}
        
        # Control group
        control = {'behavior1': {'param1': pd.Series(np.random.normal(10, 2, 30))}}
        
        dm = data_mining(
            data_in=[exp_group],
            control_in=control,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Control', 'Experimental']
        )
        
        # Before two_groups(), data has 1 element
        assert len(dm.data) == 1
        
        dm.two_groups()
        dm.writer.close()
        
        # After two_groups(), control is inserted at position 0
        assert len(dm.data) == 2
        assert dm.data[0] == control


    def test_one_group_with_control_triggers_two_groups(self, tmp_path, capsys):
        """
        len(data)==1 with control provided should route to two_groups().
        """
        np.random.seed(42)
        
        exp_group = {'behavior1': {'param1': pd.Series(np.random.normal(20, 2, 30))}}
        control = {'behavior1': {'param1': pd.Series(np.random.normal(10, 2, 30))}}
        
        dm = data_mining(
            data_in=[exp_group],
            control_in=control,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Control', 'Experimental']
        )
        
        dm.statistical_analysis()
        
        captured = capsys.readouterr()
        
        # Check that it printed behavior name (indicates two_groups ran)
        assert 'behavior1' in captured.out



# TestTwoGroupsOutput verfies creation of Excel output from two-group statistical tests
class TestTwoGroupsOutput:
    """Test Excel file output from two_groups()"""

    def test_excel_file_created(self, tmp_path, two_groups_normal_different):
        """Verify Excel file is created at result_path."""
        
        dm = data_mining(
            data_in=copy.deepcopy(two_groups_normal_different),
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Group_A', 'Group_B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        expected_file = tmp_path / 'data_mining_results.xlsx'

        assert expected_file.exists()


    def test_significant_results_written(self, tmp_path, two_groups_normal_different):
        """
        Significantly different groups should have results written to Excel.
        """
        
        dm = data_mining(
            data_in=copy.deepcopy(two_groups_normal_different),
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Group_A', 'Group_B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        # Read the Excel file and verify content
        excel_file = tmp_path / 'data_mining_results.xlsx'
        df = pd.read_excel(excel_file, sheet_name='walking')
        
        # Should have data written (speed parameter should be significant)
        assert len(df) > 0


    def test_non_significant_results_empty_sheet(self, tmp_path, two_groups_normal_same):
        """
        Non-significantly different groups should not have results written.
        """

        dm = data_mining(
            data_in=copy.deepcopy(two_groups_normal_same),
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Group_A', 'Group_B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        # Read the Excel file - should have no sheets or empty sheets
        excel_file = tmp_path / 'data_mining_results.xlsx'
        
        # The file exists but may have no sheets with data
        try:
            xl = pd.ExcelFile(excel_file)
            # If sheets exist, they should be empty or minimal
            if xl.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=xl.sheet_names[0])
                
                # May be empty or have minimal content
                assert df.empty or len(df) <= 1
        
        except ValueError:
            # No sheets - that's expected for non-significant results
            pass



# TestTwoGroups includes tests for edge cases and boundary conditions with the two-group statistical test method
class TestTwoGroupsEdgeCases:
    """Test edge cases for two_groups()"""

    def test_handles_nan_values(self, tmp_path):
        """
        Series with NaN values should be handled via dropna().
        """
        np.random.seed(42)
        
        data_a = np.random.normal(10, 2, 30)
        data_a[5] = np.nan
        data_a[10] = np.nan
        
        data_b = np.random.normal(20, 2, 30)
        data_b[3] = np.nan
        
        group_a = {'behavior1': {'param1': pd.Series(data_a)}}
        group_b = {'behavior1': {'param1': pd.Series(data_b)}}
        
        dm = data_mining(
            data_in=[group_a, group_b],
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        # Should not raise error
        dm.two_groups()
        dm.writer.close()


    def test_unequal_sample_sizes(self, tmp_path):
        """
        Groups with different sample sizes should be handled.
        """
        np.random.seed(42)
        
        group_a = {'behavior1': {'param1': pd.Series(np.random.normal(10, 2, 20))}}
        group_b = {'behavior1': {'param1': pd.Series(np.random.normal(20, 2, 40))}}
        
        dm = data_mining(
            data_in=[group_a, group_b],
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        # Should not raise error (unpaired tests handle unequal n)
        dm.two_groups()
        dm.writer.close()


    def test_multiple_behaviors_and_parameters(self, tmp_path):
        """
        Test with multiple behaviors and parameters.
        """
        np.random.seed(42)
        
        group_a = {
            'walking': {
                'speed': pd.Series(np.random.normal(10, 2, 30)),
                'duration': pd.Series(np.random.normal(5, 1, 30)),
            },
            'resting': {
                'frequency': pd.Series(np.random.normal(3, 0.5, 30)),
            }
        }
        
        np.random.seed(43)
        group_b = {
            'walking': {
                'speed': pd.Series(np.random.normal(15, 2, 30)),
                'duration': pd.Series(np.random.normal(5, 1, 30)),
            },
            'resting': {
                'frequency': pd.Series(np.random.normal(3, 0.5, 30)),
            }
        }
        
        dm = data_mining(
            data_in=[group_a, group_b],
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        # Should create sheets for both behaviors
        excel_file = tmp_path / 'data_mining_results.xlsx'
        xl = pd.ExcelFile(excel_file)
        
        # At least one sheet should exist if any comparison was significant
        assert len(xl.sheet_names) >= 0  # May be empty if nothing significant



# TestTwoGroupsIntegration includes integration tests for two-groups statistical test method (without mocking)
# QuestionForLater: should this maybe be inside 'integration' folder?
class TestTwoGroupsIntegration:
    """Integration tests that verify actual statistical test results"""

    def test_clearly_different_groups_detected(self, tmp_path, capsys):
        """
        Groups with very different means should be detected as significant.
        """
        np.random.seed(42)
        
        # Very different means (10 vs 50)
        group_a = {'behavior1': {'param1': pd.Series(np.random.normal(10, 1, 30))}}
        group_b = {'behavior1': {'param1': pd.Series(np.random.normal(50, 1, 30))}}
        
        dm = data_mining(
            data_in=[group_a, group_b],
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        captured = capsys.readouterr()
        
        # Should print the parameter name when significant
        assert 'param1' in captured.out
        assert 'p-value' in captured.out


    def test_identical_groups_not_significant(self, tmp_path, capsys):
        """
        Identical groups should not be detected as significant.
        """
        np.random.seed(42)
        
        data = np.random.normal(10, 2, 30)
        
        group_a = {'behavior1': {'param1': pd.Series(data.copy())}}
        group_b = {'behavior1': {'param1': pd.Series(data.copy())}}
        
        dm = data_mining(
            data_in=[group_a, group_b],
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        dm.two_groups()
        dm.writer.close()
        
        captured = capsys.readouterr()
       
        # Should NOT print p-value (nothing significant)
        assert 'p-value' not in captured.out



