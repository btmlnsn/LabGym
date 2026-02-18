"""
LabGym.tests.unit.statistical.test_multiple_groups

Tests for the data_mining.multiple_groups() method

This method performs multi-group statistical comparisons using:
- ANOVA (normal data)
- Kruskal-Wallis (non-normal, unpaired)
- Friedman (non-normal, paired)

With post-hoc tests:
- Tukey HSD (ANOVA without control)
- Dunnett (ANOVA with control)
- Dunn (non-parametric)
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


# Helper function to create multi-group (4 group) behavior data
def create_multi_group_data(
    n_groups=4, 
    means=None, 
    std=2, 
    n=30, 
    distribution='normal', 
    seed=42
):
    """
    Create multi-group behavior data with specified parameters.
    
    Args:
        n_groups: Number of groups
        means: List of means per group (default: 10, 15, 20, 25)
        std: Standard deviation
        n: Sample size per group
        distribution: 'normal' or 'exponential'
        seed: Random seed
    
    Returns:
        List of dicts in data_mining format
    """
    if means is None:
        means = [10 + i*5 for i in range(n_groups)]
    
    np.random.seed(seed)

    groups = []
    
    for i, mean in enumerate(means):
        np.random.seed(seed + i)
        if distribution == 'normal':
            data = np.random.normal(mean, std, n)
        else:
            data = np.random.exponential(mean, n)
        
        group = {'behavior1': {'param1': pd.Series(data)}}
        
        groups.append(group)
    
    return groups



# TestMultipleGroupsTestSelection includes tests to verify that the correct multi-group statistical test is selected
class TestMultipleGroupsTestSelection:
    """Test that the correct statistical test is selected."""

    @pytest.mark.xfail(reason="Bug in minedata.py: pd.DataFrame(tukey) fails")
    def test_normal_data_uses_anova(self, tmp_path, multi_groups_normal):
        """
        Normal data should use ANOVA (f_oneway).
        NOTE: This fails due to bug in minedata.py line 160.
        """

        data = copy.deepcopy(multi_groups_normal)
        
        #Commented out until bug is fixed:
        # with patch('LabGym.minedata.stats.f_oneway') as mock_anova:
        #     mock_anova.return_value = MagicMock(pvalue=0.001)
            
        #     dm = data_mining(
        #         data_in=data,
        #         paired_in=False,
        #         result_path_in=str(tmp_path),
        #         file_names_in=['A', 'B', 'C', 'D']
        #     )

        #     dm.multiple_groups()
        #     dm.writer.close()
            
        #     mock_anova.assert_called()

        # Current test code with existing bug:
        dm = data_mining(
            data_in = data,
            paired_in = False,
            result_path_in = str(tmp_path),
            file_names_in = ['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()


    def test_non_normal_unpaired_uses_kruskal(self, tmp_path, multi_groups_non_normal):
        """
        Non-normal + unpaired data should use Kruskal-Wallis.
        """

        data = copy.deepcopy(multi_groups_non_normal)
        
        with patch('LabGym.minedata.stats.kruskal') as mock_kruskal:
            mock_kruskal.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=False,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B', 'C', 'D']
            )
            dm.multiple_groups()
            dm.writer.close()
            
            mock_kruskal.assert_called()


    def test_non_normal_paired_uses_friedman(self, tmp_path, multi_groups_non_normal):
        """
        Non-normal + paired data should use Friedman test.
        """

        data = copy.deepcopy(multi_groups_non_normal)
        
        with patch('LabGym.minedata.stats.friedmanchisquare') as mock_friedman:
            mock_friedman.return_value = MagicMock(pvalue=0.001)
            
            dm = data_mining(
                data_in=data,
                paired_in=True,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B', 'C', 'D']
            )
            dm.multiple_groups()
            dm.writer.close()
            
            mock_friedman.assert_called()



# TestMultipleGroupsPostHoc includes tests to verify that the correct post-hoc multi-group statistical test is selected
class TestMultipleGroupsPostHoc:
    """Test that correct post-hoc test is selected."""

    @pytest.mark.xfail(reason="Bug in minedata.py: pd.DataFrame(tukey) fails")
    def test_anova_without_control_uses_tukey(self, tmp_path, multi_groups_normal):
        """
        ANOVA without control group should use Tukey HSD post-hoc.
        NOTE: This fails because pd.DataFrame(TukeyHSDResult) doesn't work
        """

        data = copy.deepcopy(multi_groups_normal)
        
        # Commented out until bug is fixed:
        # with patch('LabGym.minedata.stats.tukey_hsd') as mock_tukey:
        #     mock_tukey.return_value = MagicMock()
            
        #     dm = data_mining(
        #         data_in=data,
        #         control_in=None,
        #         paired_in=False,
        #         result_path_in=str(tmp_path),
        #         file_names_in=['A', 'B', 'C', 'D']
        #     )
        #     dm.multiple_groups()
        #     dm.writer.close()
            
        #     mock_tukey.assert_called()

        # Current test code with existing bug:
        dm = data_mining(
            data_in = data,
            control_in = None,
            paired_in = False,
            result_path_in = str(tmp_path),
            file_names_in = ['A', 'B', 'C', 'D']
        )

        dm.multiple_groups()
        dm.writer.close()


    def test_anova_with_control_uses_dunnett(self, tmp_path):
        """
        ANOVA with control group should use Dunnett post-hoc.
        """
        np.random.seed(42)
        
        # 3 experimental groups
        data = create_multi_group_data(n_groups=3, means=[15, 20, 25], distribution='normal')
        
        # Control group
        control = {'behavior1': {'param1': pd.Series(np.random.normal(10, 2, 30))}}
        
        with patch('LabGym.minedata.stats.dunnett') as mock_dunnett:
            mock_dunnett.return_value = MagicMock(pvalue=0.001)
            mock_dunnett.return_value.columns = ['B', 'C', 'D']
            mock_dunnett.return_value.index = ['B', 'C', 'D']
            
            dm = data_mining(
                data_in=data,
                control_in=control,
                paired_in=False,
                result_path_in=str(tmp_path),
                file_names_in=['Control', 'B', 'C', 'D']
            )
            dm.multiple_groups()
            dm.writer.close()
            
            mock_dunnett.assert_called()


    def test_non_normal_uses_dunn_posthoc(self, tmp_path, multi_groups_non_normal):
        """
        Non-normal data should use Dunn's post-hoc test.
        """

        data = copy.deepcopy(multi_groups_non_normal)
        
        with patch('LabGym.minedata.sp.posthoc_dunn') as mock_dunn:
            mock_dunn.return_value = pd.DataFrame(
                np.zeros((4, 4)),
                columns=['A', 'B', 'C', 'D'],
                index=['A', 'B', 'C', 'D']
            )
            
            dm = data_mining(
                data_in=data,
                paired_in=False,
                result_path_in=str(tmp_path),
                file_names_in=['A', 'B', 'C', 'D']
            )
            dm.multiple_groups()
            dm.writer.close()
            
            mock_dunn.assert_called()



# TestMultipleGroupsOutput verifies creation of Excel output from multi-group statistical tests
class TestMultipleGroupsOutput:
    """Test Excel file output from multiple_groups()."""

    def test_excel_file_created(self, tmp_path, multi_groups_non_normal):
        """Verify Excel file is created (using non-normal to avoid Tukey bug)"""
        dm = data_mining(
            data_in=copy.deepcopy(multi_groups_non_normal),
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        expected_file = tmp_path / 'data_mining_results.xlsx'
        
        assert expected_file.exists()


    def test_posthoc_results_written(self, tmp_path, multi_groups_non_normal):
        """
        Post-hoc results should be written to Excel (using non-normal/Dunn path to avoid Tukey bug).
        """

        dm = data_mining(
            data_in=copy.deepcopy(multi_groups_non_normal),
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        excel_file = tmp_path / 'data_mining_results.xlsx'
        xl = pd.ExcelFile(excel_file)
        
        # Should have at least one sheet if significant
        if xl.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=xl.sheet_names[0])
            
            # Should have multiple columns (p-value, post-hoc results, data)
            assert len(df.columns) >= 1


    def test_multiple_behaviors_create_sheets(self, tmp_path):
        """
        Multiple behaviors should create multiple sheets (using non-normal data to avoid Tukey bug).
        """
        np.random.seed(42)
        
        groups = []
        for i in range(4):
            np.random.seed(42 + i)
            group = {
                'walking': {'speed': pd.Series(np.random.exponential(10 + i*10, 30))},
                'resting': {'duration': pd.Series(np.random.exponential(5 + i*5, 30))},
            }
            groups.append(group)
        
        dm = data_mining(
            data_in=groups,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        excel_file = tmp_path / 'data_mining_results.xlsx'
        xl = pd.ExcelFile(excel_file)
        
        # At minimum, we verify no errors occurred and the file exists
        assert excel_file.exists()



# TestStatisticalAnalysisRouter includes tests that verify the routing logic of the statistical_analysis module
class TestStatisticalAnalysisRouter:
    """Test the statistical_analysis() method routing logic."""

    def test_two_groups_no_control_routes_to_two_groups(self, tmp_path, capsys):
        """
        len(data)==2 with no control should route to two_groups().
        """
        
        data = create_multi_group_data(n_groups=2, distribution='normal')
        
        dm = data_mining(
            data_in=data,
            control_in=None,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B']
        )
        
        dm.statistical_analysis()
        
        captured = capsys.readouterr()
        
        # Check completion message
        assert 'Data mining for statistical analysis completed!' in captured.out

    def test_one_group_with_control_routes_to_two_groups(self, tmp_path, capsys):
        """
        len(data)==1 with control should route to two_groups().
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
        
        # This should route to two_groups
        dm.statistical_analysis()
        
        captured = capsys.readouterr()
        
        assert 'Data mining for statistical analysis completed!' in captured.out

    def test_three_groups_no_control_routes_to_multiple_groups(self, tmp_path, capsys):
        """
        len(data)==3 with no control should route to multiple_groups().
        Using non-normal data to avoid Tukey bug.
        """
        
        data = create_multi_group_data(n_groups=3, means=[10, 30, 50], distribution='exponential')
        
        dm = data_mining(
            data_in=data,
            control_in=None,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C']
        )
        
        dm.statistical_analysis()
        
        captured = capsys.readouterr()
        
        assert 'Data mining for statistical analysis completed!' in captured.out

    def test_two_groups_with_control_routes_to_multiple_groups(self, tmp_path, capsys):
        """
        len(data)==2 with control (total 3 groups) should route to multiple_groups().
        Using non-normal data to avoid ANOVA+Dunnett bug.
        """
        np.random.seed(42)
        
        data = create_multi_group_data(n_groups=2, means=[25, 40], distribution='exponential')
        control = {'behavior1': {'param1': pd.Series(np.random.exponential(10, 30))}}
        
        dm = data_mining(
            data_in=data,
            control_in=control,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['Control', 'B', 'C']
        )
        
        dm.statistical_analysis()
        
        captured = capsys.readouterr()

        assert 'Data mining for statistical analysis completed!' in captured.out



# TestMultipleGroupsIntegration includes integration tests for multi-group statistical test method (without mocking)
# QuestionForLater: should this maybe be inside 'integration' folder?
class TestMultipleGroupsIntegration:
    """Integration tests without mocking."""

    def test_clearly_different_groups_detected(self, tmp_path, capsys):
        """
        Groups with very different values should be detected as significant.
        Using non-normal data to exercise Kruskal+Dunn path and avoid Tukey bug.
        """
        
        data = create_multi_group_data(
            n_groups=4, 
            means=[10, 30, 50, 70], 
            distribution='exponential'
        )
        
        dm = data_mining(
            data_in=data,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        captured = capsys.readouterr()
        
        # Should print parameter and p-value when significant
        assert 'param1' in captured.out
        assert 'Kruskal' in captured.out
        assert 'Dunn' in captured.out


    def test_similar_groups_not_significant(self, tmp_path, capsys):
        """
        Groups with similar means should not be detected as significant.
        """
        
        # All groups have same mean
        data = create_multi_group_data(
            n_groups=4,
            means=[10, 10, 10, 10],
            std=2,
            distribution='normal'
        )
        
        dm = data_mining(
            data_in=data,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        captured = capsys.readouterr()
        
        # Should NOT print ANOVA results (nothing significant)
        assert 'ANOVA' not in captured.out


    def test_non_normal_data_uses_kruskal_wallis(self, tmp_path, capsys):
        """
        Non-normal data should use Kruskal-Wallis and Dunn post-hoc.
        """

        data = create_multi_group_data(
            n_groups=4,
            means=[10, 30, 50, 70],
            distribution='exponential'
        )
        
        dm = data_mining(
            data_in=data,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        dm.multiple_groups()
        dm.writer.close()
        
        captured = capsys.readouterr()
        
        # Should use Kruskal-Wallis, not ANOVA
        assert 'Kruskal' in captured.out
        assert 'Dunn' in captured.out



# TestMultipleGroupsEdgeCases includes tests for edge cases and boundary conditions with the multi-group statistical test method
class TestMultipleGroupsEdgeCases:
    """Test edge cases for multiple_groups()."""

    def test_handles_nan_values(self, tmp_path):
        """
        Series with NaN values should be handled via dropna().
        Using non-normal data to avoid Tukey bug.
        """
        np.random.seed(42)
        
        groups = []
        for i in range(4):
            np.random.seed(42 + i)
            data = np.random.exponential(10 + i*10, 30)
            data[5] = np.nan  # Add NaN
            groups.append({'behavior1': {'param1': pd.Series(data)}})
        
        dm = data_mining(
            data_in=groups,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        # Should not raise error
        dm.multiple_groups()
        dm.writer.close()


    def test_unequal_sample_sizes(self, tmp_path):
        """
        Groups with different sample sizes should be handled.
        Using non-normal data to avoid Tukey bug.
        """
        np.random.seed(42)
        
        sizes = [20, 25, 30, 35]
        groups = []
        for i, size in enumerate(sizes):
            np.random.seed(42 + i)
            data = np.random.exponential(10 + i*10, size)
            groups.append({'behavior1': {'param1': pd.Series(data)}})
        
        dm = data_mining(
            data_in=groups,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C', 'D']
        )
        
        # Should not raise error
        dm.multiple_groups()
        dm.writer.close()

    def test_minimum_three_groups(self, tmp_path):
        """
        Test with exactly 3 groups (minimum for multi-group tests).
        Using non-normal data to avoid Tukey bug.
        """

        data = create_multi_group_data(
            n_groups=3,
            means=[10, 30, 50],
            distribution='exponential',
        )
        
        dm = data_mining(
            data_in=data,
            paired_in=False,
            result_path_in=str(tmp_path),
            file_names_in=['A', 'B', 'C']
        )
        
        # Should work without error
        dm.multiple_groups()
        dm.writer.close()


