"""
LabGym.tests.unit.test_selftest

The selftest feature allows testing a (future) bundled PyInstaller build and
running the package's tests on itself (e.g. LabGym --selftest)
"""

# Standard library imports
import logging

# Related third party imports
from unittest.mock import MagicMock, patch
import pytest
import wx

pytestmark = pytest.mark.gui

# Local application/library specific imports
from LabGym import selftest

logger = logging.getLogger(__name__)


def test_selftest_module_exports():
    """Selftest module exposes the expected public API."""
    assert hasattr(selftest, 'run_selftests')
    assert hasattr(selftest, 'run_selftests_help')
    assert callable(selftest.run_selftests)
    assert callable(selftest.run_selftests_help)


@patch.object(selftest.selftest, 'pytest')
def test_run_selftest_returns_0_when_all_pass(mock_pytest):
    """When all pytest runs pass, run_selftests returns 0."""
    mock_pytest.main.return_value = 0

    result = selftest.run_selftests()

    assert result == 0
    assert mock_pytest.main.call_count >= 1


@patch.object(selftest.selftest, 'pytest')
def test_run_selftest_returns_1_when_some_fail(mock_pytest):
    """When any pytest run fails, run_selftests returns 1."""
    # First call (this file) passes, second (LabGym.tests) fails
    mock_pytest.main.side_effect = [0, 1]
    
    result = selftest.run_selftests()

    assert result == 1


@patch.object(selftest.selftest, 'pytest')
def test_run_selftests_invokes_pytest_with_pyargs(mock_pytest):
    """run_selftests invokes pytest with --pyargs LabGym.tests for package tests."""
    mock_pytest.main.return_value = 0

    selftest.run_selftests()

    # At least one call should be with --pyargs and LabGym.tests
    call_args_list = [call[0][0] for call in mock_pytest.main.call_args_list]

    assert any('--pyargs' in args and 'LabGym.tests' in args for args in call_args_list)


@patch.object(selftest.selftest, 'mywx')
def test_run_selftests_help_does_not_raise(mock_mywx):
    """run_selftests_help runs without raising (dialog is mocked)"""
    mock_dlg = MagicMock()
    mock_dlg.__enter__ = MagicMock(return_value = mock_dlg)
    mock_dlg.__exit__ = MagicMock(return_value = None)
    mock_dlg.ShowModal.return_value = wx.ID_OK
    mock_mywx.OK_Cancel_Dialog.return_value = mock_dlg

    selftest.run_selftests_help()

    mock_mywx.OK_Cancel_Dialog.assert_called_once()
    mock_dlg.ShowModal.assert_called_once()

