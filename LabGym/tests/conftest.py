"""
/tests/conftest.py

Shared pytest fixtures for LabGym tests
"""

# Standard library imports
import sys

# Related third party imports
import pytest

@pytest.fixture(scope="session")
def wx_app():
    from LabGym import mywx
    import wx
    app = wx.App()
    yield app


@pytest.fixture
def mock_argv(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['LabGym'])
