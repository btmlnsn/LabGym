import logging
import pytest

from LabGym import wx_utils
import wx


@pytest.fixture(scope="session")
def wx_app():
    """Session-scope wx.App fixture shared across all test modules.
    
    This fixture ensures that all wx tests share a single app instance, avoiding
    C++ state corruption that occurs when creating multiple apps
    after ShowModal() calls.
    """
    app = wx.App()
    logging.debug(f'Session wx_app fixture created: {app!r}')
    yield app
    # No cleanpu needed - pytest session ends after all tests
    logging.debug('Session wx_app fixture teardown')