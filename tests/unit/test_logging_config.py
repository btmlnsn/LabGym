import logging
from pathlib import Path
import sys

import pytest

from LabGym.config.logging import bootstrap as logging_config
import LabGym.config.core as core

_LOGGING_YAML = Path(logging_config.__file__).with_name("logging.yaml")

# basicConfig here isn't effective, maybe pytest has already configured logging?
# instead, use the root logger's setLevel method
rootlogger = logging.getLogger()
def rootlogger_reset():
	rootlogger.setLevel(logging.DEBUG)
	rootlogger.handlers = []


# success cases
def test_success(monkeypatch):
	# Arrange
	rootlogger_reset()
	assert rootlogger.level == logging.DEBUG
	_config = {
		'logging_configfiles': [_LOGGING_YAML],
		'logging_configfile': None,
		'logging_level': 'INFO',
		}
	monkeypatch.setattr(core, 'get_config', lambda: _config)
	logging.debug('%s: %r', '_config', _config)

	# Act
	logging_config.configure()

	# Assert
	assert rootlogger.level == logging.INFO  # per logging.yaml


# Bad logging_level produces a warning message.
def test_bad_logging_level(monkeypatch):
	# Arrange
	rootlogger_reset()
	assert rootlogger.level == logging.DEBUG
	_config = {
		'logging_configfiles': [_LOGGING_YAML],
		'logging_configfile': None,
		'logging_level': 'WALNUT',  # bad value
		}
	monkeypatch.setattr(core, 'get_config', lambda: _config)
	logging.debug('%s: %r', '_config', _config)

	# Act
	logging_config.configure()

	# Assert
	# WARNING Trouble overriding root logger level.
	assert rootlogger.level == logging.INFO  # per logging.yaml


# Bad specific logging_configfile produces a warning message.
def test_bad_specific_logging_configfile(monkeypatch):
	# Arrange
	rootlogger_reset()
	assert rootlogger.level == logging.DEBUG
	_config = {
		'logging_configfiles': [],
		'logging_configfile': Path('/bravo/charlie.yaml'),
		# 'logging_level': None,
		}
	monkeypatch.setattr(core, 'get_config', lambda: _config)
	logging.debug('%s: %r', '_config', _config)

	# Act
	logrecords = []
	logging_config.configure(logrecords)

	# Assert
	# DEBUG:Unsuitable logging configfile /bravo/charlie.yaml.  ([Errno 2] No such file or directory: '/bravo/charlie.yaml')
	# WARNING:No suitable logging configfile found.
	# WARNING:Trouble overriding root logger level.
	assert rootlogger.level == logging.DEBUG
