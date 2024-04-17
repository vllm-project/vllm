import json
import logging
import os
import sys
import tempfile
from json.decoder import JSONDecodeError
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from vllm.logger import (_DATE_FORMAT, _FORMAT, _configure_vllm_root_logger,
                         enable_trace_function_call, init_logger)
from vllm.logging import NewLineFormatter


def f1(x):
    return f2(x)


def f2(x):
    return x


def test_trace_function_call():
    fd, path = tempfile.mkstemp()
    cur_dir = os.path.dirname(__file__)
    enable_trace_function_call(path, cur_dir)
    f1(1)
    with open(path, 'r') as f:
        content = f.read()

    assert "f1" in content
    assert "f2" in content
    sys.settrace(None)
    os.remove(path)


def test_default_vllm_root_logger_configuration():
    """This test presumes that VLLM_CONFIGURE_LOGGING and
    VLLM_LOGGING_CONFIG_PATH are not configured and default behavior is
    activated"""
    logger = logging.getLogger("vllm")
    assert logger.level == logging.DEBUG
    assert not logger.propagate

    handler = logger.handlers[0]
    assert handler.stream == sys.stdout
    assert handler.level == logging.INFO

    formatter = handler.formatter
    assert formatter is not None
    assert isinstance(formatter, NewLineFormatter)
    assert formatter._fmt == _FORMAT
    assert formatter.datefmt == _DATE_FORMAT


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
@patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH", None)
def test_init_logger_configures_the_logger_like_the_root_logger():
    """This test requires VLLM_CONFIGURE_LOGGING to be enabled.
    VLLM_LOGGING_CONFIG_PATH may be configured, but is presumed to be
    unimpactful since a random logger name is used for testing."""
    root_logger = logging.getLogger("vllm")
    unique_name = str(uuid4())
    logger = init_logger(unique_name)

    assert logger.name == unique_name
    assert logger.level == root_logger.level
    assert logger.handlers == root_logger.handlers
    assert not logger.propagate


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 0)
@patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH", None)
def test_logger_configuring_can_be_disabled():
    logger_name = unique_name()
    assert logger_name not in logging.Logger.manager.loggerDict
    logger = init_logger(logger_name)
    assert logger_name in logging.Logger.manager.loggerDict

    assert logger.name == logger_name
    assert len(logger.handlers) == 0


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
@patch(
    "vllm.logger.VLLM_LOGGING_CONFIG_PATH",
    "/if/there/is/a/file/here/then/you/did/this/to/yourself.json",
)
def test_an_error_is_raised_when_custom_logging_config_file_does_not_exist():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    with pytest.raises(RuntimeError) as ex_info:
        _configure_vllm_root_logger()
    assert ex_info.type == RuntimeError
    assert "File does not exist" in str(ex_info)


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
def test_an_error_is_raised_when_custom_logging_config_is_invalid_json():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write("---\nloggers: []\nversion: 1")
        logging_config_file.flush()
        with patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH",
                   logging_config_file.name):
            with pytest.raises(JSONDecodeError) as ex_info:
                _configure_vllm_root_logger()
            assert ex_info.type == JSONDecodeError
            assert "Expecting value" in str(ex_info)


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
@pytest.mark.parametrize("unexpected_config", (
    "Invalid string",
    [{
        "version": 1,
        "loggers": []
    }],
    0,
))
def test_an_error_is_raised_when_custom_logging_config_is_unexpected_json(
    unexpected_config: Any, ):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(unexpected_config))
        logging_config_file.flush()
        with patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH",
                   logging_config_file.name):
            with pytest.raises(ValueError) as ex_info:
                _configure_vllm_root_logger()
            assert ex_info.type == ValueError
            assert "Invalid logging config. Expected Dict, got" in str(ex_info)


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
def test_custom_logging_config_is_parsed_and_used_when_provided():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""
    valid_logging_config = {
        "loggers": {
            "vllm.test_logger.logger": {
                "handlers": [],
                "propagate": False,
            }
        },
        "version": 1
    }
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(valid_logging_config))
        logging_config_file.flush()
        with patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH",
                   logging_config_file.name), patch(
                       "logging.config.dictConfig") as dict_config_mock:
            _configure_vllm_root_logger()
            assert dict_config_mock.called_with(valid_logging_config)


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
def test_init_logger_does_not_configure_loggers_configured_by_logging_config():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, the call is not intercepted, but should only impact a
    logger only known to this test."""
    logger_name = f"vllm.test_logger.{unique_name()}"
    valid_logging_config = {
        "loggers": {
            logger_name: {
                "handlers": [],
                "level": "INFO",
                "propagate": True,
            }
        },
        "version": 1
    }
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(valid_logging_config))
        logging_config_file.flush()
        with patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH",
                   logging_config_file.name):
            _configure_vllm_root_logger()
        root_logger = logging.getLogger("vllm")
        test_logger = logging.getLogger(logger_name)
        assert len(test_logger.handlers) == 0
        assert len(root_logger.handlers) > 0
        assert test_logger.level == logging.INFO
        assert test_logger.level != root_logger.level
        assert test_logger.propagate

        # Make sure auto-configuration of other loggers still works
        other_logger = init_logger("vllm.test_logger.other")
        assert other_logger.handlers == root_logger.handlers
        assert other_logger.level == root_logger.level
        assert not other_logger.propagate


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 0)
def test_custom_logging_config_can_be_used_even_if_configure_logging_is_off():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""
    valid_logging_config = {
        "loggers": {
            "vllm.test_logger.logger": {
                "handlers": [],
            }
        },
        "version": 1
    }
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(valid_logging_config))
        logging_config_file.flush()
        with patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH",
                   logging_config_file.name), patch(
                       "logging.config.dictConfig") as dict_config_mock:
            _configure_vllm_root_logger()
            assert dict_config_mock.called_with(valid_logging_config)

        # Remember! The root logger is assumed to have been configured as
        # though VLLM_CONFIGURE_LOGGING=1 and VLLM_LOGGING_CONFIG_PATH=None.
        root_logger = logging.getLogger("vllm")
        other_logger_name = f"vllm.test_logger.{unique_name()}"
        other_logger = init_logger(other_logger_name)
        assert other_logger.handlers != root_logger.handlers
        assert other_logger.level != root_logger.level
        assert other_logger.propagate


def unique_name() -> str:
    return str(uuid4())
