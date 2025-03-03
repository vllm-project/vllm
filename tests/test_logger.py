# SPDX-License-Identifier: Apache-2.0

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
from vllm.logging_utils import NewLineFormatter


def f1(x):
    return f2(x)


def f2(x):
    return x


def test_trace_function_call():
    fd, path = tempfile.mkstemp()
    cur_dir = os.path.dirname(__file__)
    enable_trace_function_call(path, cur_dir)
    f1(1)
    with open(path) as f:
        content = f.read()

    assert "f1" in content
    assert "f2" in content
    sys.settrace(None)
    os.remove(path)


def test_default_vllm_root_logger_configuration():
    """This test presumes that VLLM_CONFIGURE_LOGGING (default: True) and
    VLLM_LOGGING_CONFIG_PATH (default: None) are not configured and default
    behavior is activated."""
    logger = logging.getLogger("vllm")
    assert logger.level == logging.DEBUG
    assert not logger.propagate

    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream == sys.stdout
    # we use DEBUG level for testing by default
    # assert handler.level == logging.INFO

    formatter = handler.formatter
    assert formatter is not None
    assert isinstance(formatter, NewLineFormatter)
    assert formatter._fmt == _FORMAT
    assert formatter.datefmt == _DATE_FORMAT


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 1)
@patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH", None)
def test_descendent_loggers_depend_on_and_propagate_logs_to_root_logger():
    """This test presumes that VLLM_CONFIGURE_LOGGING (default: True) and
    VLLM_LOGGING_CONFIG_PATH (default: None) are not configured and default
    behavior is activated."""
    root_logger = logging.getLogger("vllm")
    root_handler = root_logger.handlers[0]

    unique_name = f"vllm.{uuid4()}"
    logger = init_logger(unique_name)
    assert logger.name == unique_name
    assert logger.level == logging.NOTSET
    assert not logger.handlers
    assert logger.propagate

    message = "Hello, world!"
    with patch.object(root_handler, "emit") as root_handle_mock:
        logger.info(message)

    root_handle_mock.assert_called_once()
    _, call_args, _ = root_handle_mock.mock_calls[0]
    log_record = call_args[0]
    assert unique_name == log_record.name
    assert message == log_record.msg
    assert message == log_record.msg
    assert log_record.levelno == logging.INFO


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 0)
@patch("vllm.logger.VLLM_LOGGING_CONFIG_PATH", None)
def test_logger_configuring_can_be_disabled():
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""

    with patch("vllm.logger.dictConfig") as dict_config_mock:
        _configure_vllm_root_logger()
    dict_config_mock.assert_not_called()


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
    assert ex_info.type == RuntimeError  # noqa: E721
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
        unexpected_config: Any):
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
            assert ex_info.type == ValueError  # noqa: E721
            assert "Invalid logging config. Expected dict, got" in str(ex_info)


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
                       "vllm.logger.dictConfig") as dict_config_mock:
            _configure_vllm_root_logger()
            dict_config_mock.assert_called_with(valid_logging_config)


@patch("vllm.logger.VLLM_CONFIGURE_LOGGING", 0)
def test_custom_logging_config_causes_an_error_if_configure_logging_is_off():
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
                   logging_config_file.name):
            with pytest.raises(RuntimeError) as ex_info:
                _configure_vllm_root_logger()
            assert ex_info.type is RuntimeError
            expected_message_snippet = (
                "VLLM_CONFIGURE_LOGGING evaluated to false, but "
                "VLLM_LOGGING_CONFIG_PATH was given.")
            assert expected_message_snippet in str(ex_info)

        # Remember! The root logger is assumed to have been configured as
        # though VLLM_CONFIGURE_LOGGING=1 and VLLM_LOGGING_CONFIG_PATH=None.
        root_logger = logging.getLogger("vllm")
        other_logger_name = f"vllm.test_logger.{uuid4()}"
        other_logger = init_logger(other_logger_name)
        assert other_logger.handlers != root_logger.handlers
        assert other_logger.level != root_logger.level
        assert other_logger.propagate
