# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from vllm.entrypoints.logger import RequestLogger
from vllm.logger import (
    _DATE_FORMAT,
    _FORMAT,
    _configure_vllm_root_logger,
    enable_trace_function_call,
    init_logger,
)
from vllm.logging_utils import NewLineFormatter
from vllm.logging_utils.dump_input import prepare_object_to_dump


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


def test_default_vllm_root_logger_configuration(monkeypatch):
    """This test presumes that VLLM_CONFIGURE_LOGGING (default: True) and
    VLLM_LOGGING_CONFIG_PATH (default: None) are not configured and default
    behavior is activated."""
    monkeypatch.setenv("VLLM_LOGGING_COLOR", "0")
    _configure_vllm_root_logger()

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


def test_descendent_loggers_depend_on_and_propagate_logs_to_root_logger(monkeypatch):
    """This test presumes that VLLM_CONFIGURE_LOGGING (default: True) and
    VLLM_LOGGING_CONFIG_PATH (default: None) are not configured and default
    behavior is activated."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")
    monkeypatch.delenv("VLLM_LOGGING_CONFIG_PATH", raising=False)

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


def test_logger_configuring_can_be_disabled(monkeypatch):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    monkeypatch.delenv("VLLM_LOGGING_CONFIG_PATH", raising=False)

    with patch("vllm.logger.dictConfig") as dict_config_mock:
        _configure_vllm_root_logger()
    dict_config_mock.assert_not_called()


def test_an_error_is_raised_when_custom_logging_config_file_does_not_exist(monkeypatch):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")
    monkeypatch.setenv(
        "VLLM_LOGGING_CONFIG_PATH",
        "/if/there/is/a/file/here/then/you/did/this/to/yourself.json",
    )

    with pytest.raises(RuntimeError) as ex_info:
        _configure_vllm_root_logger()
    assert ex_info.type == RuntimeError  # noqa: E721
    assert "File does not exist" in str(ex_info)


def test_an_error_is_raised_when_custom_logging_config_is_invalid_json(monkeypatch):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")

    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write("---\nloggers: []\nversion: 1")
        logging_config_file.flush()
        monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", logging_config_file.name)
        with pytest.raises(JSONDecodeError) as ex_info:
            _configure_vllm_root_logger()
        assert ex_info.type == JSONDecodeError
        assert "Expecting value" in str(ex_info)


@pytest.mark.parametrize(
    "unexpected_config",
    (
        "Invalid string",
        [{"version": 1, "loggers": []}],
        0,
    ),
)
def test_an_error_is_raised_when_custom_logging_config_is_unexpected_json(
    monkeypatch,
    unexpected_config: Any,
):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however it fails before any change in behavior or
    configuration occurs."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")

    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(unexpected_config))
        logging_config_file.flush()
        monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", logging_config_file.name)
        with pytest.raises(ValueError) as ex_info:
            _configure_vllm_root_logger()
        assert ex_info.type == ValueError  # noqa: E721
        assert "Invalid logging config. Expected dict, got" in str(ex_info)


def test_custom_logging_config_is_parsed_and_used_when_provided(monkeypatch):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")

    valid_logging_config = {
        "loggers": {
            "vllm.test_logger.logger": {
                "handlers": [],
                "propagate": False,
            }
        },
        "version": 1,
    }
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(valid_logging_config))
        logging_config_file.flush()
        monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", logging_config_file.name)
        with patch("vllm.logger.dictConfig") as dict_config_mock:
            _configure_vllm_root_logger()
            dict_config_mock.assert_called_with(valid_logging_config)


def test_custom_logging_config_causes_an_error_if_configure_logging_is_off(monkeypatch):
    """This test calls _configure_vllm_root_logger again to test custom logging
    config behavior, however mocks are used to ensure no changes in behavior or
    configuration occur."""
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    valid_logging_config = {
        "loggers": {
            "vllm.test_logger.logger": {
                "handlers": [],
            }
        },
        "version": 1,
    }
    with NamedTemporaryFile(encoding="utf-8", mode="w") as logging_config_file:
        logging_config_file.write(json.dumps(valid_logging_config))
        logging_config_file.flush()
        monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", logging_config_file.name)
        with pytest.raises(RuntimeError) as ex_info:
            _configure_vllm_root_logger()
        assert ex_info.type is RuntimeError
        expected_message_snippet = (
            "VLLM_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_LOGGING_CONFIG_PATH was given."
        )
        assert expected_message_snippet in str(ex_info)

        # Remember! The root logger is assumed to have been configured as
        # though VLLM_CONFIGURE_LOGGING=1 and VLLM_LOGGING_CONFIG_PATH=None.
        root_logger = logging.getLogger("vllm")
        other_logger_name = f"vllm.test_logger.{uuid4()}"
        other_logger = init_logger(other_logger_name)
        assert other_logger.handlers != root_logger.handlers
        assert other_logger.level != root_logger.level
        assert other_logger.propagate


def test_prepare_object_to_dump():
    str_obj = "str"
    assert prepare_object_to_dump(str_obj) == "'str'"

    list_obj = [1, 2, 3]
    assert prepare_object_to_dump(list_obj) == "[1, 2, 3]"

    dict_obj = {"a": 1, "b": "b"}
    assert prepare_object_to_dump(dict_obj) in [
        "{a: 1, b: 'b'}",
        "{b: 'b', a: 1}",
    ]

    set_obj = {1, 2, 3}
    assert prepare_object_to_dump(set_obj) == "[1, 2, 3]"

    tuple_obj = ("a", "b", "c")
    assert prepare_object_to_dump(tuple_obj) == "['a', 'b', 'c']"

    class CustomEnum(enum.Enum):
        A = enum.auto()
        B = enum.auto()
        C = enum.auto()

    assert prepare_object_to_dump(CustomEnum.A) == repr(CustomEnum.A)

    @dataclass
    class CustomClass:
        a: int
        b: str

    assert prepare_object_to_dump(CustomClass(1, "b")) == "CustomClass(a=1, b='b')"


def test_request_logger_log_outputs():
    """Test the new log_outputs functionality."""
    # Create a mock logger to capture log calls
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test basic output logging
        request_logger.log_outputs(
            request_id="test-123",
            outputs="Hello, world!",
            output_token_ids=[1, 2, 3, 4],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-123"
        assert call_args[3] == "Hello, world!"
        assert call_args[4] == [1, 2, 3, 4]
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_streaming_delta():
    """Test log_outputs with streaming delta mode."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test streaming delta logging
        request_logger.log_outputs(
            request_id="test-456",
            outputs="Hello",
            output_token_ids=[1],
            finish_reason=None,
            is_streaming=True,
            delta=True,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-456"
        assert call_args[2] == " (streaming delta)"
        assert call_args[3] == "Hello"
        assert call_args[4] == [1]
        assert call_args[5] is None


def test_request_logger_log_outputs_streaming_complete():
    """Test log_outputs with streaming complete mode."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test streaming complete logging
        request_logger.log_outputs(
            request_id="test-789",
            outputs="Complete response",
            output_token_ids=[1, 2, 3],
            finish_reason="length",
            is_streaming=True,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-789"
        assert call_args[2] == " (streaming complete)"
        assert call_args[3] == "Complete response"
        assert call_args[4] == [1, 2, 3]
        assert call_args[5] == "length"


def test_request_logger_log_outputs_with_truncation():
    """Test log_outputs respects max_log_len setting."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        # Set max_log_len to 10
        request_logger = RequestLogger(max_log_len=10)

        # Test output truncation
        long_output = "This is a very long output that should be truncated"
        long_token_ids = list(range(20))  # 20 tokens

        request_logger.log_outputs(
            request_id="test-truncate",
            outputs=long_output,
            output_token_ids=long_token_ids,
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        # Check that output was truncated to first 10 characters
        logged_output = call_args[0][3]
        assert logged_output == "This is a "
        assert len(logged_output) == 10

        # Check that token IDs were truncated to first 10 tokens
        logged_token_ids = call_args[0][4]
        assert logged_token_ids == list(range(10))
        assert len(logged_token_ids) == 10


def test_request_logger_log_outputs_none_values():
    """Test log_outputs handles None values correctly."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test with None output_token_ids
        request_logger.log_outputs(
            request_id="test-none",
            outputs="Test output",
            output_token_ids=None,
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-none"
        assert call_args[3] == "Test output"
        assert call_args[4] is None
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_empty_output():
    """Test log_outputs handles empty output correctly."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=5)

        # Test with empty output
        request_logger.log_outputs(
            request_id="test-empty",
            outputs="",
            output_token_ids=[],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args
        assert "Generated response %s%s" in call_args[0]
        assert call_args[1] == "test-empty"
        assert call_args[3] == ""
        assert call_args[4] == []
        assert call_args[5] == "stop"


def test_request_logger_log_outputs_integration():
    """Test that log_outputs can be called alongside log_inputs."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test that both methods can be called without interference
        request_logger.log_inputs(
            request_id="test-integration",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            params=None,
            lora_request=None,
        )

        request_logger.log_outputs(
            request_id="test-integration",
            outputs="Test output",
            output_token_ids=[4, 5, 6],
            finish_reason="stop",
            is_streaming=False,
            delta=False,
        )

        # Should have been called twice - once for inputs, once for outputs
        assert mock_logger.info.call_count == 2

        # Check that the calls were made with correct patterns
        input_call = mock_logger.info.call_args_list[0][0]
        output_call = mock_logger.info.call_args_list[1][0]

        assert "Received request %s" in input_call[0]
        assert input_call[1] == "test-integration"

        assert "Generated response %s%s" in output_call[0]
        assert output_call[1] == "test-integration"


def test_streaming_complete_logs_full_text_content():
    """Test that streaming complete logging includes
    full accumulated text, not just token count."""
    mock_logger = MagicMock()

    with patch("vllm.entrypoints.logger.logger", mock_logger):
        request_logger = RequestLogger(max_log_len=None)

        # Test with actual content instead of token count format
        full_response = "This is a complete response from streaming"
        request_logger.log_outputs(
            request_id="test-streaming-full-text",
            outputs=full_response,
            output_token_ids=None,
            finish_reason="streaming_complete",
            is_streaming=True,
            delta=False,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args.args

        # Verify the logged output is the full text, not a token count format
        logged_output = call_args[3]
        assert logged_output == full_response
        assert "tokens>" not in logged_output
        assert "streaming_complete" not in logged_output

        # Verify other parameters
        assert call_args[1] == "test-streaming-full-text"
        assert call_args[2] == " (streaming complete)"
        assert call_args[5] == "streaming_complete"


# Add vllm prefix to make sure logs go through the vllm logger
test_logger = init_logger("vllm.test_logger")


def mp_function(**kwargs):
    # This function runs in a subprocess

    test_logger.warning("This is a subprocess: %s", kwargs.get("a"))
    test_logger.error("This is a subprocess error.")
    test_logger.debug("This is a subprocess debug message: %s.", kwargs.get("b"))


def test_caplog_mp_fork(caplog_vllm, caplog_mp_fork):
    with caplog_vllm.at_level(logging.DEBUG), caplog_mp_fork():
        import multiprocessing

        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(
            target=mp_function,
            name=f"SubProcess{1}",
            kwargs={"a": "AAAA", "b": "BBBBB"},
        )
        p.start()
        p.join()

    assert "AAAA" in caplog_vllm.text
    assert "BBBBB" in caplog_vllm.text


def test_caplog_mp_spawn(caplog_mp_spawn):
    with caplog_mp_spawn(logging.DEBUG) as log_holder:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=mp_function,
            name=f"SubProcess{1}",
            kwargs={"a": "AAAA", "b": "BBBBB"},
        )
        p.start()
        p.join()

    assert "AAAA" in log_holder.text
    assert "BBBBB" in log_holder.text
