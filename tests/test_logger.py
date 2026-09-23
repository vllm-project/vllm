# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import io
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from vllm.logger import (
    _DATE_FORMAT,
    _FORMAT,
    _configure_vllm_root_logger,
    _use_color,
    enable_trace_function_call,
    init_logger,
)
from vllm.logging_utils import NewLineFormatter
from vllm.logging_utils.dump_input import prepare_object_to_dump
from vllm.utils.system_utils import decorate_logs


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
    assert logger.level == logging.INFO
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


def test_use_color_force_color(monkeypatch):
    """FORCE_COLOR forces colored logs without a TTY, while NO_COLOR and an
    explicit VLLM_LOGGING_COLOR=0 take precedence over it."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    for var in ("NO_COLOR", "FORCE_COLOR", "VLLM_LOGGING_COLOR"):
        monkeypatch.delenv(var, raising=False)

    assert not _use_color()

    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _use_color()

    monkeypatch.setenv("VLLM_LOGGING_COLOR", "0")
    assert not _use_color()
    monkeypatch.delenv("VLLM_LOGGING_COLOR")

    monkeypatch.setenv("NO_COLOR", "1")
    assert not _use_color()


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


# Add vllm prefix to make sure logs go through the vllm logger
test_logger = init_logger("vllm.test_logger")


def mp_function(**kwargs):
    # This function runs in a subprocess

    test_logger.warning("This is a subprocess: %s", kwargs.get("a"))
    test_logger.error("This is a subprocess error.")
    test_logger.debug("This is a subprocess debug message: %s.", kwargs.get("b"))


def test_caplog_mp_fork(caplog_vllm, caplog_mp_fork):
    with caplog_vllm.at_level(logging.DEBUG, logger="vllm"), caplog_mp_fork():
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


def _decorate_capture(monkeypatch):
    """Redirect stdio and decorate logs with a fixed process name.

    Returns the stdout capture and the previous log record factory so tests
    can restore the process-wide factory afterwards.
    """
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr("vllm.utils.system_utils.envs.VLLM_CONFIGURE_LOGGING", True)
    monkeypatch.setattr("vllm.utils.system_utils.envs.NO_COLOR", True)
    monkeypatch.setattr("vllm.utils.system_utils.envs.VLLM_LOGGING_COLOR", "0")

    previous_factory = logging.getLogRecordFactory()
    decorate_logs("Worker_DP0")
    return out, previous_factory


def test_decorate_logs_keeps_json_lines_parseable(monkeypatch):
    """JSON formatter output must stay valid NDJSON (issue #57955)."""
    out, previous_factory = _decorate_capture(monkeypatch)
    try:
        sys.stdout.write('{"message": "Engine started"}\n')
        line = out.getvalue().splitlines()[0]
        assert json.loads(line) == {"message": "Engine started"}
    finally:
        logging.setLogRecordFactory(previous_factory)


def test_decorate_logs_still_prefixes_text_lines(monkeypatch):
    """Default human-readable console logging retains its process prefix."""
    out, previous_factory = _decorate_capture(monkeypatch)
    try:
        sys.stdout.write("INFO hello\n")
        line = out.getvalue().splitlines()[0]
        assert line.startswith("(Worker_DP0 pid=")
        assert line.endswith(") INFO hello")
    finally:
        logging.setLogRecordFactory(previous_factory)


def test_decorate_logs_exposes_process_identity_on_records(monkeypatch):
    """Structured consumers get process identity from the LogRecord itself."""
    _, previous_factory = _decorate_capture(monkeypatch)
    try:
        factory = logging.getLogRecordFactory()
        record = factory("vllm.test", logging.INFO, __file__, 1, "hi", None, None)
        assert record.vllm_process_name == "Worker_DP0"
        assert record.vllm_pid == os.getpid()
    finally:
        logging.setLogRecordFactory(previous_factory)
