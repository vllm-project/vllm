# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from abc import ABC
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import WatchdogConfig
from vllm.utils import watch_dog as watch_dog_module
from vllm.utils.watch_dog import (
    FaultHandlerWrapper,
    WatchDog,
    WatchDogHandler,
    WatchDogHandlerRegistry,
    WatchDogNoop,
    WatchDogThread,
    get_watch_dog,
    start_watch_dog,
)


@pytest.fixture(autouse=True)
def _reset_global_watch_dog():
    """Reset the module-level WatchDog singleton and the handler registry
    between tests."""
    original = watch_dog_module._watch_dog
    watch_dog_module._watch_dog = WatchDogNoop()
    WatchDogThread.clear_handlers()
    yield
    watch_dog_module._watch_dog = original
    WatchDogThread.clear_handlers()


def test_watch_dog_is_abstract_base_class():
    """Verify WatchDog declares the feed()/handle_failure() contract."""
    assert issubclass(WatchDog, ABC)
    assert {"feed", "handle_failure"} <= WatchDog.__abstractmethods__
    with pytest.raises(TypeError):
        WatchDog()  # type: ignore[abstract]


def test_watch_dog_noop_is_inert():
    """Verify WatchDogNoop accepts feed() and handle_failure() without doing
    anything."""
    wd = WatchDogNoop()
    assert isinstance(wd, WatchDog)
    wd.feed()
    wd.handle_failure("timeout")
    with wd.disable():
        wd.feed()


def test_watch_dog_thread_initialization(tmp_path):
    """Verify WatchDogThread derives its settings from the given name,
    config, and logger."""
    logger = MagicMock()
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    wd = WatchDogThread("worker_0", config, logger)
    assert wd._name == "worker_0"
    assert wd._timeout == 30
    assert wd._check_interval == 5
    assert wd._logger is logger
    assert wd._thread is None
    assert not wd._stop_event.is_set()


def test_feed_updates_last_feed_time(tmp_path):
    """Verify feed() advances the last-feed timestamp."""
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    before = time.monotonic()
    wd.feed()
    assert wd._last_feed_time >= before
    assert wd._last_feed_time <= time.monotonic()


def test_handle_failure_writes_traceback_files(tmp_path):
    """Verify handle_failure() routes to the registered stack-dump handler,
    appending numbered traceback entries to the dump file."""
    config = WatchdogConfig(dump_dir=str(tmp_path / "dump"))
    wd = WatchDogThread("test_proc", config, MagicMock())
    handler = FaultHandlerWrapper("test_proc", config.dump_dir, MagicMock())
    wd.register_handler(handler)
    wd.handle_failure("timeout")
    wd.handle_failure("heartbeat lost")

    log_file = tmp_path / "dump" / (f"VLLM_STACK_DUMP_for_test_proc_{os.getpid()}.log")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Call stack dump #1" in content
    assert "due to timeout" in content
    assert "Call stack dump #2" in content
    assert "due to heartbeat lost" in content
    assert "===" in content
    assert handler._dump_seq == 3


def test_handle_failure_logs_failure_via_logger(tmp_path):
    """Verify dump failures are reported through the configured logger."""
    logger = MagicMock()
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    handler = FaultHandlerWrapper("vllm", str(tmp_path), logger)
    wd.register_handler(handler)
    with patch("builtins.open", side_effect=OSError("boom")):
        wd.handle_failure("timeout")
    logger.warning.assert_called_once()
    message = logger.warning.call_args[0][0]
    args = logger.warning.call_args[0][1:]
    assert message == "[Watchdog]Failed to dump stack trace to %s: %s"
    assert args[0] == handler._dump_file
    assert isinstance(args[1], OSError) and str(args[1]) == "boom"
    assert handler._dump_seq == 1  # not incremented on failure


def test_handle_failure_logs_success_via_logger(tmp_path):
    """Verify a successful dump is reported through the configured logger and
    increments the dump sequence."""
    config = WatchdogConfig(dump_dir=str(tmp_path / "dump"))
    logger = MagicMock()
    wd = WatchDogThread("vllm", config, MagicMock())
    handler = FaultHandlerWrapper("vllm", config.dump_dir, logger)
    wd.register_handler(handler)
    wd.handle_failure("timeout")
    logger.info.assert_called_once()
    message = logger.info.call_args[0][0]
    args = logger.info.call_args[0][1:]
    assert message == "[Watchdog]Dumped stack to %s due to %s"
    assert args == (handler._dump_file, "timeout")
    assert os.path.exists(handler._dump_file)
    assert handler._dump_seq == 2  # incremented on success


def test_start_launches_daemon_thread(tmp_path):
    """Verify start() spawns a live daemon monitor thread."""
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    assert wd._thread is not None
    assert wd._thread.is_alive()
    assert wd._thread.daemon is True
    assert not wd._stop_event.is_set()
    wd.stop()


def test_start_is_noop_when_thread_already_running(tmp_path):
    """Verify a second start() does not spawn another monitor thread."""
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    original_thread = wd._thread
    wd.start()
    assert wd._thread is original_thread
    wd.stop()


def test_stop_sets_event_and_clears_thread(tmp_path):
    """Verify stop() sets the stop event and clears the thread reference."""
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


def test_stop_without_start_does_not_raise(tmp_path):
    """Verify stop() is safe to call before start()."""
    wd = WatchDogThread("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


class TestCheckLoop:
    """Tests for the internal background check loop."""

    @staticmethod
    def _make_watchdog(tmp_path) -> WatchDogThread:
        return WatchDogThread(
            "vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock()
        )

    @staticmethod
    def _fake_wait_that_exits_after(count, target=2):
        def fake_wait(_):
            count["n"] += 1
            return count["n"] >= target  # break out of the infinite loop

        return fake_wait

    def test_dumps_stack_when_timed_out(self, tmp_path):
        """Verify the check loop dumps the stack once on feed timeout."""
        wd = self._make_watchdog(tmp_path)
        wd._timeout = 0.01  # type: ignore[assignment]
        wd._last_feed_time = time.monotonic() - 10  # already timed out

        with (
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            wd._check_loop()

        mock_dump.assert_called_once()
        assert wd._last_timeout_log >= wd._last_feed_time

    def test_dumps_stack_only_once_per_stale_feed(self, tmp_path):
        """Verify the check loop dumps at most once per stale feed epoch."""
        wd = self._make_watchdog(tmp_path)
        wd._timeout = 0.01  # type: ignore[assignment]
        wd._last_feed_time = time.monotonic() - 10  # already timed out

        with (
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}, target=3),
            ),
        ):
            wd._check_loop()

        assert mock_dump.call_count == 1

        # A fresh staleness epoch (as after a feed) dumps again once stale.
        wd._last_timeout_log = 0.0
        wd._last_feed_time = time.monotonic() - 10
        with (
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            wd._check_loop()

        mock_dump.assert_called_once()

    def test_skips_duplicate_timeout_log(self, tmp_path):
        """Verify the check loop skips a timeout already logged since the
        last feed."""
        wd = self._make_watchdog(tmp_path)
        wd._timeout = 0.01  # type: ignore[assignment]
        # Timed out, but the timeout has already been logged after the
        # last feed.
        wd._last_feed_time = time.monotonic() - 10
        wd._last_timeout_log = wd._last_feed_time + 1

        with (
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            wd._check_loop()

        mock_dump.assert_not_called()

    def test_does_nothing_when_not_timed_out(self, tmp_path):
        """Verify the check loop stays quiet while the watchdog is fed."""
        wd = self._make_watchdog(tmp_path)
        wd._last_feed_time = time.monotonic()  # freshly fed

        with (
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            wd._check_loop()

        mock_dump.assert_not_called()

    def test_disable_suspends_timeout_detection(self, tmp_path):
        """Verify the check loop skips timeout detection while disabled and
        refreshes the feed time so the window restarts once re-enabled."""
        wd = self._make_watchdog(tmp_path)
        wd._timeout = 0.01  # type: ignore[assignment]
        wd._last_feed_time = time.monotonic() - 10  # already timed out

        assert wd._active
        with (
            wd.disable(),
            patch.object(wd, "handle_failure") as mock_dump,
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            assert not wd._active
            wd._check_loop()

        assert wd._active
        mock_dump.assert_not_called()
        assert wd._last_feed_time > time.monotonic() - 1

    def test_disable_is_not_reentrant(self, tmp_path):
        """Verify disable() rejects a nested disable() call."""
        wd = self._make_watchdog(tmp_path)
        outer = wd.disable()
        outer.__enter__()
        try:
            inner = wd.disable()
            with pytest.raises(AssertionError):
                inner.__enter__()
        finally:
            outer.__exit__(None, None, None)

    def test_disable_restores_active_on_exception(self, tmp_path):
        """Verify disable() re-enables the watchdog when the wrapped block
        raises."""
        wd = self._make_watchdog(tmp_path)
        with pytest.raises(RuntimeError), wd.disable():
            raise RuntimeError("boom")
        assert wd._active


def test_get_watch_dog_returns_shared_noop_singleton():
    """Verify get_watch_dog() returns the same dormant WatchDogNoop instance
    until start_watch_dog() swaps it."""
    first = get_watch_dog()
    second = get_watch_dog()
    assert isinstance(first, WatchDogNoop)
    assert first is second


def test_start_watch_dog_stays_dormant_without_dump_dir():
    """Verify start_watch_dog() keeps the watchdog dormant when the config
    disables the feature via an empty dump_dir."""
    logger = MagicMock()
    config = WatchdogConfig()  # dump_dir defaults to ""
    noop = WatchDogNoop()
    with patch.object(watch_dog_module, "_watch_dog", noop):
        result = start_watch_dog("engine_0", config, logger)
    assert result is noop
    assert isinstance(result, WatchDogNoop)


def test_start_watch_dog_creates_thread_and_starts(tmp_path):
    """Verify start_watch_dog() swaps in a WatchDogThread with a registered
    FaultHandlerWrapper and starts its background thread when dump_dir is
    set."""
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    noop = WatchDogNoop()
    with patch.object(watch_dog_module, "_watch_dog", noop):
        result = start_watch_dog("engine_7", config, MagicMock())
    try:
        assert isinstance(result, WatchDogThread)
        assert result._name == "engine_7"
        assert result._timeout == 30
        assert result._check_interval == 5
        assert len(result._registry) == 1
        handler = result._registry[0]
        assert isinstance(handler, FaultHandlerWrapper)
        assert handler._dump_dir == str(tmp_path)
        assert handler._dump_file == os.path.join(
            str(tmp_path), f"VLLM_STACK_DUMP_for_engine_7_{os.getpid()}.log"
        )
        assert result._thread is not None
        assert result._thread.is_alive()
        assert result._thread.daemon is True
    finally:
        cast(WatchDogThread, result).stop()


def test_start_watch_dog_attaches_logger(tmp_path):
    """Verify start_watch_dog() passes the logger through to the thread
    watchdog."""
    config = WatchdogConfig(dump_dir=str(tmp_path))
    logger = MagicMock()
    noop = WatchDogNoop()
    with patch.object(watch_dog_module, "_watch_dog", noop):
        result = start_watch_dog("engine_0", config, logger)
    try:
        assert isinstance(result, WatchDogThread)
        assert result._logger is logger
    finally:
        cast(WatchDogThread, result).stop()


def test_registered_handler_called_via_watchdog_singleton(tmp_path):
    """Verify a custom handler registered on the global watchdog is invoked
    when handle_failure() is called on the singleton."""
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    with patch.object(watch_dog_module, "_watch_dog", WatchDogNoop()):
        handler = MagicMock()
        get_watch_dog().register_handler(handler)
        start_watch_dog("engine_0", config, MagicMock())
        try:
            get_watch_dog().handle_failure("test")
            handler.handle.assert_called_once_with("test")
        finally:
            cast(WatchDogThread, get_watch_dog()).stop()


def test_watch_dog_handler_is_abstract_base_class():
    """Verify WatchDogHandler declares a handle() contract that concrete
    handlers are forced to implement."""
    assert issubclass(WatchDogHandler, ABC)
    assert "handle" in WatchDogHandler.__abstractmethods__
    with pytest.raises(TypeError):
        WatchDogHandler()  # type: ignore[abstract]

    class _ConcreteHandler(WatchDogHandler):
        def handle(self, reason):
            return f"handled {reason}"

    result = _ConcreteHandler().handle("feed timeout")
    assert result == "handled feed timeout"


def test_fault_handler_wrapper_initialization(tmp_path):
    """Verify FaultHandlerWrapper wires up the dump dir/file from the name,
    seeds the dump sequence, and enables faulthandler for all threads."""
    logger = MagicMock()
    with patch("faulthandler.enable") as mock_enable:
        handler = FaultHandlerWrapper("proc_1", str(tmp_path / "dumps"), logger)
    assert handler._dump_dir == str(tmp_path / "dumps")
    assert handler._dump_file == os.path.join(
        str(tmp_path / "dumps"),
        f"VLLM_STACK_DUMP_for_proc_1_{os.getpid()}.log",
    )
    assert handler._logger is logger
    assert handler._dump_seq == 1
    mock_enable.assert_called_once_with(all_threads=True)


def test_fault_handler_wrapper_handle_success(tmp_path):
    """Verify handle() creates the dump dir, appends a numbered traceback
    entry, increments the sequence, and reports success via the logger."""
    logger = MagicMock()
    handler = FaultHandlerWrapper("proc_1", str(tmp_path / "nested"), logger)
    handler.handle("feed timeout")

    log_file = handler._dump_file
    assert os.path.exists(log_file)
    with open(log_file, encoding="utf-8") as f:
        content = f.read()
    assert "Call stack dump #1" in content
    assert "due to feed timeout" in content
    assert "================================================================" in content

    assert handler._dump_seq == 2
    logger.info.assert_called_once()
    message = logger.info.call_args[0][0]
    args = logger.info.call_args[0][1:]
    assert message == "[Watchdog]Dumped stack to %s due to %s"
    assert args == (log_file, "feed timeout")


def test_fault_handler_wrapper_handle_failure(tmp_path):
    """Verify handle() reports a dump failure through the logger and leaves
    the dump sequence unchanged."""
    logger = MagicMock()
    handler = FaultHandlerWrapper("proc_1", str(tmp_path), logger)
    handler._dump_seq = 5

    with patch("builtins.open", side_effect=OSError("boom")):
        handler.handle("feed timeout")

    assert handler._dump_seq == 5  # not incremented on failure
    logger.warning.assert_called_once()
    message = logger.warning.call_args[0][0]
    args = logger.warning.call_args[0][1:]
    assert message == "[Watchdog]Failed to dump stack trace to %s: %s"
    assert args[0] == handler._dump_file
    assert isinstance(args[1], OSError) and str(args[1]) == "boom"


def test_handler_registry_register_handlers_and_call():
    """Verify register_handler() appends handlers and call_handlers()
    forwards the reason to every registered handler."""
    WatchDogHandlerRegistry.clear_handlers()
    first = MagicMock()
    second = MagicMock()
    try:
        WatchDogHandlerRegistry.register_handler(first)
        WatchDogHandlerRegistry.register_handler(second)
        assert len(WatchDogHandlerRegistry._registry) == 2

        WatchDogHandlerRegistry.call_handlers("feed timeout")
        first.handle.assert_called_once_with("feed timeout")
        second.handle.assert_called_once_with("feed timeout")
    finally:
        WatchDogHandlerRegistry.clear_handlers()


def test_handler_registry_clear_handlers():
    """Verify clear_handlers() empties the shared registry and call_handlers()
    on an empty registry is a no-op."""
    WatchDogHandlerRegistry.clear_handlers()
    handler = MagicMock()
    WatchDogHandlerRegistry.register_handler(handler)
    assert len(WatchDogHandlerRegistry._registry) == 1

    WatchDogHandlerRegistry.clear_handlers()
    assert WatchDogHandlerRegistry._registry == []
    WatchDogHandlerRegistry.call_handlers("feed timeout")
    handler.handle.assert_not_called()
