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
    WatchDog,
    WatchDogNoop,
    WatchDogRaw,
    get_watch_dog,
    start_watch_dog,
)


@pytest.fixture(autouse=True)
def _reset_global_watch_dog():
    """Reset the module-level WatchDog singleton between tests."""
    original = watch_dog_module._watch_dog
    watch_dog_module._watch_dog = WatchDogNoop()
    yield
    watch_dog_module._watch_dog = original


def test_watch_dog_is_abstract_base_class():
    """Verify WatchDog declares the feed()/dump_stack() contract."""
    assert issubclass(WatchDog, ABC)
    assert {"feed", "dump_stack"} <= WatchDog.__abstractmethods__
    with pytest.raises(TypeError):
        WatchDog()  # type: ignore[abstract]


def test_watch_dog_noop_is_inert():
    """Verify WatchDogNoop accepts feed() and dump_stack() without doing
    anything."""
    wd = WatchDogNoop()
    assert isinstance(wd, WatchDog)
    wd.feed()
    wd.dump_stack("timeout")
    with wd.disable():
        wd.feed()


def test_watch_dog_raw_initialization(tmp_path):
    """Verify WatchDogRaw derives its settings from the given name, config,
    and logger."""
    logger = MagicMock()
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    wd = WatchDogRaw("worker_0", config, logger)
    assert wd._name == "worker_0"
    assert wd._timeout == 30
    assert wd._check_interval == 5
    assert wd._dump_dir == str(tmp_path)
    assert wd._dump_file == os.path.join(
        str(tmp_path), f"VLLM_STACK_DUMP_for_worker_0_{os.getpid()}.log"
    )
    assert wd._logger is logger
    assert wd._dump_seq == 1
    assert wd._thread is None
    assert not wd._stop_event.is_set()


def test_feed_updates_last_feed_time(tmp_path):
    """Verify feed() advances the last-feed timestamp."""
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    before = time.monotonic()
    wd.feed()
    assert wd._last_feed_time >= before
    assert wd._last_feed_time <= time.monotonic()


def test_dump_stack_writes_traceback_files(tmp_path):
    """Verify dump_stack() appends numbered traceback entries to the dump
    file. The dump directory must exist before dumping."""
    config = WatchdogConfig(dump_dir=str(tmp_path / "dump"))
    os.makedirs(config.dump_dir)
    wd = WatchDogRaw("test_proc", config, MagicMock())
    wd.dump_stack("timeout")
    wd.dump_stack("heartbeat lost")

    log_file = tmp_path / "dump" / (f"VLLM_STACK_DUMP_for_test_proc_{os.getpid()}.log")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Call stack dump #1" in content
    assert "due to timeout" in content
    assert "Call stack dump #2" in content
    assert "due to heartbeat lost" in content
    assert "===" in content
    assert wd._dump_seq == 3


def test_dump_stack_logs_failure_via_logger(tmp_path):
    """Verify dump failures are reported through the configured logger."""
    logger = MagicMock()
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), logger)
    with patch("builtins.open", side_effect=OSError("boom")):
        wd.dump_stack("timeout")
    logger.warning.assert_called_once()
    message = logger.warning.call_args[0][0]
    args = logger.warning.call_args[0][1:]
    assert message == "[Watchdog]Failed to dump stack trace to %s: %s"
    assert args[0] == wd._dump_file
    assert isinstance(args[1], OSError) and str(args[1]) == "boom"
    assert wd._dump_seq == 1  # not incremented on failure


def test_dump_stack_logs_success_via_logger(tmp_path):
    """Verify a successful dump is reported through the configured logger and
    increments the dump sequence."""
    config = WatchdogConfig(dump_dir=str(tmp_path / "dump"))
    os.makedirs(config.dump_dir)
    logger = MagicMock()
    wd = WatchDogRaw("vllm", config, logger)
    wd.dump_stack("timeout")
    logger.info.assert_called_once()
    message = logger.info.call_args[0][0]
    args = logger.info.call_args[0][1:]
    assert message == "[Watchdog]Dumped stack to %s due to %s"
    assert args == (wd._dump_file, "timeout")
    assert os.path.exists(wd._dump_file)
    assert wd._dump_seq == 2  # incremented on success


def test_start_launches_daemon_thread(tmp_path):
    """Verify start() spawns a live daemon monitor thread."""
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    assert wd._thread is not None
    assert wd._thread.is_alive()
    assert wd._thread.daemon is True
    assert not wd._stop_event.is_set()
    wd.stop()


def test_start_is_noop_when_thread_already_running(tmp_path):
    """Verify a second start() does not spawn another monitor thread."""
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    original_thread = wd._thread
    wd.start()
    assert wd._thread is original_thread
    wd.stop()


def test_stop_sets_event_and_clears_thread(tmp_path):
    """Verify stop() sets the stop event and clears the thread reference."""
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.start()
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


def test_stop_without_start_does_not_raise(tmp_path):
    """Verify stop() is safe to call before start()."""
    wd = WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


class TestCheckLoop:
    """Tests for the internal background check loop."""

    @staticmethod
    def _make_watchdog(tmp_path) -> WatchDogRaw:
        return WatchDogRaw("vllm", WatchdogConfig(dump_dir=str(tmp_path)), MagicMock())

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
            patch.object(wd, "dump_stack") as mock_dump,
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
            patch.object(wd, "dump_stack") as mock_dump,
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
            patch.object(wd, "dump_stack") as mock_dump,
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
            patch.object(wd, "dump_stack") as mock_dump,
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
            patch.object(wd, "dump_stack") as mock_dump,
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
            patch.object(wd, "dump_stack") as mock_dump,
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


def test_start_watch_dog_creates_raw_and_starts(tmp_path):
    """Verify start_watch_dog() swaps in a WatchDogRaw and starts its
    background thread when dump_dir is set."""
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    noop = WatchDogNoop()
    with patch.object(watch_dog_module, "_watch_dog", noop):
        result = start_watch_dog("engine_7", config, MagicMock())
    try:
        assert isinstance(result, WatchDogRaw)
        assert result._name == "engine_7"
        assert result._timeout == 30
        assert result._check_interval == 5
        assert result._dump_dir == str(tmp_path)
        assert result._dump_file == os.path.join(
            str(tmp_path), f"VLLM_STACK_DUMP_for_engine_7_{os.getpid()}.log"
        )
        assert result._thread is not None
        assert result._thread.is_alive()
        assert result._thread.daemon is True
    finally:
        cast(WatchDogRaw, result).stop()


def test_start_watch_dog_attaches_logger(tmp_path):
    """Verify start_watch_dog() passes the logger through to the raw
    watchdog."""
    config = WatchdogConfig(dump_dir=str(tmp_path))
    logger = MagicMock()
    noop = WatchDogNoop()
    with patch.object(watch_dog_module, "_watch_dog", noop):
        result = start_watch_dog("engine_0", config, logger)
    try:
        assert isinstance(result, WatchDogRaw)
        assert result._logger is logger
    finally:
        cast(WatchDogRaw, result).stop()
