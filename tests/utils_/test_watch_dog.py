# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from unittest.mock import MagicMock, patch

from vllm.utils.safe_fs import get_user_root_dir
from vllm.utils.watch_dog import WatchDog, get_watch_dog


def test_default_initialization():
    """Verify the watchdog initializes with the default settings."""
    wd = WatchDog()
    assert wd._name == "vllm"
    assert wd._timeout == 300
    assert wd._check_interval == 10
    assert wd._sequence == 1
    assert wd._thread is None
    assert wd._logger is None
    assert not wd._stop_event.is_set()
    assert wd._dump_dir == os.path.join(get_user_root_dir(), "dump")
    assert wd._dump_file == os.path.join(
        wd._dump_dir, f"VLLM_STACK_DUMP_for_vllm_{os.getpid()}.log")


def test_set_name_updates_dump_file():
    """Verify the watchdog name can be overridden and the dump file name
    follows it."""
    wd = WatchDog()
    wd.set_name("worker_0")
    assert wd._name == "worker_0"
    assert wd._dump_file == os.path.join(
        wd._dump_dir, f"VLLM_STACK_DUMP_for_worker_0_{os.getpid()}.log")


def test_feed_updates_last_feed_time():
    """Verify feed() advances the last-feed timestamp."""
    wd = WatchDog()
    before = time.monotonic()
    wd.feed()
    assert wd._last_feed_time >= before
    assert wd._last_feed_time <= time.monotonic()


def test_dump_stack_writes_traceback_files(tmp_path):
    """Verify dump_stack() appends numbered traceback entries to the dump
    file."""
    with patch("vllm.utils.watch_dog.get_user_root_dir",
               return_value=str(tmp_path)):
        wd = WatchDog()
    wd.set_name("test_proc")
    wd.dump_stack("timeout")
    wd.dump_stack("heartbeat lost")

    log_file = tmp_path / "dump" / (
        f"VLLM_STACK_DUMP_for_test_proc_{os.getpid()}.log")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Call stack dump #1" in content
    assert "due to timeout" in content
    assert "Call stack dump #2" in content
    assert "due to heartbeat lost" in content
    assert "===" in content
    assert wd._sequence == 3


def test_dump_stack_prepares_private_dir_and_safe_opens(tmp_path):
    """Verify dump_stack() prepares the private dump directory and opens the
    file with no-follow semantics."""
    with patch("vllm.utils.watch_dog.get_user_root_dir",
               return_value=str(tmp_path)):
        wd = WatchDog()
    with (
        patch("vllm.utils.watch_dog.prepare_private_dir") as mock_prep,
        patch("vllm.utils.watch_dog.safe_open_file") as mock_open,
    ):
        wd.dump_stack("timeout")
    mock_prep.assert_called_once_with(str(tmp_path / "dump"))
    mock_open.assert_called_once_with(wd._dump_file, "a")


def test_dump_stack_logs_failure_via_logger(tmp_path):
    """Verify dump failures are reported through the configured logger."""
    with patch("vllm.utils.watch_dog.get_user_root_dir",
               return_value=str(tmp_path)):
        wd = WatchDog()
    logger = MagicMock()
    wd.set_logger(logger)
    with patch("vllm.utils.watch_dog.prepare_private_dir",
               side_effect=OSError("boom")):
        wd.dump_stack("timeout")
    logger.warning.assert_called_once()
    message = logger.warning.call_args[0][0]
    assert "Failed to dump stack trace" in message
    assert "boom" in message
    assert wd._sequence == 1  # not incremented on failure


def test_dump_stack_swallows_failure_without_logger(tmp_path):
    """Verify dump failures are silently ignored without a logger."""
    with patch("vllm.utils.watch_dog.get_user_root_dir",
               return_value=str(tmp_path)):
        wd = WatchDog()
    with patch("vllm.utils.watch_dog.prepare_private_dir",
               side_effect=OSError("boom")):
        wd.dump_stack("timeout")  # must not raise
    assert wd._sequence == 1  # not incremented on failure


def test_start_launches_daemon_thread():
    """Verify start() spawns a live daemon monitor thread."""
    wd = WatchDog()
    wd.start()
    assert wd._thread is not None
    assert wd._thread.is_alive()
    assert wd._thread.daemon is True
    assert not wd._stop_event.is_set()
    wd.stop()


def test_start_is_noop_when_thread_already_running():
    """Verify a second start() does not spawn another monitor thread."""
    wd = WatchDog()
    wd.start()
    original_thread = wd._thread
    wd.start()
    assert wd._thread is original_thread
    wd.stop()


def test_stop_sets_event_and_clears_thread():
    """Verify stop() sets the stop event and clears the thread reference."""
    wd = WatchDog()
    wd.start()
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


def test_stop_without_start_does_not_raise():
    """Verify stop() is safe to call before start()."""
    wd = WatchDog()
    wd.stop()
    assert wd._stop_event.is_set()
    assert wd._thread is None


class TestCheckLoop:
    """Tests for the internal background check loop."""

    @staticmethod
    def _fake_wait_that_exits_after(count, target=2):
        def fake_wait(_):
            count["n"] += 1
            return count["n"] >= target  # break out of the infinite loop

        return fake_wait

    def test_dumps_stack_when_timed_out(self):
        """Verify the check loop dumps the stack once on feed timeout."""
        wd = WatchDog()
        # The constructor takes no args; tighten the timeout for the test.
        wd._timeout = 0.01
        wd._last_feed_time = time.monotonic() - 10  # already timed out

        with patch.object(wd, "dump_stack") as mock_dump, patch.object(
                wd._stop_event, "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0})):
            wd._check_loop()

        mock_dump.assert_called_once()
        assert wd._last_timeout_log >= wd._last_feed_time

    def test_skips_duplicate_timeout_log(self):
        """Verify the check loop skips a timeout already logged since the
        last feed."""
        wd = WatchDog()
        # The constructor takes no args; tighten the timeout for the test.
        wd._timeout = 0.01
        # Timed out, but the timeout has already been logged after the
        # last feed.
        wd._last_feed_time = time.monotonic() - 10
        wd._last_timeout_log = wd._last_feed_time + 1

        with patch.object(wd, "dump_stack") as mock_dump, patch.object(
                wd._stop_event, "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0})):
            wd._check_loop()

        mock_dump.assert_not_called()

    def test_does_nothing_when_not_timed_out(self):
        """Verify the check loop stays quiet while the watchdog is fed."""
        wd = WatchDog()
        wd._last_feed_time = time.monotonic()  # freshly fed

        with patch.object(wd, "dump_stack") as mock_dump, patch.object(
                wd._stop_event, "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0})):
            wd._check_loop()

        mock_dump.assert_not_called()


def test_get_watch_dog_returns_shared_singleton():
    """Verify get_watch_dog() returns the same WatchDog instance every
    time."""
    first = get_watch_dog()
    second = get_watch_dog()
    assert isinstance(first, WatchDog)
    assert first is second
