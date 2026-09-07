# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from unittest.mock import MagicMock, patch

from vllm.config import WatchdogConfig
from vllm.utils.safe_fs import get_user_root_dir
from vllm.utils.watch_dog import WatchDog, get_watch_dog, start_watch_dog


def test_default_initialization():
    """Verify the watchdog initializes with the default settings."""
    wd = WatchDog()
    assert wd._name == "vllm"
    assert wd._timeout == 300
    assert wd._check_interval == 10
    assert wd._dump_seq == 1
    assert wd._thread is None
    assert wd._logger is None
    assert not wd._stop_event.is_set()
    assert wd._dump_dir == os.path.join(get_user_root_dir(), "dump")
    assert wd._dump_file == os.path.join(
        wd._dump_dir, f"VLLM_STACK_DUMP_for_vllm_{os.getpid()}.log"
    )
    assert wd.num_timeouts == 0
    assert wd.num_recoveries == 0
    assert not wd._is_timeout_active


def test_set_name_does_not_rebuild_dump_file():
    """Verify set_name() only overrides the watchdog name; the dump file is
    rebuilt later by set_config(dump_dir=...) on top of the current name."""
    wd = WatchDog()
    default_dump_file = wd._dump_file
    wd.set_name("worker_0")
    assert wd._name == "worker_0"
    assert wd._dump_file == default_dump_file


def test_set_config_updates_dump_dir_and_dump_file(tmp_path):
    """Verify set_config(dump_dir=...) rebuilds the dump file name from the
    current watchdog name."""
    wd = WatchDog()
    wd.set_name("worker_0")
    wd.set_config(dump_dir=str(tmp_path))
    assert wd._dump_dir == str(tmp_path)
    assert wd._dump_file == os.path.join(
        str(tmp_path), f"VLLM_STACK_DUMP_for_worker_0_{os.getpid()}.log"
    )


def test_set_config_updates_timeout_only():
    """Verify set_config(timeout=...) leaves the other parameters untouched."""
    wd = WatchDog()
    dump_dir_before = wd._dump_dir
    wd.set_config(timeout=60)
    assert wd._timeout == 60
    assert wd._check_interval == 10
    assert wd._dump_dir is dump_dir_before
    assert wd._dump_file == os.path.join(
        wd._dump_dir, f"VLLM_STACK_DUMP_for_{wd._name}_{os.getpid()}.log"
    )


def test_set_config_updates_check_interval_only():
    """Verify set_config(check_interval=...) leaves the other parameters
    untouched."""
    wd = WatchDog()
    wd.set_config(check_interval=1)
    assert wd._check_interval == 1
    assert wd._timeout == 300
    assert wd._dump_dir == os.path.join(get_user_root_dir(), "dump")


def test_set_config_combines_all_parameters(tmp_path):
    """Verify set_config() applies all provided parameters at once."""
    wd = WatchDog()
    wd.set_name("engine_0")
    wd.set_config(timeout=100, check_interval=2, dump_dir=str(tmp_path))
    assert wd._timeout == 100
    assert wd._check_interval == 2
    assert wd._dump_dir == str(tmp_path)
    assert wd._dump_file == os.path.join(
        str(tmp_path), f"VLLM_STACK_DUMP_for_engine_0_{os.getpid()}.log"
    )


def test_set_config_noop_when_all_none():
    """Verify set_config() with no arguments changes nothing."""
    wd = WatchDog()
    state = (wd._timeout, wd._check_interval, wd._dump_dir, wd._dump_file)
    wd.set_config()
    assert (wd._timeout, wd._check_interval, wd._dump_dir, wd._dump_file) == state


def test_feed_updates_last_feed_time():
    """Verify feed() advances the last-feed timestamp."""
    wd = WatchDog()
    before = time.monotonic()
    wd.feed()
    assert wd._last_feed_time >= before
    assert wd._last_feed_time <= time.monotonic()


def test_dump_stack_writes_traceback_files(tmp_path):
    """Verify dump_stack() appends numbered traceback entries to the dump
    file. The dump directory is prepared by start(), so the watchdog must be
    started first."""
    with patch("vllm.utils.watch_dog.get_user_root_dir", return_value=str(tmp_path)):
        wd = WatchDog()
    wd.set_name("test_proc")
    wd.set_config(dump_dir=str(tmp_path / "dump"))
    wd.start()  # prepares the private dump directory
    try:
        wd.dump_stack("timeout")
        wd.dump_stack("heartbeat lost")
    finally:
        wd.stop()

    log_file = tmp_path / "dump" / (f"VLLM_STACK_DUMP_for_test_proc_{os.getpid()}.log")
    assert log_file.exists()
    content = log_file.read_text()
    assert "Call stack dump #1" in content
    assert "due to timeout" in content
    assert "Call stack dump #2" in content
    assert "due to heartbeat lost" in content
    assert "===" in content
    assert wd._dump_seq == 3


def test_start_prepares_private_dir_and_dump_safe_opens(tmp_path):
    """Verify start() prepares the private dump directory and dump_stack()
    opens the file with no-follow semantics."""
    with patch("vllm.utils.watch_dog.get_user_root_dir", return_value=str(tmp_path)):
        wd = WatchDog()
    with (
        patch("vllm.utils.watch_dog.prepare_private_dir") as mock_prep,
        patch("vllm.utils.watch_dog.safe_open_file") as mock_open,
        patch("vllm.utils.watch_dog.threading.Thread") as mock_thread,
    ):
        wd.start()
        wd.dump_stack("timeout")
        wd.stop()
    mock_prep.assert_called_once_with(str(tmp_path / "dump"))
    mock_open.assert_called_once_with(wd._dump_file, "a")
    mock_thread.assert_called_once()


def test_dump_stack_logs_failure_via_logger(tmp_path):
    """Verify dump failures are reported through the configured logger."""
    with patch("vllm.utils.watch_dog.get_user_root_dir", return_value=str(tmp_path)):
        wd = WatchDog()
    logger = MagicMock()
    wd.set_logger(logger)
    with patch("vllm.utils.watch_dog.safe_open_file", side_effect=OSError("boom")):
        wd.dump_stack("timeout")
    logger.warning.assert_called_once()
    message = logger.warning.call_args[0][0]
    assert "[Watchdog]Failed to dump stack trace" in message
    assert wd._dump_file in message
    assert "boom" in message
    assert wd._dump_seq == 1  # not incremented on failure


def test_dump_stack_swallows_failure_without_logger(tmp_path):
    """Verify dump failures are silently ignored without a logger."""
    with patch("vllm.utils.watch_dog.get_user_root_dir", return_value=str(tmp_path)):
        wd = WatchDog()
    with patch("vllm.utils.watch_dog.safe_open_file", side_effect=OSError("boom")):
        wd.dump_stack("timeout")  # must not raise
    assert wd._dump_seq == 1  # not incremented on failure


def test_start_launches_daemon_thread():
    """Verify start() spawns a live daemon monitor thread."""
    wd = WatchDog()
    with patch("vllm.utils.watch_dog.prepare_private_dir"):
        wd.start()
    assert wd._thread is not None
    assert wd._thread.is_alive()
    assert wd._thread.daemon is True
    assert not wd._stop_event.is_set()
    wd.stop()


def test_start_is_noop_when_thread_already_running():
    """Verify a second start() does not spawn another monitor thread."""
    wd = WatchDog()
    with patch("vllm.utils.watch_dog.prepare_private_dir"):
        wd.start()
        original_thread = wd._thread
        wd.start()
    assert wd._thread is original_thread
    wd.stop()


def test_stop_sets_event_and_clears_thread():
    """Verify stop() sets the stop event and clears the thread reference."""
    wd = WatchDog()
    with patch("vllm.utils.watch_dog.prepare_private_dir"):
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

    def test_does_nothing_when_not_timed_out(self):
        """Verify the check loop stays quiet while the watchdog is fed."""
        wd = WatchDog()
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
        assert wd.num_timeouts == 0
        assert not wd._is_timeout_active

    def test_counts_timeout_once_and_recovery_on_feed(self):
        """Verify the check loop counts a timeout only once per stale feed
        epoch and feed() records the recovery."""
        wd = WatchDog()
        wd._timeout = 0.01
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

        assert wd.num_timeouts == 1
        assert mock_dump.call_count == 1
        assert wd._is_timeout_active

        wd.feed()  # the process recovers
        assert wd.num_timeouts == 1
        assert wd.num_recoveries == 1
        assert not wd._is_timeout_active

        # A subsequent feed in the normal state must not double-count.
        wd.feed()
        assert wd.num_recoveries == 1
        assert wd.num_timeouts == 1

    def test_take_timeout_stats_returns_and_resets(self):
        """Verify take_timeout_stats() returns the accumulated statistics
        (including the watchdog name) and resets the counters for the next
        reporting window."""
        wd = WatchDog()
        wd._timeout = 0.01
        wd._last_feed_time = time.monotonic() - 10  # already timed out

        with (
            patch.object(wd, "dump_stack"),
            patch.object(
                wd._stop_event,
                "wait",
                side_effect=self._fake_wait_that_exits_after({"n": 0}),
            ),
        ):
            wd._check_loop()

        stats = wd.take_timeout_stats()
        assert stats.name == "vllm"
        assert stats.num_timeouts == 1
        assert stats.num_recoveries == 0

        wd.feed()  # record the recovery
        stats = wd.take_timeout_stats()
        assert stats.name == "vllm"
        assert stats.num_timeouts == 0
        assert stats.num_recoveries == 1

        assert wd.num_timeouts == 0
        assert wd.num_recoveries == 0
        assert not wd._is_timeout_active


def test_get_watch_dog_returns_shared_singleton():
    """Verify get_watch_dog() returns the same WatchDog instance every
    time."""
    first = get_watch_dog()
    second = get_watch_dog()
    assert isinstance(first, WatchDog)
    assert first is second


def test_start_watch_dog_stays_dormant_without_dump_dir():
    """Verify start_watch_dog() keeps the watchdog dormant (not started, no
    state change) when the config disables the feature via empty dump_dir."""
    watchdog = WatchDog()
    with patch("vllm.utils.watch_dog._watch_dog", watchdog):
        result = start_watch_dog("engine_0", WatchdogConfig())
    assert result is watchdog
    assert watchdog._name == "vllm"
    assert watchdog._timeout == 300
    assert watchdog._check_interval == 10
    assert watchdog._thread is None


def test_start_watch_dog_applies_config_and_starts(tmp_path):
    """Verify start_watch_dog() configures and starts the shared watchdog
    background thread when dump_dir is set."""
    watchdog = WatchDog()
    config = WatchdogConfig(timeout=30, check_interval=5, dump_dir=str(tmp_path))
    with patch("vllm.utils.watch_dog._watch_dog", watchdog):
        result = start_watch_dog("engine_7", config)
    try:
        assert result is watchdog
        assert watchdog._name == "engine_7"
        assert watchdog._timeout == 30
        assert watchdog._check_interval == 5
        assert watchdog._dump_dir == str(tmp_path)
        assert watchdog._dump_file == os.path.join(
            str(tmp_path), f"VLLM_STACK_DUMP_for_engine_7_{os.getpid()}.log"
        )
        assert watchdog._thread is not None
        assert watchdog._thread.is_alive()
        assert watchdog._thread.daemon is True
    finally:
        watchdog.stop()


def test_dump_stack_logs_success_via_logger(tmp_path):
    """Verify a successful dump is reported through the configured logger and
    increments the dump sequence."""
    with patch("vllm.utils.watch_dog.get_user_root_dir", return_value=str(tmp_path)):
        wd = WatchDog()
    wd.set_config(dump_dir=str(tmp_path / "dump"))
    os.makedirs(wd._dump_dir)  # prepare the dump directory
    logger = MagicMock()
    wd.set_logger(logger)
    with patch("vllm.utils.watch_dog.safe_open_file", side_effect=open):
        wd.dump_stack("timeout")
    logger.info.assert_called_once()
    message = logger.info.call_args[0][0]
    assert "[Watchdog]Dumped stack to" in message
    assert wd._dump_file in message
    assert "timeout" in message
    assert os.path.exists(wd._dump_file)
    assert wd._dump_seq == 2  # incremented on success
