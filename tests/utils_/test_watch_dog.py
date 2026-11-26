# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import regex as re

from vllm.config import WatchdogConfig
from vllm.utils import watch_dog as watch_dog_module
from vllm.utils.watch_dog import (
    NoopWatchDog,
    ThreadedWatchDog,
    get_watch_dog,
    start_watch_dog,
)


@pytest.fixture(autouse=True)
def _reset_global_watch_dog():
    original_singleton = watch_dog_module._watch_dog
    original_handler = ThreadedWatchDog.handler
    yield
    watch_dog_module._watch_dog = original_singleton
    ThreadedWatchDog.handler = original_handler


def _make_watch_dog(dump_dir, **kwargs) -> ThreadedWatchDog:
    config = WatchdogConfig(dump_dir=str(dump_dir), **kwargs)
    return ThreadedWatchDog("engine_0", config, MagicMock())


def _run_check_loop(wd: ThreadedWatchDog, ticks: int) -> int:
    """Run the check loop for `ticks` checks; return the dump count."""
    with (
        patch.object(wd.stop_event, "wait", side_effect=[False] * ticks + [True]),
        patch.object(wd, "handle_failure") as handle_failure,
    ):
        wd._check_loop()
    return handle_failure.call_count


def test_check_loop_dumps_once_per_stall(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.timeout = 0
    assert _run_check_loop(wd, ticks=3) == 1
    # A new feed followed by another stall dumps again.
    wd.feed()
    assert _run_check_loop(wd, ticks=3) == 1


def test_check_loop_stays_quiet_while_fed(tmp_path):
    wd = _make_watch_dog(tmp_path)
    wd.feed()
    assert _run_check_loop(wd, ticks=3) == 0


def test_disable_suspends_detection_and_restarts_window(tmp_path):
    wd = _make_watch_dog(tmp_path, timeout=60)
    wd.last_feed_time = time.monotonic() - 3600
    with wd.disable():
        assert _run_check_loop(wd, ticks=3) == 0
    assert _run_check_loop(wd, ticks=3) == 0


def test_handle_failure_appends_numbered_dumps(tmp_path):
    wd = _make_watch_dog(tmp_path / "dumps")
    wd.handle_failure("feed timeout")
    wd.handle_failure("SIGTERM")

    with open(wd.dump_file) as f:
        content = f.read()
    assert "dump #1" in content and "due to feed timeout" in content
    assert "dump #2" in content and "due to SIGTERM" in content
    assert test_handle_failure_appends_numbered_dumps.__name__ in content
    # New per-thread header format: "--- {name}thread {tid} ---".
    assert re.search(r"--- thread \d+ ---", content)


@pytest.mark.timeout(10)
def test_handle_failure_is_reentrant(tmp_path):
    # A second SIGTERM can re-enter handle_failure() while a dump holds the lock.
    wd = _make_watch_dog(tmp_path)
    with wd.lock:
        wd.handle_failure("SIGTERM")
    with open(wd.dump_file) as f:
        assert "due to SIGTERM" in f.read()


def test_handle_failure_does_not_raise_on_error(tmp_path):
    logger = MagicMock()
    config = WatchdogConfig(dump_dir=str(tmp_path))
    wd = ThreadedWatchDog("engine_0", config, logger)
    with patch("builtins.open", side_effect=OSError("boom")):
        wd.handle_failure("feed timeout")
    logger.warning.assert_called_once()


def test_start_watch_dog_without_dump_dir_stays_noop():
    logger = MagicMock()
    wd = start_watch_dog("engine_0", WatchdogConfig(), logger)
    assert isinstance(wd, NoopWatchDog)
    assert wd is get_watch_dog()
    logger.info.assert_not_called()


def test_start_watch_dog_with_dump_dir(tmp_path):
    config = WatchdogConfig(dump_dir=str(tmp_path))
    with patch("faulthandler.enable") as enable:
        wd = start_watch_dog("worker_1", config, MagicMock())
    assert isinstance(wd, ThreadedWatchDog)
    try:
        assert wd is get_watch_dog()
        enable.assert_called_once_with(all_threads=True)
        assert wd.thread is not None
        assert wd.thread.is_alive() and wd.thread.daemon
        wd.handle_failure("shutdown")
        assert os.path.exists(
            tmp_path / f"VLLM_STACK_DUMP_for_worker_1_{os.getpid()}.log"
        )
    finally:
        wd.stop()


def test_dump_stack_writes_threads_in_reverse_order(tmp_path, monkeypatch):
    wd = _make_watch_dog(tmp_path)
    # Inject a known insertion order so the expected reverse sequence is exact.
    fake_frames = {11: None, 22: None, 33: None}
    monkeypatch.setattr("vllm.utils.watch_dog.sys._current_frames", lambda: fake_frames)

    wd.dump_stack("feed timeout")

    with open(wd.dump_file) as f:
        content = f.read()
    tids = [int(m) for m in re.findall(r"--- ?thread (\d+) ---", content)]
    assert tids == [33, 22, 11]


def test_set_handler_is_classmethod_shared_across_instances(tmp_path):
    calls = []

    def handler(reason):
        calls.append(reason)

    wd_a = _make_watch_dog(tmp_path / "a")
    wd_b = _make_watch_dog(tmp_path / "b")
    # Install via the classmethod; the handler is shared on the class.
    # Use `type(wd_b).handler` to bypass the function descriptor, which would
    # otherwise auto-bind a plain function into a fresh bound method.
    wd_a.set_handler(handler)
    assert ThreadedWatchDog.handler is handler
    assert type(wd_b).handler is handler

    wd_a.handle_failure("reason-a")
    wd_b.handle_failure("reason-b")
    assert calls == ["reason-a", "reason-b"]


def test_handle_failure_invokes_class_installed_handler(tmp_path):
    handler = MagicMock()
    ThreadedWatchDog.set_handler(handler)
    wd = _make_watch_dog(tmp_path)
    wd.handle_failure("feed timeout")
    handler.assert_called_once_with("feed timeout")
    assert os.path.exists(wd.dump_file)


def test_handle_failure_invokes_plain_function_handler(tmp_path):
    # Regression: installing a plain `def` (or lambda) on `cls.handler` must
    # not be auto-bound by the function descriptor into a bound method, which
    # would make `handler(reason)` become `handler(self, reason)` and raise
    # TypeError. See vllm/utils/watch_dog.py::handle_failure.
    calls = []

    def handler(reason):
        calls.append(reason)

    ThreadedWatchDog.set_handler(handler)
    wd = _make_watch_dog(tmp_path)
    wd.handle_failure("feed timeout")
    assert calls == ["feed timeout"]


def test_handle_failure_invokes_lambda_handler(tmp_path):
    # Same descriptor-binding pitfall as
    # test_handle_failure_invokes_plain_function_handler, exercised through
    # a lambda since lambdas are a common real-world handler.
    calls = []
    ThreadedWatchDog.set_handler(lambda reason: calls.append(reason))
    wd = _make_watch_dog(tmp_path)
    wd.handle_failure("feed timeout")
    assert calls == ["feed timeout"]


def test_dump_stack_marks_main_thread(tmp_path, monkeypatch):
    wd = _make_watch_dog(tmp_path)
    wd.main_thread_id = 22
    frames = {11: None, 22: None, 33: None}
    monkeypatch.setattr("vllm.utils.watch_dog.sys._current_frames", lambda: frames)

    wd.dump_stack("feed timeout")

    with open(wd.dump_file) as f:
        content = f.read()
    headers = re.findall(r"--- .*?thread (\d+) ---", content)
    assert headers == ["33", "22", "11"]
    # Only the main-thread header carries the (Main) annotation.
    assert "--- (Main)thread 22 ---" in content
    assert "--- (Main)thread 11 ---" not in content
    assert "--- (Main)thread 33 ---" not in content


@pytest.mark.timeout(60)
def test_dump_stack_finds_main_thread_among_200_threads(tmp_path):
    # Concurrency smoke: with 200 real threads blocked on a barrier, the
    # dump produced on the main thread must (a) include the real main-thread
    # tid, (b) have exactly one "(Main)" header, and (c) carry at least one
    # worker thread in the output.
    import threading as _threading

    num = 200
    barrier = _threading.Barrier(num + 1)

    def worker():
        barrier.wait(timeout=5)
        barrier.wait(timeout=5)

    threads = [
        _threading.Thread(target=worker, name=f"wd_probe_{i}") for i in range(num)
    ]
    for t in threads:
        t.start()

    main_tid = _threading.get_ident()
    wd = _make_watch_dog(tmp_path)
    wd.main_thread_id = main_tid

    try:
        barrier.wait(timeout=5)  # all 200 workers are now blocked
        wd.dump_stack("feed timeout")
        barrier.wait(timeout=5)  # release workers
    finally:
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

    with open(wd.dump_file) as f:
        content = f.read()

    main_headers = re.findall(r"--- \(Main\)thread \d+ ---", content)
    assert len(main_headers) == 1, main_headers

    main_tid_in_dump = f"--- (Main)thread {main_tid} ---" in content
    sample_headers = re.findall(r"--- .*?thread \d+ ---", content)[:20]
    assert main_tid_in_dump, (
        f"main thread {main_tid} missing from dump; headers sampled: {sample_headers}"
    )

    all_tids = set(re.findall(r"--- (?:\(Main\))?thread (\d+) ---", content))
    assert len(all_tids) >= 2, "expected at least the main thread plus a worker"
