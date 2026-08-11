# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
import time
from typing import Any

import pytest

from vllm.v1.engine.utils import CoreEngineProcManager, _AdoptedProcess


def test_snapshot_waits_for_liveness_monitor():
    process = type(
        "Process",
        (),
        {"pid": 123, "is_alive": lambda self: True},
    )()
    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [process]
    manager._process_lock = threading.Lock()
    manager._monitor_ready = threading.Event()
    manager._replacement_event = threading.Event()
    manager._snapshot_exit_event = threading.Event()
    manager._expected_exit_pids = set()
    manager._snapshot_pid = None
    result: list[int] = []

    thread = threading.Thread(target=lambda: result.append(manager.prepare_snapshot()))
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    manager._monitor_ready.set()
    thread.join(timeout=1)

    assert result == [123]


def test_liveness_monitor_does_not_mask_unexpected_exit(monkeypatch):
    import vllm.v1.engine.utils as engine_utils

    joined: list[int] = []

    class Process:
        def __init__(self, pid, name, sentinel):
            self.pid = pid
            self.name = name
            self.sentinel = sentinel
            self.exitcode = -1

        def is_alive(self):
            return True

        def join(self, timeout=None):
            joined.append(self.pid)

    expected = Process(123, "expected", 1)
    unexpected = Process(456, "unexpected", 2)
    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [expected, unexpected]
    manager.manager_stopped = threading.Event()
    manager.failed_proc_name = None
    manager._process_lock = threading.Lock()
    manager._replacement_event = threading.Event()
    manager._monitor_ready = threading.Event()
    manager._snapshot_exit_event = threading.Event()
    manager._expected_exit_pids = {123}
    manager._snapshot_pid = None
    wait_calls = 0

    def wait(sentinels, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return [1, 2]
        manager.manager_stopped.set()
        return []

    shutdown: list[bool] = []

    def stop():
        shutdown.append(True)
        manager.manager_stopped.set()

    monkeypatch.setattr(engine_utils.connection, "wait", wait)
    monkeypatch.setattr(manager, "shutdown", stop)

    manager.monitor_engine_liveness()

    assert joined == [123]
    assert manager.failed_proc_name == "unexpected"
    assert shutdown == [True]


def test_discard_restored_process_keeps_expected_exit_until_monitored(monkeypatch):
    import vllm.v1.engine.utils as engine_utils

    alive = iter((True, False))
    process = type(
        "Process",
        (),
        {
            "pid": 123,
            "is_alive": lambda self: next(alive),
            "join": lambda self, timeout=None: None,
        },
    )()
    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [process]
    manager._process_lock = threading.Lock()
    manager._replacement_event = threading.Event()
    manager._expected_exit_pids = set()
    manager._snapshot_pid = None
    monkeypatch.setattr(engine_utils, "kill_process_tree", lambda pid: None)

    manager.discard_restored_process()

    assert manager._expected_exit_pids == {123}


def test_discard_restored_process_unmarks_failed_exit(monkeypatch):
    import vllm.v1.engine.utils as engine_utils

    process = type(
        "Process",
        (),
        {
            "pid": 123,
            "is_alive": lambda self: True,
            "join": lambda self, timeout=None: None,
        },
    )()
    manager = object.__new__(CoreEngineProcManager)
    manager.processes = [process]
    manager._process_lock = threading.Lock()
    manager._replacement_event = threading.Event()
    manager._expected_exit_pids = set()
    manager._snapshot_pid = None
    monkeypatch.setattr(engine_utils, "kill_process_tree", lambda pid: None)

    with pytest.raises(RuntimeError, match="did not exit"):
        manager.discard_restored_process()

    assert manager._expected_exit_pids == set()


def test_adopted_process_close_is_idempotent(monkeypatch):
    process = object.__new__(_AdoptedProcess)
    process._pidfd = 123
    closed: list[int] = []
    monkeypatch.setattr(os, "close", closed.append)

    process.close()
    process.close()

    assert closed == [123]


def test_snapshot_stdio_rebinds_to_source_process(monkeypatch):
    import vllm.snapshot.process as snapshot_process

    opened: list[tuple[str, int]] = []
    duplicated: list[tuple[int, int, bool]] = []
    closed: list[int] = []
    source_fds = iter((101, 102))
    monkeypatch.setattr(snapshot_process, "process_starttime", lambda pid: 456)

    def open_fd(path, flags):
        opened.append((path, flags))
        return next(source_fds)

    monkeypatch.setattr(os, "open", open_fd)
    monkeypatch.setattr(
        os,
        "dup2",
        lambda source, target, inheritable=True: duplicated.append(
            (source, target, inheritable)
        ),
    )
    monkeypatch.setattr(os, "close", closed.append)

    snapshot_process.rebind_stdio(123, expected_starttime=456)

    assert [path for path, _ in opened] == [
        "/proc/123/fd/1",
        "/proc/123/fd/2",
    ]
    assert duplicated == [(101, 1, True), (102, 2, True)]
    assert closed == [101, 102]


def test_snapshot_io_detach_requires_opt_in():
    from vllm.v1.engine.core import EngineCoreProc

    engine = object.__new__(EngineCoreProc)
    engine._engine_snapshot_enabled = False

    with pytest.raises(RuntimeError, match="snapshots are not enabled"):
        engine.snapshot_detach_io("nonce", 1, "snapshot", "config", "marker", "durable")


def test_snapshot_stdio_rejects_reused_source_pid(monkeypatch):
    import vllm.snapshot.process as snapshot_process

    monkeypatch.setattr(snapshot_process, "process_starttime", lambda pid: 789)

    with pytest.raises(RuntimeError, match="source process identity changed"):
        snapshot_process.rebind_stdio(123, expected_starttime=456)


def test_snapshot_stdio_rejects_source_change_during_open(monkeypatch):
    import vllm.snapshot.process as snapshot_process

    starttimes = iter((456, 789))
    duplicated: list[Any] = []
    closed: list[int] = []
    monkeypatch.setattr(
        snapshot_process, "process_starttime", lambda pid: next(starttimes)
    )
    source_fds = iter((100, 101))
    monkeypatch.setattr(os, "open", lambda path, flags: next(source_fds))
    monkeypatch.setattr(os, "dup2", lambda *args, **kwargs: duplicated.append(args))
    monkeypatch.setattr(os, "close", closed.append)

    with pytest.raises(RuntimeError, match="source process identity changed"):
        snapshot_process.rebind_stdio(123)

    assert duplicated == []
    assert closed == [100, 101]


@pytest.mark.parametrize(
    ("persistence", "expected_fsync_calls"),
    (("durable", 1), ("page_cache", 0)),
)
def test_snapshot_detach_marker_honors_persistence(
    tmp_path, monkeypatch, persistence, expected_fsync_calls
):
    from vllm.v1.engine.core import EngineCoreProc

    engine = object.__new__(EngineCoreProc)
    engine._engine_snapshot_enabled = True
    engine._snapshot_input_stopped = threading.Event()
    engine._snapshot_output_stopped = threading.Event()
    engine._snapshot_input_stopped.set()
    engine._snapshot_output_stopped.set()
    engine._snapshot_io_lock = threading.Lock()
    engine._snapshot_io_active = True
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", fsync_calls.append)

    marker_path = tmp_path / "detached.json"
    engine.snapshot_detach_io(
        "nonce",
        1,
        "snapshot",
        "config",
        str(marker_path),
        persistence,
    )
    engine._finish_snapshot_io_detach()

    marker = json.loads(marker_path.read_text())
    assert marker["state"] == "detached"
    assert len(fsync_calls) == expected_fsync_calls
