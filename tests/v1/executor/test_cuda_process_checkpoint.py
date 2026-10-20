# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controller ordering and helper cleanup without CUDA initialization."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from vllm.utils.cuda_process_checkpoint import CudaProcessCheckpoint, Driver


@pytest.mark.parametrize("rank_count", [1, 2, 3, 4, 8])
def test_all_ranks_restore_before_any_unlock(rank_count):
    events: list[tuple[int, str]] = []
    with patch("vllm.utils.cuda_process_checkpoint.Driver") as driver:
        driver.return_value.run_stage.side_effect = lambda pids, op: events.extend(
            (pid, op) for pid in pids
        )
        checkpoint = CudaProcessCheckpoint(lambda **event: None)
        checkpoint.suspend(list(range(1, rank_count + 1)))
        checkpoint.resume()
    assert events == [
        (pid, op)
        for op in ("Lock", "Checkpoint", "Restore", "Unlock")
        for pid in range(1, rank_count + 1)
    ]
    assert checkpoint.pids == []


def test_partial_failure_prevents_further_rank_operations():
    events = []

    def invoke(pid, op):
        events.append((pid, op))
        if pid == 22 and op == "Restore":
            raise RuntimeError("injected restore failure")

    with patch("vllm.utils.cuda_process_checkpoint.Driver") as driver:
        driver.return_value.run_stage.side_effect = lambda pids, op: [
            invoke(pid, op) for pid in pids
        ]
        checkpoint = CudaProcessCheckpoint(lambda **event: None)
        checkpoint.suspend([11, 22])
        with pytest.raises(RuntimeError, match="injected"):
            checkpoint.resume()
        count = len(events)
        with pytest.raises(RuntimeError, match="restart required"):
            checkpoint.resume()
        assert len(events) == count
        assert not any(op == "Unlock" for _, op in events)
        assert checkpoint.failed


def test_helper_initialization_failure_reaps_process():
    proc = Mock()
    proc.poll.return_value = None
    with (
        patch("subprocess.Popen", return_value=proc),
        patch.object(Driver, "_read", side_effect=RuntimeError("init")),
        pytest.raises(RuntimeError, match="init"),
    ):
        Driver(lambda **event: None)
    proc.kill.assert_called_once()
    proc.communicate.assert_called_once_with(timeout=5)


def test_helper_read_timeout_kills_process():
    driver = Driver.__new__(Driver)
    driver.proc = Mock()
    selector = Mock()
    selector.select.return_value = []
    with (
        patch("selectors.DefaultSelector", return_value=selector),
        pytest.raises(TimeoutError, match="deadline"),
    ):
        driver._read(0)
    driver.proc.kill.assert_called_once()
    selector.close.assert_called_once()


def test_helper_close_terminates_blocked_driver_call():
    driver = Driver.__new__(Driver)
    driver.proc = Mock()
    driver.proc.poll.return_value = None
    driver.close()
    driver.proc.kill.assert_called_once()
    driver.proc.communicate.assert_called_once_with(timeout=1)


def test_helper_close_reaps_real_process():
    driver = Driver.__new__(Driver)
    driver.proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        driver.close()
        assert driver.proc.returncode is not None
        assert driver.proc.stdin is not None and driver.proc.stdout is not None
        assert driver.proc.stdin.closed and driver.proc.stdout.closed
    finally:
        if driver.proc.poll() is None:
            driver.proc.kill()
            driver.proc.wait(timeout=5)


def test_driver_stage_runs_ranks_concurrently():
    import threading

    from vllm.utils.cuda_process_checkpoint import _run_parallel_stage

    pids = [11, 22, 33, 44]
    barrier = threading.Barrier(len(pids), timeout=2)

    def invoke(pid, op):
        barrier.wait()
        return {"pid": pid, "op": op, "rc": 0}

    rows = _run_parallel_stage(pids, "Restore", invoke)
    assert rows == [{"pid": pid, "op": "Restore", "rc": 0} for pid in pids]


def test_driver_stage_collects_rank_failure():
    from vllm.utils.cuda_process_checkpoint import _run_parallel_stage

    def invoke(pid, op):
        if pid == 22:
            raise RuntimeError("injected")
        return {"pid": pid, "op": op, "rc": 0}

    rows = _run_parallel_stage([11, 22], "Restore", invoke)
    assert rows[0]["rc"] == 0
    assert rows[1]["pid"] == 22 and rows[1]["op"] == "Restore"
    assert "injected" in rows[1]["error"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "wrong_op", "bad_state", "failed"]
)
def test_driver_rejects_invalid_stage_response(fault, monkeypatch):
    import io

    driver = Driver.__new__(Driver)
    driver.emit = Mock()
    driver.proc = Mock(stdin=io.StringIO())
    rows = [
        {"pid": pid, "op": "Restore", "rc": 0, "after": {"rc": 0, "value": 1}}
        for pid in [11, 22]
    ]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["pid"] = 11
    elif fault == "wrong_op":
        rows[1]["op"] = "Unlock"
    elif fault == "bad_state":
        rows[1]["after"] = {"rc": 0, "value": 2}
    else:
        rows[1]["rc"] = 2
    monkeypatch.setattr(
        driver,
        "_read",
        Mock(return_value={"kind": "stage", "op": "Restore", "results": rows}),
    )
    with pytest.raises(RuntimeError):
        driver.run_stage([11, 22], "Restore")


@pytest.mark.parametrize("response", [None, [], "bad", 0])
def test_driver_rejects_non_object_response(response, monkeypatch):
    import io

    driver = Driver.__new__(Driver)
    driver.emit = Mock()
    driver.proc = Mock(stdin=io.StringIO())
    monkeypatch.setattr(driver, "_read", Mock(return_value=response))
    with pytest.raises(RuntimeError, match="Invalid CUDA checkpoint stage response"):
        driver.run_stage([11], "Restore")


def test_driver_accepts_complete_stage_response(monkeypatch):
    import io
    import json

    driver = Driver.__new__(Driver)
    driver.emit = Mock()
    driver.proc = Mock(stdin=io.StringIO())
    rows = [
        {"pid": pid, "op": "Restore", "rc": 0, "after": {"rc": 0, "value": 1}}
        for pid in [11, 22]
    ]
    monkeypatch.setattr(
        driver,
        "_read",
        Mock(return_value={"kind": "stage", "op": "Restore", "results": rows}),
    )
    driver.run_stage([11, 22], "Restore")
    assert json.loads(driver.proc.stdin.getvalue()) == [[11, 22], "Restore"]
    for row in rows:
        driver.emit.assert_any_call(**row)


def test_rank_failure_with_blocked_peer_times_out_and_reaps_helper():
    import textwrap

    driver = Driver.__new__(Driver)
    driver.emit = Mock()
    driver.proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            textwrap.dedent("""
                import json
                import sys
                import time
                from vllm.utils.cuda_process_checkpoint import _run_parallel_stage

                def invoke(pid, op):
                    if pid == 11:
                        raise RuntimeError("injected rank failure")
                    time.sleep(60)

                print(json.dumps({"kind": "ready"}), flush=True)
                pids, op = json.loads(sys.stdin.readline())
                rows = _run_parallel_stage(pids, op, invoke)
                print(json.dumps({"kind": "stage", "op": op, "results": rows}))
            """),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert driver._read(10) == {"kind": "ready"}
        read = driver._read
        with (
            patch.object(driver, "_read", side_effect=lambda timeout: read(0.2)),
            pytest.raises(TimeoutError, match="deadline"),
        ):
            driver.run_stage([11, 22], "Restore")
        driver.proc.wait(timeout=5)
        assert driver.proc.returncode != 0
    finally:
        driver.close()
