# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controller ordering and helper cleanup without CUDA initialization."""

from unittest.mock import Mock, patch

import pytest

from vllm.utils.cuda_process_checkpoint import CudaProcessCheckpoint, Driver


def test_all_ranks_restore_before_any_unlock():
    events = []
    with patch("vllm.utils.cuda_process_checkpoint.Driver") as driver:
        driver.return_value.side_effect = lambda pid, op: events.append((pid, op))
        checkpoint = CudaProcessCheckpoint(lambda **event: None)
        checkpoint.suspend([11, 22, 33, 44])
        checkpoint.resume()
    assert events == [
        (pid, op)
        for op in ("Lock", "Checkpoint", "Restore", "Unlock")
        for pid in (11, 22, 33, 44)
    ]
    assert checkpoint.pids == []


def test_partial_failure_prevents_further_rank_operations():
    events = []

    def invoke(pid, op):
        events.append((pid, op))
        if pid == 22 and op == "Restore":
            raise RuntimeError("injected restore failure")

    with patch("vllm.utils.cuda_process_checkpoint.Driver") as driver:
        driver.return_value.side_effect = invoke
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
