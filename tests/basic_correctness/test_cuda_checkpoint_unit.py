# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA checkpoint state transitions with mocked driver calls."""

from unittest.mock import Mock, call

import pytest

from vllm.device_allocator import cuda_checkpoint


@pytest.mark.parametrize("explicit_handle", [False, True])
def test_resume_retries_unlock_without_restoring(monkeypatch, explicit_handle):
    driver = Mock()
    for name in (
        "process_lock",
        "process_checkpoint",
        "process_restore",
        "process_unlock",
    ):
        monkeypatch.setattr(cuda_checkpoint, name, getattr(driver, name))
    monkeypatch.setattr(cuda_checkpoint.torch.cuda, "synchronize", Mock())
    checkpointer = cuda_checkpoint.CudaCheckpointer()
    pid = checkpointer.suspend()
    handle = pid if explicit_handle else None
    driver.reset_mock()
    driver.process_unlock.side_effect = [
        RuntimeError("unlock failed"),
        RuntimeError("unlock failed"),
        None,
    ]

    for _ in range(2):
        with pytest.raises(RuntimeError, match="unlock failed"):
            checkpointer.resume(handle)
        assert checkpointer.is_suspended
        driver.process_restore.assert_called_once_with(pid)
        with pytest.raises(RuntimeError, match="already suspended"):
            checkpointer.suspend()

    # A different handle must not bypass the pending unlock.
    with pytest.raises(RuntimeError, match="restored but still locked"):
        checkpointer.resume(pid + 1)

    checkpointer.resume(handle)
    assert not checkpointer.is_suspended
    assert driver.mock_calls == [
        call.process_restore(pid),
        call.process_unlock(pid),
        call.process_unlock(pid),
        call.process_unlock(pid),
    ]

    # The next checkpoint cycle must restore GPU state again.
    driver.reset_mock()
    driver.process_unlock.side_effect = None
    checkpointer.suspend()
    checkpointer.resume()
    driver.process_restore.assert_called_once_with(pid)
    driver.process_unlock.assert_called_once_with(pid)
    assert not checkpointer.is_suspended


def test_resume_retries_failed_restore(monkeypatch):
    restore = Mock(side_effect=[RuntimeError("restore failed"), None])
    unlock = Mock()
    monkeypatch.setattr(cuda_checkpoint, "process_restore", restore)
    monkeypatch.setattr(cuda_checkpoint, "process_unlock", unlock)
    monkeypatch.setattr(cuda_checkpoint, "process_lock", Mock())
    monkeypatch.setattr(cuda_checkpoint, "process_checkpoint", Mock())
    monkeypatch.setattr(cuda_checkpoint.torch.cuda, "synchronize", Mock())
    checkpointer = cuda_checkpoint.CudaCheckpointer()
    pid = checkpointer.suspend()

    with pytest.raises(RuntimeError, match="restore failed"):
        checkpointer.resume()
    assert checkpointer.is_suspended
    unlock.assert_not_called()

    checkpointer.resume()
    assert restore.call_args_list == [call(pid), call(pid)]
    unlock.assert_called_once_with(pid)
    assert not checkpointer.is_suspended


@pytest.mark.parametrize(
    "state, restore_count, unlock_count",
    [
        (cuda_checkpoint.PROCESS_STATE_RUNNING, 0, 0),
        (cuda_checkpoint.PROCESS_STATE_LOCKED, 0, 1),
        (cuda_checkpoint.PROCESS_STATE_CHECKPOINTED, 1, 1),
    ],
)
def test_recover_partial_suspend(monkeypatch, state, restore_count, unlock_count):
    driver = Mock()
    driver.get_state.return_value = state
    for name in ("get_state", "process_restore", "process_unlock"):
        monkeypatch.setattr(cuda_checkpoint, name, getattr(driver, name))
    checkpointer = cuda_checkpoint.CudaCheckpointer()

    checkpointer.recover()

    pid = cuda_checkpoint.os.getpid()
    expected = [call.get_state(pid)]
    expected += [call.process_restore(pid)] * restore_count
    expected += [call.process_unlock(pid)] * unlock_count
    assert driver.mock_calls == expected
    assert not checkpointer.is_suspended


def test_recover_retains_state_after_unlock_failure(monkeypatch):
    monkeypatch.setattr(
        cuda_checkpoint,
        "get_state",
        Mock(return_value=cuda_checkpoint.PROCESS_STATE_LOCKED),
    )
    restore = Mock()
    unlock = Mock(side_effect=[RuntimeError("unlock failed"), None])
    monkeypatch.setattr(cuda_checkpoint, "process_restore", restore)
    monkeypatch.setattr(cuda_checkpoint, "process_unlock", unlock)
    checkpointer = cuda_checkpoint.CudaCheckpointer()

    with pytest.raises(RuntimeError, match="unlock failed"):
        checkpointer.recover()
    assert checkpointer.is_suspended

    checkpointer.resume()
    restore.assert_not_called()
    assert unlock.call_count == 2
    assert not checkpointer.is_suspended


def test_recover_failed_driver_state(monkeypatch):
    monkeypatch.setattr(
        cuda_checkpoint,
        "get_state",
        Mock(return_value=cuda_checkpoint.PROCESS_STATE_FAILED),
    )
    restore = Mock()
    unlock = Mock()
    monkeypatch.setattr(cuda_checkpoint, "process_restore", restore)
    monkeypatch.setattr(cuda_checkpoint, "process_unlock", unlock)
    checkpointer = cuda_checkpoint.CudaCheckpointer()

    with pytest.raises(RuntimeError, match="Cannot recover CUDA process"):
        checkpointer.recover()
    assert checkpointer.is_suspended
    restore.assert_not_called()
    unlock.assert_not_called()
