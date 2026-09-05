# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, call

import pytest

from vllm.device_allocator.cuda_checkpoint import CudaCheckpointer
from vllm.v1.worker.gpu_worker import Worker


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("recovery_fails", [False, True])
@pytest.mark.parametrize("reinit_fails", [False, True])
def test_suspend_failure_recovers_before_reinit(
    monkeypatch, error_type, recovery_fails, reinit_fails
):
    calls = Mock()
    worker = object.__new__(Worker)
    worker._destroy_nccl_communicators = calls.destroy
    worker._reinit_nccl_communicators = calls.reinit
    monkeypatch.setattr(CudaCheckpointer, "get_instance", lambda: calls.checkpointer)
    original_error = error_type("suspend failed")
    calls.checkpointer.suspend.side_effect = original_error
    if recovery_fails:
        calls.checkpointer.recover.side_effect = RuntimeError("recovery failed")
    if reinit_fails:
        calls.reinit.side_effect = RuntimeError("reinit failed")

    with pytest.raises(error_type) as exc_info:
        worker.suspend()

    assert exc_info.value is original_error
    assert calls.mock_calls == [
        call.destroy(),
        call.checkpointer.suspend(),
        call.checkpointer.recover(),
        call.reinit(),
    ]


def test_suspend_success_leaves_communicators_destroyed(monkeypatch):
    calls = Mock()
    worker = object.__new__(Worker)
    worker._destroy_nccl_communicators = calls.destroy
    worker._reinit_nccl_communicators = calls.reinit
    monkeypatch.setattr(CudaCheckpointer, "get_instance", lambda: calls.checkpointer)
    calls.checkpointer.suspend.return_value = 123

    assert worker.suspend() == {"checkpoint_handle": 123}
    assert calls.mock_calls == [call.destroy(), call.checkpointer.suspend()]
