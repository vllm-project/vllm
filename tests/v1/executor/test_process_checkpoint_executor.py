# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle tests for the opt-in process checkpoint executor."""

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.process_checkpoint_executor import ProcessCheckpointExecutor


@pytest.fixture
def executor():
    calls: list[tuple[Any, ...]] = []
    obj = ProcessCheckpointExecutor.__new__(ProcessCheckpointExecutor)
    obj._checkpoint_lock = threading.RLock()
    obj._checkpoint_active = False
    obj._use_nccl_suspend = False
    obj._checkpoint_flashinfer = False
    obj.is_failed = False
    obj.sleeping_tags = set()
    obj.failure_callback = None
    obj.workers = [SimpleNamespace(proc=SimpleNamespace(pid=p)) for p in (11, 22)]
    obj._checkpoint = SimpleNamespace(
        suspend=lambda pids: calls.append(("checkpoint", pids)),
        resume=lambda: calls.append(("restore",)),
        close=lambda: calls.append(("close",)),
    )

    def sleep(self, level=1):
        if "weights" in self.sleeping_tags:
            return
        calls.append(("sleep", level))
        self.sleeping_tags = {"weights", "kv_cache"}

    def wake(self, tags=None):
        calls.append(("wake", tags))
        self.sleeping_tags -= set(tags) if tags else self.sleeping_tags.copy()

    with (
        patch.object(MultiprocExecutor, "sleep", sleep),
        patch.object(MultiprocExecutor, "wake_up", wake),
        patch.object(
            MultiprocExecutor, "shutdown", lambda _: calls.append(("shutdown",))
        ),
    ):
        yield obj, calls
    obj._checkpoint = None


def test_sleep_then_restore_precedes_worker_wake(executor):
    obj, calls = executor
    obj.sleep()
    obj.wake_up()
    assert calls == [
        ("sleep", 1),
        ("checkpoint", [11, 22]),
        ("restore",),
        ("wake", None),
    ]


def test_repeated_sleep_does_not_checkpoint_twice(executor):
    obj, calls = executor
    obj.sleep()
    obj.sleep()
    assert calls == [("sleep", 1), ("checkpoint", [11, 22])]


def test_partial_wake_restores_process_only_once(executor):
    obj, calls = executor
    obj.sleep()
    obj.wake_up(["weights"])
    obj.wake_up(["kv_cache"])
    assert calls.count(("restore",)) == 1
    assert not obj.is_sleeping


def test_invalid_tag_does_not_restore(executor):
    obj, calls = executor
    obj.sleep()
    with pytest.raises(ValueError):
        obj.wake_up(["invalid"])
    assert ("restore",) not in calls


def test_failed_checkpoint_stops_workers_and_blocks_wake(executor):
    obj, calls = executor

    def fail(_):
        raise RuntimeError("partial checkpoint")

    obj._checkpoint.suspend = fail
    with pytest.raises(RuntimeError, match="partial"):
        obj.sleep()
    assert obj.is_failed and ("shutdown",) in calls and ("close",) in calls
    with pytest.raises(RuntimeError, match="failed"):
        obj.wake_up()
    assert not any(c[0] == "wake" for c in calls)


def test_shutdown_still_terminates_when_restore_fails(executor):
    obj, calls = executor
    obj.sleep()

    def fail():
        raise RuntimeError("restore out of memory")

    obj._checkpoint.resume = fail
    obj.shutdown()
    assert obj.is_failed and ("shutdown",) in calls and ("close",) in calls


def test_level_two_rejected_before_sleep(executor):
    obj, calls = executor
    with pytest.raises(ValueError):
        obj.sleep(2)
    assert not calls


def test_checkpointed_worker_rejects_cuda_rpc_without_blocking_wake(executor):
    obj, _ = executor
    obj.sleep()
    with pytest.raises(RuntimeError, match="checkpointed"):
        obj.collective_rpc("profile")
    obj.wake_up()


def test_monitor_shutdown_waits_for_inflight_checkpoint(executor):
    obj, calls = executor
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def suspend(pids):
        entered.set()
        assert release.wait(2)

    obj._checkpoint.suspend = suspend
    obj._checkpoint.close = lambda: closed.set()
    sleeper = threading.Thread(target=obj.sleep)
    closer = threading.Thread(target=obj.shutdown)
    sleeper.start()
    assert entered.wait(2)
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    sleeper.join(2)
    closer.join(2)
    assert not sleeper.is_alive() and not closer.is_alive()
    assert closed.is_set()


def test_repeated_pause_acknowledges_already_completed_device_work(executor):
    obj, _ = executor
    obj.sleep()
    for method in ("synchronize_device", "reset_mm_cache", "reset_encoder_cache"):
        assert obj.collective_rpc(method) == [None, None]
    obj.wake_up()


def test_nonblocking_pause_returns_completed_future(executor):
    obj, _ = executor
    obj.sleep()
    result = obj.collective_rpc("synchronize_device", non_block=True)
    assert result.done() and result.result() == [None, None]


def test_failed_executor_never_acknowledges_pause(executor):
    obj, _ = executor
    obj.sleep()
    obj.is_failed = True
    with pytest.raises(RuntimeError, match="failed"):
        obj.collective_rpc("synchronize_device")


@pytest.mark.parametrize(
    "nvls,suspend,accepted,fusion,backend",
    [
        ("0", True, True, False, None),
        ("1", True, False, False, None),
        ("0", False, False, False, None),
        ("0", True, False, True, None),
        ("0", True, True, True, "trtllm"),
        ("0", True, False, True, "mnnvl"),
    ],
)
def test_p2p_requires_suspension_and_no_nvls(
    monkeypatch, nvls, suspend, accepted, fusion, backend
):
    env = {
        "NCCL_P2P_DISABLE": "0",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_CUMEM_HOST_ENABLE": "0",
        "NCCL_NVLS_ENABLE": nvls,
        "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
        "VLLM_USE_NCCL_SYMM_MEM": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", backend)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode=3 if fusion else 0,
            pass_config=SimpleNamespace(fuse_allreduce_rms=fusion),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            nnodes=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            world_size=2,
            disable_custom_all_reduce=True,
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=True, enable_nccl_comm_suspend=suspend
        ),
        kv_transfer_config=None,
        ec_transfer_config=None,
    )
    with (
        patch.object(MultiprocExecutor, "__init__", return_value=None),
        patch.object(MultiprocExecutor, "shutdown"),
    ):
        if accepted:
            obj = ProcessCheckpointExecutor(config)
            assert obj._checkpoint_flashinfer == fusion
            obj.shutdown()
        else:
            with pytest.raises(ValueError, match="NCCL checkpoint configuration"):
                ProcessCheckpointExecutor(config)


@pytest.mark.parametrize(
    "has_symbols,actual,expected",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_validate_rejects_missing_or_inconsistent_nccl(
    monkeypatch, has_symbols, actual, expected
):
    from vllm.v1.executor.process_checkpoint_executor import _validate_nccl_checkpoint

    comm = SimpleNamespace(
        available=True,
        disabled=False,
        _suspended=actual,
        nccl=SimpleNamespace(has_symbol=lambda _: has_symbols),
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=comm))
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: group)
    with pytest.raises(RuntimeError, match="active suspend/resume"):
        _validate_nccl_checkpoint(None, expected)


def test_flashinfer_hooks_wrap_driver_checkpoint_once(executor):
    obj, calls = executor
    obj._checkpoint_flashinfer = True
    with patch.object(
        obj, "collective_rpc", side_effect=lambda method, **kw: calls.append((method,))
    ):
        obj.sleep()
        obj.wake_up(["weights"])
        obj.wake_up(["kv_cache"])
    assert calls == [
        ("sleep", 1),
        ("checkpoint_prepare",),
        ("checkpoint", [11, 22]),
        ("restore",),
        ("checkpoint_restore",),
        ("wake", ["weights"]),
        ("wake", ["kv_cache"]),
    ]


def test_sleep_worker_rpc_has_deadline(executor):
    obj, _ = executor
    with patch.object(MultiprocExecutor, "collective_rpc", return_value=[]) as rpc:
        obj.collective_rpc("sleep")
        assert rpc.call_args.kwargs["timeout"] == 120
        obj.collective_rpc("wake_up", timeout=7)
        assert rpc.call_args.kwargs["timeout"] == 7


def test_terminated_workers_are_reaped():
    proc = Mock()
    with patch.object(MultiprocExecutor, "_ensure_worker_termination") as terminate:
        ProcessCheckpointExecutor._ensure_worker_termination([proc])
    terminate.assert_called_once_with([proc])
    proc.join.assert_called_once()
    assert 0 <= proc.join.call_args.kwargs["timeout"] <= 5


def test_kv_first_partial_wake_then_sleep_preserves_parent_noop(executor):
    obj, calls = executor
    obj._use_nccl_suspend = True
    obj.collective_rpc = Mock()
    obj.sleep()
    obj.wake_up(["kv_cache"])
    assert obj.sleeping_tags == {"weights"}
    before = list(calls)
    rpc_count = obj.collective_rpc.call_count
    obj.sleep()
    assert calls == before
    assert obj.collective_rpc.call_count == rpc_count
    assert not obj.is_failed
    obj.wake_up(["weights"])
    assert not obj.is_sleeping
