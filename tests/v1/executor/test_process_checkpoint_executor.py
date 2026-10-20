# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle tests for the opt-in process checkpoint executor."""

import multiprocessing
import threading
import time
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


def test_shutdown_does_not_restore_checkpointed_memory(executor):
    obj, calls = executor
    obj.sleep()

    def fail():
        raise RuntimeError("restore out of memory")

    obj._checkpoint.resume = fail
    obj.shutdown()
    assert not obj.is_failed
    assert ("shutdown",) in calls and ("close",) in calls
    assert ("wake", None) not in calls


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
@pytest.mark.parametrize("p2p,shm", [("0", "1"), ("0", "0"), ("1", "0"), (None, None)])
@pytest.mark.parametrize("tp", [2, 3, 8])
def test_transport_requires_suspension_and_no_nvls(
    monkeypatch, nvls, suspend, accepted, fusion, backend, p2p, shm, tp
):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    env = {
        "NCCL_P2P_DISABLE": p2p,
        "NCCL_SHM_DISABLE": shm,
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
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
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
            tensor_parallel_size=tp,
            nnodes=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            world_size=tp,
            disable_custom_all_reduce=True,
            enable_expert_parallel=False,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            enable_nccl_comm_suspend=suspend,
            sleep_mode_backend="cumem",
        ),
        use_v2_model_runner=True,
        weight_transfer_config=None,
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
    proc.is_alive.return_value = True
    proc.exitcode = -9
    with patch.object(MultiprocExecutor, "_ensure_worker_termination") as terminate:
        ProcessCheckpointExecutor._ensure_worker_termination([proc])
    terminate.assert_not_called()
    proc.kill.assert_called_once()
    proc.join.assert_called_once()
    assert 0 <= proc.join.call_args.kwargs["timeout"] <= 5


def test_worker_termination_reaps_real_processes():
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=time.sleep, args=(60,)) for _ in range(2)]
    try:
        for proc in processes:
            proc.start()
        ProcessCheckpointExecutor._ensure_worker_termination(processes)
        assert all(proc.exitcode is not None for proc in processes)
        assert not any(proc.is_alive() for proc in processes)
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
            proc.close()


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


@pytest.mark.parametrize(
    "component,field,value",
    [
        ("parallel_config", "tensor_parallel_size", 0),
        ("parallel_config", "tensor_parallel_size", -1),
        ("parallel_config", "nnodes", 2),
        ("parallel_config", "enable_expert_parallel", True),
        ("parallel_config", "worker_cls", "custom.Worker"),
        ("model_config", "sleep_mode_backend", "custom"),
        (None, "use_v2_model_runner", False),
        (None, "weight_transfer_config", object()),
        ("platform", "is_cuda", False),
        ("platform", "linux", False),
    ],
)
def test_unsupported_checkpoint_configuration_fails_before_workers(
    component, field, value, monkeypatch
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            world_size=1,
            nnodes=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=True,
            enable_nccl_comm_suspend=False,
            sleep_mode_backend="cumem",
        ),
        use_v2_model_runner=True,
        weight_transfer_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
    )
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    if component == "platform":
        if field == "is_cuda":
            monkeypatch.setattr(
                "vllm.platforms.current_platform.is_cuda", lambda: value
            )
        else:
            monkeypatch.setattr("sys.platform", "darwin")
    else:
        setattr(getattr(config, component) if component else config, field, value)
    with (
        patch.object(MultiprocExecutor, "__init__") as initialize,
        patch.object(MultiprocExecutor, "shutdown"),
        pytest.raises(ValueError, match="checkpoint"),
    ):
        ProcessCheckpointExecutor(config)
    initialize.assert_not_called()


@pytest.mark.parametrize(
    "module,name",
    [
        ("cuda_checkpoint_executor", "CUDACheckpointExecutor"),
        ("process_checkpoint_executor", "ProcessCheckpointExecutor"),
    ],
)
@pytest.mark.parametrize("use_class", [False, True])
def test_checkpoint_executor_names_resolve(module, name, use_class):
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.executor.cuda_checkpoint_executor import CUDACheckpointExecutor

    backend = f"vllm.v1.executor.{module}.{name}"
    if use_class:
        backend = resolve_obj_by_qualname(backend)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=backend)
    )
    assert Executor.get_class(config) is CUDACheckpointExecutor
