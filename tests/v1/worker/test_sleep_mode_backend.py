# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the sleep-mode backend abstraction (RFC #34303).

These cover the registry/factory contract and capability flags. They do not
touch CUDA - the ``cumem`` suspend/resume path is exercised end-to-end on GPU
in ``tests/basic_correctness/test_cumem.py``.
"""

from unittest.mock import MagicMock

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)


@pytest.mark.parametrize("operation", ["workspace_clear", "driver_allocation"])
def test_graph_discard_preserves_workspaces_and_avoids_driver_pressure(
    monkeypatch, operation
):
    """Graph destruction must not clear unrelated workspaces or force an OOM."""
    from types import SimpleNamespace

    driver = pytest.importorskip("cuda.bindings.driver")
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker.model_runner = MagicMock()
    worker._sleep_mode_backend = MagicMock()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_nccl_comm_suspend=False)
    )
    monkeypatch.setattr("vllm.envs.VLLM_SLEEP_DISCARD_GRAPHS", True)
    # A stale deployment setting must not re-enable the removed operation.
    monkeypatch.setenv("VLLM_SLEEP_RECLAIM_GRAPH_MEMORY", "1")
    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.get_memory_info", lambda: (1024, 4096))
    monkeypatch.setattr("torch.accelerator.empty_cache", lambda: None)
    workspace_clear = MagicMock()
    monkeypatch.setattr(
        "torch._C._cuda_clearCublasWorkspaces", workspace_clear, raising=False
    )
    driver_allocation = MagicMock(
        return_value=(driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY, 0)
    )
    monkeypatch.setattr(driver, "cuMemAlloc", driver_allocation)

    worker.sleep(level=1)

    worker.model_runner.cudagraph_manager.discard_full_graphs.assert_called_once()
    worker._sleep_mode_backend.suspend.assert_called_once_with(1)
    if operation == "workspace_clear":
        workspace_clear.assert_not_called()
    else:
        driver_allocation.assert_not_called()


@pytest.mark.parametrize("unsupported", [None, "tp", "compile", "inproc", "checkpoint"])
def test_graph_sleep_rejects_unsupported_resource_owners(monkeypatch, unsupported):
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.v1.worker.gpu.sleep_graphs import validate_graph_sleep

    config = MagicMock()
    config.use_v2_model_runner = True
    mc = config.model_config
    mc.enable_sleep_mode = True
    mc.sleep_mode_backend = "cumem"
    mc.enforce_eager = mc.enable_nccl_comm_suspend = False
    mc.quantization = None
    mc.architectures = ["Qwen3ForCausalLM"]
    pc = config.parallel_config
    pc.world_size = pc.data_parallel_size = 1
    pc.prefill_context_parallel_size = pc.decode_context_parallel_size = 1
    pc.distributed_executor_backend = "uni"
    pc.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    pc.use_ubatching = config.scheduler_config.async_scheduling = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.compile_sizes = None
    config.lora_config = config.speculative_config = None
    config.kv_transfer_config = config.weight_transfer_config = None
    config.offload_config.uva.cpu_offload_gb = 0
    config.offload_config.prefetch.offload_group_size = 0
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    monkeypatch.setattr("vllm.envs.VLLM_ENABLE_V1_MULTIPROCESSING", True)
    monkeypatch.delenv("VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", raising=False)
    if unsupported == "tp":
        pc.world_size = 2
    elif unsupported == "compile":
        config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    elif unsupported == "inproc":
        monkeypatch.setattr("vllm.envs.VLLM_ENABLE_V1_MULTIPROCESSING", False)
    elif unsupported == "checkpoint":
        monkeypatch.setenv("VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", "1")
    if unsupported:
        with pytest.raises(ValueError, match="Graph-discard sleep requires"):
            validate_graph_sleep(config)
    else:
        validate_graph_sleep(config)


@pytest.mark.parametrize("poisoned", [False, True])
def test_graph_recapture_oom_requires_a_healthy_device(monkeypatch, poisoned):
    import torch

    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker.model_runner = MagicMock()
    worker.model_runner.capture_one_sleep_graph.side_effect = torch.OutOfMemoryError()
    worker.synchronize_device = MagicMock()
    monkeypatch.setattr("torch.accelerator.empty_cache", lambda: None)
    manager = worker.model_runner.cudagraph_manager
    if poisoned:
        worker.synchronize_device.side_effect = RuntimeError("device failure")
        with pytest.raises(RuntimeError, match="device failure"):
            worker.recapture_sleep_graph()
        manager.discard_full_graphs.assert_not_called()
    else:
        assert worker.recapture_sleep_graph() == 0
        manager.discard_full_graphs.assert_called_once()


def test_cumem_is_the_default_registered_backend():
    backend_cls = SleepModeBackendFactory.get_backend_class("cumem")
    assert backend_cls is CuMemBackend
    assert issubclass(backend_cls, SleepModeBackend)


def test_cumem_capability_flags():
    # cumem leaves communicators untouched but does not preserve compiled
    # artifacts, graphs, or durable state - these flags are what the executor and
    # /health introspect to decide reinit / persistence behavior.
    assert CuMemBackend.is_supported() is True
    assert CuMemBackend.preserves_communicators() is True
    assert CuMemBackend.preserves_compiled_artifacts() is False
    assert CuMemBackend.preserves_graphs_with_communicators() is False
    assert CuMemBackend.supports_durable_storage() is False


def test_new_backend_starts_in_running_state():
    # Constructing a backend must not touch the GPU; only suspend/resume do.
    assert CuMemBackend().state() == "RUNNING"


@pytest.mark.parametrize("enable_nccl_comm_suspend", [True, False])
def test_worker_drives_communicator_suspension(monkeypatch, enable_nccl_comm_suspend):
    """Comm walkers run around sleep/wake only when explicitly enabled."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    calls: list[tuple[str, object]] = []

    class Backend:
        def suspend(self, level: int = 1) -> None:
            calls.append(("backend.suspend", level))

        def resume(self, tags: list[str] | None = None) -> None:
            calls.append(("backend.resume", tuple(tags) if tags else None))

    worker = object.__new__(Worker)
    worker._sleep_mode_backend = Backend()
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_nccl_comm_suspend=enable_nccl_comm_suspend)
    )

    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.get_memory_info", lambda: (0, 0))
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.suspend_device_comms",
        lambda: calls.append(("comms.suspend", None)),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.resume_device_comms",
        lambda: calls.append(("comms.resume", None)),
    )

    worker.sleep(level=1)
    worker.wake_up(tags=["weights"])

    expected = [
        ("backend.suspend", 1),
        ("comms.suspend", None),
        ("backend.resume", ("weights",)),
        ("comms.resume", None),
    ]
    if not enable_nccl_comm_suspend:
        expected = [c for c in expected if not c[0].startswith("comms.")]
    assert calls == expected


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported sleep-mode backend"):
        SleepModeBackendFactory.get_backend_class("does-not-exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        SleepModeBackendFactory.register_backend(
            "cumem",
            "vllm.device_allocator.sleep_mode_backend",
            "CuMemBackend",
        )


def test_third_party_backend_registration_and_resolution():
    """A plugin registers a backend by name; the factory resolves it lazily."""
    name = "_pytest_dummy_backend"
    try:
        SleepModeBackendFactory.register_backend(
            name,
            "tests.v1.worker.test_sleep_mode_backend",
            "DummyBackend",
        )
        resolved = SleepModeBackendFactory.get_backend_class(name)
        assert resolved is DummyBackend
        assert resolved.supports_durable_storage() is True
    finally:
        SleepModeBackendFactory._registry.pop(name, None)


def test_suspend_resume_state_transitions():
    """Lifecycle state advances RUNNING -> SUSPENDED -> RUNNING without GPU."""
    backend = DummyBackend()
    assert backend.state() == "RUNNING"
    backend.suspend(level=1)
    assert backend.state() == "SUSPENDED"
    backend.resume()
    assert backend.state() == "RUNNING"


class DummyBackend(SleepModeBackend):
    """A no-GPU backend used to exercise lifecycle + registration in CPU tests."""

    def suspend(self, level: int = 1) -> None:
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RUNNING"

    @classmethod
    def supports_durable_storage(cls) -> bool:
        return True
