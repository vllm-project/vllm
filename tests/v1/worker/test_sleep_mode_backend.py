# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the sleep-mode backend abstraction (RFC #34303).

These cover the registry/factory contract and capability flags. They do not
touch CUDA - the ``cumem`` suspend/resume path is exercised end-to-end on GPU
in ``tests/basic_correctness/test_cumem.py``.
"""

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)


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


@pytest.fixture
def backup_allocator(monkeypatch):
    """Use real allocator bookkeeping without allocating device memory."""
    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData

    allocator = cumem.CuMemAllocator()
    allocator.pointer_to_data = {
        100: AllocationData((0, 3, 100, 1), "weights"),
        200: AllocationData((0, 5, 200, 2), "weights"),
        300: AllocationData((0, 100, 300, 3), "kv_cache"),
    }
    monkeypatch.setattr(cumem, "PIN_MEMORY", True)
    return allocator


def test_prepare_sleep_backups_default_disabled(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_SLEEP_PREPARE_BACKUP_MAX_BYTES", raising=False)
    assert envs.VLLM_SLEEP_PREPARE_BACKUP_MAX_BYTES == 0


@pytest.mark.parametrize(
    "budget, expected", [(0, 0), (-1, 0), (8, 0), (11, 0), (12, 12)]
)
def test_prepare_sleep_backups_rounded_budget(
    backup_allocator, monkeypatch, budget, expected
):
    from unittest.mock import Mock

    import torch

    allocate = Mock()
    monkeypatch.setattr(torch, "empty", allocate)
    assert backup_allocator._prepare_sleep_backups(budget) == expected
    assert allocate.call_count == (2 if expected else 0)
    assert all(
        d.cpu_backup_tensor is None for d in backup_allocator.pointer_to_data.values()
    )
    for call in allocate.call_args_list:
        assert call.kwargs["pin_memory"] is True
        assert call.kwargs["device"] == "cpu"


@pytest.mark.parametrize("failure", ["out_of_memory", "memory", "unknown_runtime"])
def test_prepare_sleep_backups_failure_releases_partial_capacity(
    backup_allocator, monkeypatch, failure
):
    import weakref

    import torch

    refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def allocate(*args, **kwargs):
        if refs:
            if failure == "out_of_memory":
                raise torch.OutOfMemoryError("injected pinned allocation failure")
            if failure == "memory":
                raise MemoryError("injected host allocation failure")
            raise RuntimeError("injected unknown CUDA failure")
        tensor = torch.Tensor([1])
        refs.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    if failure == "unknown_runtime":
        with pytest.raises(RuntimeError, match="injected unknown CUDA failure"):
            backup_allocator._prepare_sleep_backups(12)
    else:
        assert backup_allocator._prepare_sleep_backups(12) == 0
    assert refs[0]() is None
    assert not backup_allocator._prepared_sleep_backups
    assert all(not d.is_asleep for d in backup_allocator.pointer_to_data.values())
    assert all(
        d.cpu_backup_tensor is None for d in backup_allocator.pointer_to_data.values()
    )


@pytest.mark.parametrize(
    "action", ["clear", "close", "malloc", "free", "discard", "level2"]
)
def test_prepare_sleep_backups_cancellation(backup_allocator, monkeypatch, action):
    import weakref

    import torch

    import vllm.device_allocator.cumem as cumem

    refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def allocate(*args, **kwargs):
        tensor = torch.Tensor([1])
        refs.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args: None)
    monkeypatch.setattr(cumem, "unmap_and_release", lambda handle: None)
    assert backup_allocator._prepare_sleep_backups(12) == 12
    if action == "clear":
        backup_allocator._clear_prepared_sleep_backups()
    elif action == "close":
        backup_allocator.close()
    elif action == "malloc":
        backup_allocator.current_tag = "weights"
        backup_allocator._python_malloc_callback((0, 7, 400, 4))
    elif action == "free":
        backup_allocator._python_free_callback(100)
    elif action == "discard":
        backup_allocator.discard("weights")
    else:
        backup_allocator.sleep(offload_tags=())
    assert not backup_allocator._prepared_sleep_backups
    assert all(ref() is None for ref in refs)


def test_prepare_sleep_backups_unrelated_allocations(backup_allocator, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: torch.Tensor([1]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    assert backup_allocator._prepare_sleep_backups(12) == 12
    backup_allocator.current_tag = "kv_cache"
    backup_allocator._python_malloc_callback((0, 7, 400, 4))
    backup_allocator._python_free_callback(400)
    assert len(backup_allocator._prepared_sleep_backups) == 2
    backup_allocator._clear_prepared_sleep_backups()


@pytest.mark.parametrize("change", ["identity", "handle"])
def test_prepare_sleep_backups_rejects_stale_allocation(
    backup_allocator, monkeypatch, change
):
    from unittest.mock import Mock

    import torch

    import vllm.device_allocator.cumem as cumem
    from vllm.device_allocator import AllocationData

    allocate = Mock(side_effect=lambda *args, **kwargs: torch.Tensor([1]))
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(cumem, "unmap_and_release", lambda handle: None)
    monkeypatch.setattr(cumem.libcudart, "cudaMemcpy", lambda *args: None)
    assert backup_allocator._prepare_sleep_backups(12) == 12
    data = backup_allocator.pointer_to_data[100]
    if change == "identity":
        backup_allocator.pointer_to_data[100] = AllocationData(data.handle, "weights")
    else:
        data.handle = (1, 3, 100, 1)
    allocate.reset_mock()
    backup_allocator.sleep(offload_tags=("weights",))
    assert allocate.call_count == 2
    assert not backup_allocator._prepared_sleep_backups


@pytest.mark.parametrize("operation", ["reload", "shutdown_error"])
def test_worker_cancels_prepared_backups(backup_allocator, monkeypatch, operation):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    import vllm.device_allocator.cumem as cumem
    import vllm.v1.worker.gpu_worker as gpu_worker

    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: torch.Tensor([1]))
    monkeypatch.setattr(cumem.CuMemAllocator, "instance", backup_allocator)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)
    assert backup_allocator._prepare_sleep_backups(12) == 12
    worker = object.__new__(gpu_worker.Worker)
    if operation == "reload":
        worker.vllm_config = None
        monkeypatch.setattr(
            gpu_worker, "set_current_vllm_config", lambda _: nullcontext()
        )
        calls = []
        worker.model_runner = SimpleNamespace(
            reload_weights=lambda: calls.append(
                bool(backup_allocator._prepared_sleep_backups)
            )
        )
        worker.reload_weights()
        assert calls == [False]
    else:

        def fail_shutdown():
            raise RuntimeError("injected communicator shutdown failure")

        monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_shutdown", fail_shutdown)
        with pytest.raises(
            RuntimeError, match="injected communicator shutdown failure"
        ):
            worker.shutdown()
    assert not backup_allocator._prepared_sleep_backups
