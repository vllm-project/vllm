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
