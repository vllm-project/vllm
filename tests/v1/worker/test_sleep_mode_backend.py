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
    # cumem preserves communicator identity but does not preserve compiled
    # artifacts, graphs, or durable state. These flags are what the executor and
    # /health introspect to decide reinit and persistence behavior.
    assert CuMemBackend.is_supported() is True
    assert CuMemBackend.preserves_communicators() is True
    assert CuMemBackend.preserves_compiled_artifacts() is False
    assert CuMemBackend.preserves_graphs_with_communicators() is False
    assert CuMemBackend.supports_durable_storage() is False


def test_new_backend_starts_in_running_state():
    # Constructing a backend must not touch the GPU; only suspend/resume do.
    assert CuMemBackend().state() == "RUNNING"


def test_cumem_wraps_allocator_with_communicator_memory_lifecycle(monkeypatch):
    calls = []

    class Allocator:
        def sleep(self, *, offload_tags):
            calls.append(("allocator.sleep", offload_tags))

        def wake_up(self, tags):
            calls.append(("allocator.wake_up", tags))

    monkeypatch.setattr("vllm.device_allocator.get_mem_allocator_instance", Allocator)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.suspend_device_comms",
        lambda: calls.append(("comms.suspend", None)),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.resume_device_comms",
        lambda: calls.append(("comms.resume", None)),
    )

    backend = CuMemBackend()
    backend.suspend(level=1)
    backend.resume(tags=["weights"])

    assert calls == [
        ("allocator.sleep", ("weights",)),
        ("comms.suspend", None),
        ("allocator.wake_up", ["weights"]),
        ("comms.resume", None),
    ]


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
