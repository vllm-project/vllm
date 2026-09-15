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
@pytest.mark.parametrize("checkpoint", ["disabled", "enabled", "failure"])
def test_worker_drives_communicator_suspension(
    monkeypatch, enable_nccl_comm_suspend, checkpoint
):
    """Restore the CUDA context before touching allocator or communicator state."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    calls: list[tuple[str, object]] = []

    class Backend:
        def suspend(self, level: int = 1) -> None:
            calls.append(("backend.suspend", level))

        def resume(self, tags: list[str] | None = None) -> None:
            calls.append(("backend.resume", tuple(tags) if tags else None))

    worker = object.__new__(Worker)

    class Checkpoint:
        suspended = False

        def suspend(self):
            calls.append(("checkpoint.suspend", None))
            if checkpoint == "failure":
                raise RuntimeError("checkpoint failed")

        def resume(self):
            calls.append(("checkpoint.resume", None))

    worker._cuda_checkpoint = None if checkpoint == "disabled" else Checkpoint()
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

    if checkpoint == "failure":
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            worker.sleep(level=1)
        # Recovery is requested by the executor; failed sleep must stay paused.
        worker.wake_up()
    else:
        worker.sleep(level=1)
        worker.wake_up(tags=["weights"])

    expected = [
        ("backend.suspend", 1),
        ("comms.suspend", None),
        ("checkpoint.suspend", None),
        ("checkpoint.resume", None),
        ("backend.resume", None if checkpoint == "failure" else ("weights",)),
        ("comms.resume", None),
    ]
    if not enable_nccl_comm_suspend:
        expected = [c for c in expected if not c[0].startswith("comms.")]
    if checkpoint == "disabled":
        expected = [c for c in expected if not c[0].startswith("checkpoint.")]
    assert calls == expected


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported sleep-mode backend"):
        SleepModeBackendFactory.get_backend_class("does-not-exist")


@pytest.mark.parametrize(
    "initial,operation,calls_expected",
    [
        ("RUNNING", "suspend", ["lock", "checkpoint"]),
        ("LOCKED", "suspend", ["checkpoint"]),
        ("CHECKPOINTED", "suspend", []),
        ("RUNNING", "resume", []),
        ("LOCKED", "resume", ["unlock"]),
        ("CHECKPOINTED", "resume", ["restore", "unlock"]),
        ("FAILED", "resume", None),
        ("FAILED", "suspend", None),
    ],
)
def test_checkpoint_driver_state_recovery(
    monkeypatch, initial, operation, calls_expected
):
    """Retries use the driver state left by an interrupted helper."""
    from types import SimpleNamespace

    from cuda.bindings import driver

    from vllm.device_allocator.cuda_checkpoint import _main

    states = driver.CUprocessState
    state = getattr(states, f"CU_PROCESS_STATE_{initial}")
    calls = []
    success = driver.CUresult.CUDA_SUCCESS

    def transition(name, target):
        def call(*args):
            nonlocal state
            calls.append(name)
            state = getattr(states, f"CU_PROCESS_STATE_{target}")
            return (success,)

        return call

    monkeypatch.setattr(driver, "cuInit", lambda _: (success,))
    monkeypatch.setattr(driver, "cuDriverGetVersion", lambda: (success, 13020))
    monkeypatch.setattr(
        driver, "cuCheckpointProcessGetState", lambda _: (success, state)
    )
    monkeypatch.setattr(driver, "CUcheckpointLockArgs", SimpleNamespace)
    for action, target in (
        ("Lock", "LOCKED"),
        ("Checkpoint", "CHECKPOINTED"),
        ("Restore", "LOCKED"),
        ("Unlock", "RUNNING"),
    ):
        monkeypatch.setattr(
            driver, f"cuCheckpointProcess{action}", transition(action.lower(), target)
        )
    if calls_expected is None:
        with pytest.raises(RuntimeError, match="Cannot|not running"):
            _main(operation, 123)
        assert calls == []
    else:
        _main(operation, 123)
        assert calls == calls_expected
        expected = "CHECKPOINTED" if operation == "suspend" else "RUNNING"
        assert state == getattr(states, f"CU_PROCESS_STATE_{expected}")


def test_checkpoint_timeout_keeps_recovery_serialized(monkeypatch):
    """An unkillable helper must neither hang the caller nor race a retry."""
    import subprocess
    from unittest.mock import MagicMock

    from vllm.device_allocator.cuda_checkpoint import CudaCheckpoint

    checkpoint = object.__new__(CudaCheckpoint)
    checkpoint.suspended = False
    checkpoint._helper = None
    helper = MagicMock()
    helper.wait.side_effect = subprocess.TimeoutExpired("checkpoint", 60)
    helper.poll.return_value = None
    spawn = MagicMock(return_value=helper)
    monkeypatch.setattr(subprocess, "Popen", spawn)

    with pytest.raises(RuntimeError, match="timed out"):
        checkpoint.suspend()
    assert checkpoint.suspended
    assert [call.kwargs["timeout"] for call in helper.wait.call_args_list] == [60, 5]
    helper.kill.assert_called_once()
    with pytest.raises(RuntimeError, match="has not exited"):
        checkpoint.resume()
    assert checkpoint.suspended
    assert spawn.call_count == 1

    helper.poll.return_value = -9
    recovered = MagicMock(returncode=0)
    spawn.return_value = recovered
    checkpoint.resume()
    assert not checkpoint.suspended
    assert spawn.call_count == 2


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
