# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config import ParallelConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Suspend/resume: Worker.sleep refuses the configurations #46877's
# communicator checkpoint hooks cannot cover (they own the lifecycle itself).


class _StubSleepBackend:
    """Flag defaults mirror the SleepModeBackend ABC (both False)."""

    def __init__(self, events, comms=False, graphs=False):
        self.events = events
        self.preserves_communicators = lambda: comms
        self.preserves_graphs_with_communicators = lambda: graphs

    def suspend(self, level=1):
        self.events.append("suspend")

    def resume(self, tags=None):
        self.events.append("resume")


def _sleep_worker(backend, parallel_config=None):
    worker = object.__new__(Worker)
    worker._sleep_mode_backend = backend
    worker._sleep_saved_buffers = {}
    worker._sleep_rebuild_draft_metadata_buffers = False
    worker.parallel_config = parallel_config or ParallelConfig()
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(named_buffers=lambda: [("b", torch.zeros(1))]),
        post_kv_cache_wake_up=lambda: None,
    )
    return worker


@pytest.fixture
def sleep_events(monkeypatch):
    accel = gpu_worker_module.torch.accelerator
    monkeypatch.setattr(accel, "synchronize", lambda: None)
    monkeypatch.setattr(accel, "get_memory_info", lambda: (0, 0))
    return []


@pytest.mark.parametrize(
    "parallel_config",
    [ParallelConfig(tensor_parallel_size=2), ParallelConfig(data_parallel_size=2)],
)
def test_sleep_refuses_multi_process_without_communicator_support(
    sleep_events, parallel_config
):
    # Level 2 with a named buffer: a raise after the bookkeeping would save it.
    worker = _sleep_worker(_StubSleepBackend(sleep_events), parallel_config)
    with pytest.raises(NotImplementedError, match="world_size_across_dp"):
        worker.sleep(level=2)
    assert sleep_events == []
    assert worker._sleep_saved_buffers == {}


def test_sleep_allows_single_process_without_communicator_support(sleep_events):
    # world_size 1 has no device communicator to lose, so nothing to refuse.
    _sleep_worker(_StubSleepBackend(sleep_events)).sleep(level=1)
    assert sleep_events == ["suspend"]


def test_sleep_allows_multi_process_when_communicators_are_preserved(sleep_events):
    worker = _sleep_worker(
        _StubSleepBackend(sleep_events, comms=True),
        ParallelConfig(tensor_parallel_size=2),
    )
    worker.sleep(level=1)
    assert sleep_events == ["suspend"]


def test_full_checkpoint_lifecycle_calls_each_hook_once(sleep_events, monkeypatch):
    """#46877 owns the checkpoint hooks; sleep/wake must not call them too.

    Regression: an earlier revision called checkpoint_prepare_distributed_state
    from sleep() while Worker.checkpoint_prepare already did, so the documented
    orchestration produced prepare, prepare, suspend.
    """
    for name, label in [
        ("checkpoint_prepare_distributed_state", "prepare"),
        ("checkpoint_restore_distributed_state", "restore"),
    ]:
        monkeypatch.setattr(
            gpu_worker_module, name, lambda label=label: sleep_events.append(label)
        )
    worker = _sleep_worker(_StubSleepBackend(sleep_events))
    worker.checkpoint_prepare()
    worker.sleep(level=1)
    worker.wake_up()
    worker.checkpoint_restore()
    assert sleep_events == ["prepare", "suspend", "resume", "restore"]
