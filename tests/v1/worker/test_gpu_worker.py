# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config.parallel import ParallelConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    PersistentWorkspaceProfilingSupport,
)
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)
from vllm.v1.worker.utils import requires_persistent_attention_workspace_profiling
from vllm.v1.worker.workspace import get_num_workspace_ubatches


@pytest.mark.parametrize(
    ("parallel_config", "expected"),
    [
        pytest.param(ParallelConfig(), 1, id="single-ubatch-default"),
        pytest.param(ParallelConfig(ubatch_size=3), 3, id="manual-ubatch-size"),
        pytest.param(ParallelConfig(enable_dbo=True), 2, id="dbo"),
    ],
)
def test_num_workspace_ubatches_covers_all_configurations(parallel_config, expected):
    assert get_num_workspace_ubatches(parallel_config) == expected


@pytest.mark.parametrize("profile_persistent_workspace", [False, True])
def test_initialize_kv_cache_finalizes_persistent_workspace(
    monkeypatch, profile_persistent_workspace
):
    events = []
    worker = object.__new__(Worker)
    worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
    worker.vllm_config = object()
    worker.model_config = SimpleNamespace(enable_return_routed_experts=False)
    worker._maybe_get_memory_pool_context = lambda **kwargs: nullcontext()
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda config: events.append("initialize_kv_cache"),
        reserve_persistent_attention_workspace=lambda: events.append(
            "reserve_workspace"
        ),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=8,
        needs_kv_cache_zeroing=False,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_initialized",
        lambda *args, **kwargs: events.append("initialize_connector"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "requires_persistent_attention_workspace_profiling",
        lambda config: profile_persistent_workspace,
    )

    worker.initialize_from_config(kv_cache_config)

    assert worker.cache_config.num_gpu_blocks == 8
    expected_events = [
        "initialize_connector",
        "initialize_kv_cache",
    ]
    if profile_persistent_workspace:
        expected_events.append("reserve_workspace")
    assert events == expected_events


@pytest.mark.parametrize(
    ("builder_support", "speculative", "elastic_ep", "expected"),
    [
        pytest.param(
            [PersistentWorkspaceProfilingSupport.REQUIRED],
            False,
            False,
            True,
            id="single-required",
        ),
        pytest.param(
            [
                PersistentWorkspaceProfilingSupport.REQUIRED,
                PersistentWorkspaceProfilingSupport.REQUIRED,
            ],
            False,
            False,
            True,
            id="all-required",
        ),
        pytest.param(
            [
                PersistentWorkspaceProfilingSupport.REQUIRED,
                PersistentWorkspaceProfilingSupport.NEUTRAL,
            ],
            False,
            False,
            True,
            id="required-with-neutral",
        ),
        pytest.param(
            [
                PersistentWorkspaceProfilingSupport.REQUIRED,
                PersistentWorkspaceProfilingSupport.UNSUPPORTED,
            ],
            False,
            False,
            False,
            id="unsupported-vetoes-required",
        ),
        pytest.param(
            [PersistentWorkspaceProfilingSupport.NEUTRAL],
            False,
            False,
            False,
            id="neutral-only",
        ),
        pytest.param([object()], False, False, False, id="unknown-fails-closed"),
        pytest.param([], False, False, False, id="no-builders"),
        pytest.param(
            [PersistentWorkspaceProfilingSupport.REQUIRED],
            True,
            False,
            False,
            id="speculative-fallback",
        ),
        pytest.param(
            [PersistentWorkspaceProfilingSupport.REQUIRED],
            False,
            True,
            False,
            id="elastic-ep-fallback",
        ),
    ],
)
def test_persistent_workspace_profiling_supports_neutral_builders(
    monkeypatch, builder_support, speculative, elastic_ep, expected
):
    class Builder:
        def __init__(self, support):
            self.support = support

        def get_persistent_workspace_memory_profiling_support(self, config, spec):
            return self.support

    class Backend:
        def __init__(self, support):
            self.builder = Builder(support)

        def get_builder_cls(self):
            return self.builder

    class Layer:
        def __init__(self, support):
            self.backend = Backend(support)

        def get_kv_cache_spec(self, config):
            return object()

        def get_attn_backend(self):
            return self.backend

    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        parallel_config=SimpleNamespace(enable_elastic_ep=elastic_ep),
    )
    layers = {
        f"layer-{index}": Layer(support)
        for index, support in enumerate(builder_support)
    }
    monkeypatch.setattr(
        "vllm.v1.worker.utils.get_layers_from_vllm_config",
        lambda config, layer_type: layers,
    )

    assert requires_persistent_attention_workspace_profiling(config) is expected


def test_gdn_builder_is_neutral_for_persistent_workspace_profiling():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

    assert (
        GDNAttentionMetadataBuilder.get_persistent_workspace_memory_profiling_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is PersistentWorkspaceProfilingSupport.NEUTRAL
    )


def test_unknown_builder_fails_closed_for_persistent_workspace_profiling():
    assert (
        AttentionMetadataBuilder.get_persistent_workspace_memory_profiling_support(
            SimpleNamespace(), SimpleNamespace()
        )
        is PersistentWorkspaceProfilingSupport.UNSUPPORTED
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


def test_startup_plan_rejects_stale_schema(plan_env, monkeypatch):
    worker = _plan_worker()
    fingerprint = startup_plan.compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    maybe_save_startup_plan(worker, 50 * GiB_bytes)

    monkeypatch.setattr(
        startup_plan,
        "PLAN_SCHEMA_VERSION",
        startup_plan.PLAN_SCHEMA_VERSION + 1,
    )

    assert startup_plan._load_plan(fingerprint) is None


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
