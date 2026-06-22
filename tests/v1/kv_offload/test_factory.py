# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for OffloadingSpecFactory.

These tests verify:
1. Pre-registration integrity — registered module paths can actually import
   and yield correct OffloadingSpec subclasses (CI sentinel against file moves).
2. End-to-end factory → spec construction with real configs.
3. Downstream collaboration — build_metric_definitions delegation.
4. Error paths — unregistered specs, missing config, duplicate registration.
"""

import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import OffloadingSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore OffloadingSpecFactory._registry between tests."""
    original = dict(OffloadingSpecFactory._registry)
    yield
    OffloadingSpecFactory._registry = original


def _make_vllm_config(
    spec_name: str | None = "CPUOffloadingSpec",
    cpu_bytes_to_use: int | None = None,
    store_threshold: int = 0,
    extra_config: dict | None = None,
):
    """Build a real VllmConfig with kv_transfer_config set for offloading."""
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )

    cfg = extra_config or {}
    if cpu_bytes_to_use is not None:
        cfg["cpu_bytes_to_use"] = cpu_bytes_to_use
    cfg["spec_name"] = spec_name
    if store_threshold > 0:
        cfg["store_threshold"] = store_threshold

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=cfg,
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_kv_cache_config():
    """Build a minimal KVCacheConfig with one KV cache tensor."""
    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(
        size=num_blocks * page_size, shared_by=["layer"], block_stride=0
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Pre-registration integrity (CI sentinel)
# ---------------------------------------------------------------------------


def test_pre_registered_specs_can_be_imported():
    """If someone moves cpu/spec.py but forgets to update factory.py, CI fails."""
    for name in OffloadingSpecFactory._registry:
        cls = OffloadingSpecFactory._registry[name]()
        assert issubclass(cls, OffloadingSpec)


def test_cpu_spec_registered():
    """CPUOffloadingSpec is registered and importable."""
    cls = OffloadingSpecFactory._registry["CPUOffloadingSpec"]()
    assert cls is CPUOffloadingSpec


def test_tiering_spec_registered():
    """TieringOffloadingSpec is registered and importable."""
    cls = OffloadingSpecFactory._registry["TieringOffloadingSpec"]()
    assert cls is TieringOffloadingSpec


# ---------------------------------------------------------------------------
# Normal path — get_spec_cls
# ---------------------------------------------------------------------------


def test_get_spec_cls_returns_registered_class():
    """Registered spec_name returns correct class."""
    config = _make_vllm_config(spec_name="CPUOffloadingSpec")
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    assert spec_cls is CPUOffloadingSpec


def test_get_spec_cls_default_to_cpu():
    """Default spec_name (absent from config) resolves to CPUOffloadingSpec."""
    config = _make_vllm_config(spec_name=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("spec_name", None)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# End-to-end — create_spec
# ---------------------------------------------------------------------------


def test_create_cpu_offloading_spec_end_to_end():
    """Full factory → spec construction with real VllmConfig/KVCacheConfig.

    Verifies:
    - cpu_bytes_to_use validation and num_blocks calculation
    - block_size % hash_block_size assertion
    - spec instance is CPUOffloadingSpec
    """
    config = _make_vllm_config(cpu_bytes_to_use=65536)
    kv_cache_config = _make_kv_cache_config()
    spec = OffloadingSpecFactory.create_spec(config, kv_cache_config)
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_blocks > 0


# ---------------------------------------------------------------------------
# Dynamic import via spec_module_path
# ---------------------------------------------------------------------------


def test_dynamic_load_via_spec_module_path():
    """External spec loaded via spec_module_path.

    This is how external projects (e.g., llm-d-kv-cache SharedStorageOffloadingSpec)
    integrate with vLLM without being pre-registered in the factory.
    The fallback path: registry miss → spec_module_path → importlib.import_module.
    """
    config = _make_vllm_config(spec_name="CPUOffloadingSpec")
    # Delete from registry to force the dynamic import path
    del OffloadingSpecFactory._registry["CPUOffloadingSpec"]
    # spec_name not in registry → falls through to spec_module_path
    config.kv_transfer_config.kv_connector_extra_config["spec_module_path"] = (
        "vllm.v1.kv_offload.cpu.spec"
    )
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unregistered_spec_without_module_path_raises():
    """spec_name not in registry + no spec_module_path → ValueError."""
    config = _make_vllm_config(spec_name="NonexistentSpec")
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.get_spec_cls(config)

    # create_spec should also fail (calls get_spec_cls internally)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.create_spec(config, kv_cache_config)


def test_cpu_spec_missing_cpu_bytes_to_use_raises():
    """CPUOffloadingSpec requires cpu_bytes_to_use → Exception."""
    config = _make_vllm_config(cpu_bytes_to_use=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("cpu_bytes_to_use", None)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        OffloadingSpecFactory.create_spec(config, kv_cache_config)


def test_duplicate_registration_raises():
    """register_spec with existing name → ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        OffloadingSpecFactory.register_spec(
            "CPUOffloadingSpec", "some.module", "SomeClass"
        )


# ---------------------------------------------------------------------------
# Downstream collaboration — build_metric_definitions
# ---------------------------------------------------------------------------


def test_build_metric_definitions_empty_below_threshold():
    """store_threshold < 2 → only base metric (no stores_skipped)."""
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_vllm_config(store_threshold=1)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    metrics = spec_cls.build_metric_definitions(
        config.kv_transfer_config.kv_connector_extra_config
    )
    assert CPUOffloadingMetrics.STORES_SKIPPED not in metrics


def test_build_metric_definitions_returns_counter_at_threshold():
    """store_threshold >= 2 → returns stores_skipped counter definition."""
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_vllm_config(store_threshold=2)
    spec_cls = OffloadingSpecFactory.get_spec_cls(config)
    metrics = spec_cls.build_metric_definitions(
        config.kv_transfer_config.kv_connector_extra_config
    )
    assert CPUOffloadingMetrics.STORES_SKIPPED in metrics
