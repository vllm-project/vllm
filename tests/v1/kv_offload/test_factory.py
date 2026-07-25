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

from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
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


def _get_extra_config(config: VllmConfig) -> dict:
    assert config.kv_transfer_config is not None
    return config.kv_transfer_config.kv_connector_extra_config


def _create_spec(config: VllmConfig, kv_cache_config: KVCacheConfig) -> OffloadingSpec:
    return OffloadingSpecFactory.create_spec(
        build_offloading_config(config, kv_cache_config)
    )


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


def _make_layout_vllm_config(
    spec_name: str = "CPUOffloadingSpec",
    cpu_bytes_to_use: int | None = None,
    extra_config: dict | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> VllmConfig:
    config = MagicMock()
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.cache_dtype = torch.float16
    config.model_config.model = "test-model"
    world_size = (
        tensor_parallel_size * pipeline_parallel_size * prefill_context_parallel_size
    )
    with patch.object(current_platform, "device_count", return_value=world_size):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
        )
    config.kv_events_config = None
    config.use_v2_model_runner = False

    connector_extra_config = dict(extra_config or {})
    connector_extra_config["spec_name"] = spec_name
    if cpu_bytes_to_use is not None:
        connector_extra_config["cpu_bytes_to_use"] = cpu_bytes_to_use
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=connector_extra_config,
    )
    return cast(VllmConfig, config)


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


def _make_sizing_kv_cache_config(packed: bool) -> KVCacheConfig:
    num_blocks = 4
    if packed:
        kv_cache_tensors = [
            KVCacheTensor(
                size=64,
                shared_by=[layer_name],
                block_stride=16,
            )
            for layer_name in ("layer0", "layer1")
        ]
    else:
        kv_cache_tensors = [
            KVCacheTensor(size=40, shared_by=["layer0"]),
            KVCacheTensor(size=24, shared_by=["layer1"]),
        ]

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0", "layer1"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _make_hybrid_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=40, shared_by=["full_layer"]),
            KVCacheTensor(size=24, shared_by=["mla_layer"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=12,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mla_layer"],
                MLAAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=576,
                    dtype=torch.float32,
                ),
            ),
        ],
    )


class SingleArgExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


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
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    assert spec_cls is CPUOffloadingSpec


def test_get_spec_cls_default_to_cpu():
    """Default spec_name (absent from config) resolves to CPUOffloadingSpec."""
    config = _make_vllm_config(spec_name=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("spec_name", None)
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# End-to-end — create_spec
# ---------------------------------------------------------------------------


def test_create_cpu_offloading_spec_end_to_end():
    """Full factory → spec construction with real VllmConfig/KVCacheConfig.

    Verifies:
    - cpu_bytes_to_use validation and num_blocks calculation
    - block_size % tokens_per_hash assertion
    - spec instance is CPUOffloadingSpec
    """
    config = _make_vllm_config(cpu_bytes_to_use=65536)
    kv_cache_config = _make_kv_cache_config()
    spec = _create_spec(config, kv_cache_config)
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_blocks > 0


@pytest.mark.parametrize("packed", [False, True])
def test_cpu_spec_sizing_preserves_tensor_layout(packed: bool):
    cpu_bytes_to_use = 1920
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=cpu_bytes_to_use,
        extra_config={"block_size": 32},
        tensor_parallel_size=3,
        pipeline_parallel_size=2,
    )

    spec = _create_spec(config, _make_sizing_kv_cache_config(packed))

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == 192
    assert spec.num_blocks == cpu_bytes_to_use // 192


def test_cpu_spec_rejects_partially_packed_tensor_layout():
    config = _make_layout_vllm_config(cpu_bytes_to_use=65536)
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.kv_cache_tensors[0].block_stride = 16

    with pytest.raises(AssertionError):
        _create_spec(config, kv_cache_config)


def test_cpu_spec_zero_blocks_skips_tensor_layout_validation():
    config = _make_layout_vllm_config(cpu_bytes_to_use=65536)
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.num_blocks = 0
    kv_cache_config.kv_cache_tensors[0].block_stride = 16

    spec = _create_spec(config, kv_cache_config)

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 0
    assert spec.kv_bytes_per_chunk == 0
    assert spec.num_blocks == 0


def test_tiering_spec_aligns_row_size():
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    cpu_bytes_to_use = alignment * 3
    config = _make_layout_vllm_config(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=cpu_bytes_to_use,
        extra_config={"block_size": 32},
        tensor_parallel_size=3,
        pipeline_parallel_size=2,
    )

    spec = _create_spec(config, _make_sizing_kv_cache_config(packed=False))

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == alignment
    assert spec.num_blocks == cpu_bytes_to_use // alignment


def test_offloading_spec_kv_sharding_ignores_prefill_context_parallel():
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=65536,
        extra_config={"block_size": 64},
        prefill_context_parallel_size=2,
    )

    spec = _create_spec(config, _make_kv_cache_config())

    assert spec.tokens_per_block == (16,)
    assert spec.tokens_per_hash == 16
    assert spec.blocks_per_chunk == 4


def test_offloading_config_preserves_data_parallel_index():
    config = _make_layout_vllm_config()
    config.parallel_config.data_parallel_index = 2

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert offloading_config.parallel.data_parallel_index == 2


def test_offloading_spec_resolves_heterogeneous_hybrid_block_sizes():
    config = _make_layout_vllm_config(cpu_bytes_to_use=65536)
    config.cache_config.block_size = 4

    spec = _create_spec(config, _make_hybrid_kv_cache_config())

    assert spec.tokens_per_block == (12, 16)
    assert spec.tokens_per_hash == 4
    assert spec.blocks_per_chunk == 1


def _full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=4, head_size=128, dtype=torch.float32
    )


def _parallelism_agnostic(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    config = _make_layout_vllm_config()
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    offloading_config = build_offloading_config(config, kv_cache_config)
    return offloading_config.parallel.is_parallelism_agnostic


def test_parallelism_agnostic_for_single_full_attention_group():
    assert _parallelism_agnostic([KVCacheGroupSpec(["l0"], _full_attention_spec())])


@pytest.mark.parametrize(
    "kv_cache_groups",
    [
        # MLA latent KV is replicated per rank, never head-sharded.
        [
            KVCacheGroupSpec(
                ["l0"],
                MLAAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=576, dtype=torch.float32
                ),
            )
        ],
        # Sliding window is not full attention.
        [
            KVCacheGroupSpec(
                ["l0"],
                SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=128,
                    dtype=torch.float32,
                    sliding_window=128,
                ),
            )
        ],
        # Hybrid model: more than one KV cache group.
        [
            KVCacheGroupSpec(["l0"], _full_attention_spec()),
            KVCacheGroupSpec(["l1"], _full_attention_spec()),
        ],
    ],
)
def test_parallelism_agnostic_excluded(kv_cache_groups: list[KVCacheGroupSpec]):
    assert not _parallelism_agnostic(kv_cache_groups)


def test_parallelism_agnostic_disabled_on_v2_model_runner():
    config = _make_layout_vllm_config()
    config.use_v2_model_runner = True
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_attention_spec())],
    )
    offloading_config = build_offloading_config(config, kv_cache_config)
    assert not offloading_config.parallel.is_parallelism_agnostic


def test_create_dynamic_spec_receives_translated_config():
    config = _make_layout_vllm_config(
        spec_name="SingleArgExternalOffloadingSpec",
        extra_config={
            "spec_module_path": "tests.v1.kv_offload.test_factory",
        },
    )
    kv_cache_config = _make_kv_cache_config()
    offloading_config = build_offloading_config(config, kv_cache_config)

    spec = OffloadingSpecFactory.create_spec(offloading_config)

    assert isinstance(spec, SingleArgExternalOffloadingSpec)
    assert spec.config is offloading_config


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
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    assert spec_cls is CPUOffloadingSpec


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unregistered_spec_without_module_path_raises():
    """spec_name not in registry + no spec_module_path → ValueError."""
    config = _make_vllm_config(spec_name="NonexistentSpec")
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))

    # create_spec should also fail (calls get_spec_cls internally)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(ValueError, match="Unsupported spec type"):
        _create_spec(config, kv_cache_config)


def test_cpu_spec_missing_cpu_bytes_to_use_raises():
    """CPUOffloadingSpec requires cpu_bytes_to_use → Exception."""
    config = _make_vllm_config(cpu_bytes_to_use=None)
    config.kv_transfer_config.kv_connector_extra_config.pop("cpu_bytes_to_use", None)
    kv_cache_config = _make_kv_cache_config()
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        _create_spec(config, kv_cache_config)


def test_duplicate_registration_raises():
    """register_spec with existing name → ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        OffloadingSpecFactory.register_spec(
            "CPUOffloadingSpec", "some.module", "SomeClass"
        )


# ---------------------------------------------------------------------------
# Downstream collaboration — build_metric_definitions
# ---------------------------------------------------------------------------


def test_build_metric_definitions_below_threshold():
    """store_threshold < 2 keeps stores_skipped disabled."""
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_vllm_config(store_threshold=1)
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    metrics = spec_cls.build_metric_definitions(
        config.kv_transfer_config.kv_connector_extra_config
    )
    assert CPUOffloadingMetrics.STORES_SKIPPED not in metrics
    assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE in metrics


def test_build_metric_definitions_allocation_size_histogram():
    """CPU allocation size is always reported as a histogram."""
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_vllm_config(store_threshold=0)
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    metrics = spec_cls.build_metric_definitions(
        config.kv_transfer_config.kv_connector_extra_config
    )
    metadata = metrics[CPUOffloadingMetrics.CPU_ALLOCATION_SIZE]
    assert isinstance(metadata, OffloadingHistogramMetadata)
    assert metadata.buckets == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
    )


def test_build_metric_definitions_returns_counter_at_threshold():
    """store_threshold >= 2 → returns stores_skipped counter definition."""
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    config = _make_vllm_config(store_threshold=2)
    spec_cls = OffloadingSpecFactory.get_spec_cls(_get_extra_config(config))
    metrics = spec_cls.build_metric_definitions(
        config.kv_transfer_config.kv_connector_extra_config
    )
    assert CPUOffloadingMetrics.STORES_SKIPPED in metrics


def test_offloading_spec_accepts_blocks_per_chunk_for_heterogeneous_groups():
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=65536,
        extra_config={"blocks_per_chunk": 2},
    )

    spec = _create_spec(config, _make_hybrid_kv_cache_config())

    assert spec.tokens_per_block == (12, 16)
    assert spec.blocks_per_chunk == 2


def test_block_size_and_blocks_per_chunk_are_mutually_exclusive():
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=65536,
        extra_config={
            "block_size": 64,
            "blocks_per_chunk": 2,
        },
    )

    with pytest.raises(ValueError, match="Specify only one"):
        _create_spec(config, _make_kv_cache_config())


def test_blocks_per_chunk_must_be_positive():
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=65536,
        extra_config={
            "blocks_per_chunk": 0,
        },
    )

    with pytest.raises(ValueError, match="greater than 0"):
        _create_spec(config, _make_kv_cache_config())
