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

from typing import Any, cast
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
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
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


def _make_mla_kv_cache_config(
    layer_names: list[str] | None = None,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    num_blocks: int = 4,
) -> KVCacheConfig:
    if layer_names is None:
        layer_names = ["layer0", "layer1"]
    spec = _mla_spec(head_size=head_size, dtype=dtype)
    kv_cache_tensors = [
        KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[layer_name])
        for layer_name in layer_names
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
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


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tiering_spec_replicated_sizing_removes_world_factor(world_size: int):
    kv_cache_config = _make_mla_kv_cache_config()
    worker_kv_bytes_per_block = kv_cache_config.kv_cache_groups[
        0
    ].kv_cache_spec.page_size_bytes * len(
        kv_cache_config.kv_cache_groups[0].layer_names
    )
    cpu_bytes_to_use = worker_kv_bytes_per_block * 8
    config = _make_layout_vllm_config(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=cpu_bytes_to_use,
        tensor_parallel_size=world_size,
    )
    config.model_config.use_mla = True

    spec = _create_spec(config, kv_cache_config)

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block
    assert spec.num_blocks == cpu_bytes_to_use // worker_kv_bytes_per_block


def test_tiering_spec_create_worker_uses_single_slot_for_replicated_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    kv_cache_config = _make_mla_kv_cache_config()
    worker_kv_bytes_per_block = kv_cache_config.kv_cache_groups[
        0
    ].kv_cache_spec.page_size_bytes * len(
        kv_cache_config.kv_cache_groups[0].layer_names
    )
    config = _make_layout_vllm_config(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        tensor_parallel_size=4,
    )
    config.model_config.use_mla = True
    spec = _create_spec(config, kv_cache_config)
    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.replicated_layout is True

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    # Device index 5 makes a regression to the device-index fold yield
    # rank 1 (5 % 4) instead of coincidentally passing with rank 0.
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator, "current_device_index", lambda: 5
    )

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    assert region_calls[0]["rank"] == 0
    assert region_calls[0]["replicated_layout"] is True
    assert region_calls[0]["kv_bytes_per_block"] == worker_kv_bytes_per_block
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


def test_tiering_spec_create_worker_folds_device_index_for_sharded_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    config = _make_layout_vllm_config(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=65536,
        tensor_parallel_size=4,
    )
    spec = _create_spec(config, _make_kv_cache_config())
    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.replicated_layout is False

    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator,
        "current_device_index",
        lambda: 5,
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["rank"] == 1
    assert region_calls[0]["replicated_layout"] is False


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_config_preserves_per_rank_sizing(world_size: int):
    kv_cache_config = _make_mla_kv_cache_config()
    worker_kv_bytes_per_block = kv_cache_config.kv_cache_groups[
        0
    ].kv_cache_spec.page_size_bytes * len(
        kv_cache_config.kv_cache_groups[0].layer_names
    )
    cpu_bytes_to_use = worker_kv_bytes_per_block * world_size * 2
    config = _make_layout_vllm_config(
        cpu_bytes_to_use=cpu_bytes_to_use,
        tensor_parallel_size=world_size,
    )
    config.model_config.use_mla = True

    spec = _create_spec(config, kv_cache_config)

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is False
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block * world_size
    assert spec.num_blocks == cpu_bytes_to_use // (
        worker_kv_bytes_per_block * world_size
    )


def test_offloading_spec_has_replicated_layout_default():
    config = _make_layout_vllm_config()
    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    spec = SingleArgExternalOffloadingSpec(offloading_config)

    assert spec.replicated_layout is False


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


def _mla_spec(
    block_size: int = 16,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=dtype
    )


def _parallelism_agnostic(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    config = _make_layout_vllm_config()
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
    )
    offloading_config = build_offloading_config(config, kv_cache_config)
    return offloading_config.parallel.is_parallelism_agnostic


def _replicated_layout(
    kv_cache_config: KVCacheConfig,
    *,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    use_mla: bool = True,
    use_v2_model_runner: bool = False,
    distributed_executor_backend: Any = "mp",
    nnodes: int = 1,
    world_size: int | None = None,
) -> bool:
    config = _make_layout_vllm_config(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
    )
    config.model_config.use_mla = use_mla
    config.use_v2_model_runner = use_v2_model_runner
    config.parallel_config.distributed_executor_backend = distributed_executor_backend
    config.parallel_config.nnodes = nnodes
    if world_size is not None:
        config.parallel_config.world_size = world_size
    return build_offloading_config(config, kv_cache_config).replicated_layout


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_replicated_layout_enabled_for_pure_mla_tp_mp_single_node(
    world_size: int,
):
    assert _replicated_layout(
        _make_mla_kv_cache_config(), tensor_parallel_size=world_size
    )


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["layer"],
                    )
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer"],
                        SlidingWindowMLASpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=512,
                            dtype=torch.float32,
                            sliding_window=128,
                        ),
                    )
                ],
            ),
            "sliding-window-mla",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["layer"],
                    )
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer"],
                        HiddenStateCacheSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=512,
                            dtype=torch.float32,
                        ),
                    )
                ],
            ),
            "hidden-state",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["layer0"],
                    ),
                    KVCacheTensor(
                        size=_mla_spec(head_size=256).page_size_bytes * 4,
                        shared_by=["layer1"],
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer0", "layer1"],
                        UniformTypeKVCacheSpecs(
                            block_size=16,
                            kv_cache_specs={
                                "layer0": _mla_spec(),
                                "layer1": _mla_spec(head_size=256),
                            },
                        ),
                    )
                ],
            ),
            "uniform-wrapper",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["mla"],
                    ),
                    KVCacheTensor(
                        size=_full_attention_spec().page_size_bytes * 4,
                        shared_by=["full"],
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["mla"], _mla_spec()),
                    KVCacheGroupSpec(["full"], _full_attention_spec()),
                ],
            ),
            "mla-full-hybrid",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["mla"],
                    ),
                    KVCacheTensor(size=64 * 4, shared_by=["mamba"]),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["mla"], _mla_spec()),
                    KVCacheGroupSpec(
                        ["mamba"],
                        MambaSpec(
                            block_size=16,
                            shapes=((16, 1),),
                            dtypes=(torch.float32,),
                        ),
                    ),
                ],
            ),
            "mla-mamba-hybrid",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["layer0"],
                    ),
                    KVCacheTensor(
                        size=_mla_spec().page_size_bytes * 4,
                        shared_by=["layer1"],
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["layer0"], _mla_spec()),
                    KVCacheGroupSpec(["layer1"], _mla_spec()),
                ],
            ),
            "multi-group-mla",
        ),
    ],
    ids=[
        "sliding-window-mla",
        "hidden-state",
        "uniform-wrapper",
        "mla-full-hybrid",
        "mla-mamba-hybrid",
        "multi-group-mla",
    ],
)
def test_replicated_layout_excludes_unproven_cache_shapes(
    kv_cache_config: KVCacheConfig, case: str
):
    assert not _replicated_layout(kv_cache_config), case


def test_replicated_layout_rejects_bare_mla_with_mixed_page_accounting():
    num_blocks = 4
    main_spec = _mla_spec(head_size=512)
    indexer_spec = _mla_spec(head_size=128, dtype=torch.uint8)
    main_layers = [f"main_{i}" for i in range(61)]
    indexer_layers = [f"indexer_{i}" for i in range(61)]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=main_spec.page_size_bytes * len(main_layers) * num_blocks,
                shared_by=main_layers,
            ),
            KVCacheTensor(
                size=indexer_spec.page_size_bytes * len(indexer_layers) * num_blocks,
                shared_by=indexer_layers,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(main_layers + indexer_layers, main_spec),
        ],
    )

    assert not _replicated_layout(kv_cache_config)


@pytest.mark.parametrize(
    ("kwargs", "case"),
    [
        ({"tensor_parallel_size": 1}, "tp1"),
        ({"use_mla": False}, "use-mla-false"),
        ({"pipeline_parallel_size": 2, "world_size": 4}, "pp2"),
        ({"prefill_context_parallel_size": 2, "world_size": 4}, "pcp2"),
        ({"decode_context_parallel_size": 2}, "dcp2"),
        ({"world_size": 8}, "world-ne-tp"),
        ({"distributed_executor_backend": "ray"}, "ray"),
        ({"distributed_executor_backend": "uni"}, "uni"),
        ({"distributed_executor_backend": type("DummyExecutor", (), {})}, "class"),
        ({"nnodes": 2}, "multi-node"),
        ({"use_v2_model_runner": True}, "v2-runner"),
    ],
    ids=[
        "tp1",
        "use-mla-false",
        "pp2",
        "pcp2",
        "dcp2",
        "world-ne-tp",
        "ray",
        "uni",
        "class",
        "multi-node",
        "v2-runner",
    ],
)
def test_replicated_layout_parallel_gate(kwargs: dict[str, Any], case: str):
    assert not _replicated_layout(_make_mla_kv_cache_config(), **kwargs), case


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
