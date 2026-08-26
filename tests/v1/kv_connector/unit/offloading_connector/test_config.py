# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for translating vLLM cache metadata to native offloading config."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.config import KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    SchedulerOffloadConfig,
    is_store_reachable_swa_chunk,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def _make_vllm_config(
    *,
    extra_config: dict[str, Any] | None = None,
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
    config.model_config.use_mla = False
    # _full_attention_spec's heads at tp=1: the parallelism-agnostic gate
    # requires the head shard to cover the model's KV heads exactly
    config.model_config.get_total_num_kv_heads.return_value = 4
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
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra_config or {}),
    )
    return cast(VllmConfig, config)


def _make_kv_cache_config() -> KVCacheConfig:
    num_blocks = 16
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_tensor = KVCacheTensor(
        size=spec.page_size_bytes * num_blocks,
        layers=["layer"],
        layer_stride=spec.page_size_bytes * num_blocks,
        block_stride=spec.page_size_bytes,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _make_sizing_kv_cache_config(packed: bool) -> KVCacheConfig:
    """One 16 byte-per-block allocation, described two ways.

    Packed: both layers are one dense run. Unpacked: the same bytes as two runs, the
    second starting after the first layer's region. Either way the connector accounts
    for 16 KV bytes per block.
    """
    num_blocks = 4
    page = 8
    size = 2 * page * num_blocks
    if packed:
        kv_cache_tensors = [
            KVCacheTensor(
                size=size,
                layers=["layer0", "layer1"],
                layer_stride=page * num_blocks,
                block_stride=page,
            )
        ]
    else:
        kv_cache_tensors = [
            KVCacheTensor(
                size=size,
                layers=[layer],
                layer_stride=page * num_blocks,
                block_stride=page,
                offset=i * page * num_blocks,
            )
            for i, layer in enumerate(("layer0", "layer1"))
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


def _full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
    )


def _mla_spec(
    block_size: int = 16,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
    )


_MAMBA_SPEC = MambaSpec(
    block_size=16,
    shapes=((16, 1),),
    dtypes=(torch.float32,),
)
# Page sizes of the specs the replicated-layout cases below are built from.
_MLA_PAGE = _mla_spec().page_size_bytes
_HALF_MLA_PAGE = _mla_spec(head_size=256).page_size_bytes
_FULL_PAGE = _full_attention_spec().page_size_bytes
_MAMBA_PAGE = _MAMBA_SPEC.page_size_bytes


def _make_mla_kv_cache_config(
    layer_names: list[str] | None = None,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    num_blocks: int = 4,
) -> KVCacheConfig:
    if layer_names is None:
        layer_names = ["layer0", "layer1"]
    spec = _mla_spec(head_size=head_size, dtype=dtype)
    layer_stride = spec.page_size_bytes * num_blocks
    kv_cache_tensors = [
        KVCacheTensor(
            size=layer_stride * len(layer_names),
            layers=layer_names,
            layer_stride=layer_stride,
            block_stride=spec.page_size_bytes,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )


def _make_hybrid_kv_cache_config() -> KVCacheConfig:
    num_blocks = 4
    full_spec = _full_attention_spec(block_size=12)
    mla_spec = _mla_spec()
    # Mixed page sizes across overlaying groups: a block is a window of the largest
    # group's packing, so the layer dim sits inside the block dim.
    window = max(full_spec.page_size_bytes, mla_spec.page_size_bytes)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=window * num_blocks,
                layers=["full_layer"],
                layer_stride=full_spec.page_size_bytes,
                block_stride=window,
            ),
            KVCacheTensor(
                size=window * num_blocks,
                layers=["mla_layer"],
                layer_stride=mla_spec.page_size_bytes,
                block_stride=window,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_layer"], full_spec),
            KVCacheGroupSpec(["mla_layer"], mla_spec),
        ],
    )


def _make_mamba_hybrid_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_layer"], _full_attention_spec()),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=16,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


def _parallelism_agnostic(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    config = _make_vllm_config()
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )
    return build_offloading_config(
        config, kv_cache_config
    ).parallel.is_parallelism_agnostic


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
    config = _make_vllm_config(
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


@pytest.mark.parametrize("packed", [False, True])
def test_worker_kv_bytes_preserves_tensor_layout(packed: bool):
    config = _make_vllm_config(
        extra_config={"block_size": 32},
        tensor_parallel_size=3,
        pipeline_parallel_size=2,
    )

    offloading_config = build_offloading_config(
        config, _make_sizing_kv_cache_config(packed)
    )

    assert offloading_config.worker_kv_bytes_per_block == 16
    assert offloading_config.parallel.world_size == 6
    assert offloading_config.cache.blocks_per_chunk == 2


def test_hisparse_offloads_only_indexer_group():
    source = KVCacheGroupSpec(
        ["source"],
        _full_attention_spec(),
        block_pool_id=None,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
    )
    indexer = KVCacheGroupSpec(
        ["indexer"],
        _full_attention_spec(),
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[source, indexer],
        hisparse_host_num_blocks=4,
    )
    config = _make_vllm_config()

    offloading_config = build_offloading_config(config, kv_cache_config)
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config), config, kv_cache_config
    )

    assert [
        (group.group_id, group.layer_names) for group in offloading_config.groups
    ] == [(1, ("indexer",))]
    assert [group.group_idx for group in scheduler_config.kv_group_configs] == [1]


def test_zero_blocks_skips_tensor_layout_validation():
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.num_blocks = 0

    offloading_config = build_offloading_config(_make_vllm_config(), kv_cache_config)

    assert offloading_config.worker_kv_bytes_per_block == 0


def test_prefill_context_parallelism_does_not_scale_group_blocks():
    config = _make_vllm_config(
        extra_config={"block_size": 64},
        prefill_context_parallel_size=2,
    )

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (16,)
    assert offloading_config.cache.tokens_per_hash == 16
    assert offloading_config.cache.blocks_per_chunk == 4


def test_dcp_scales_attention_but_not_mamba_group_blocks():
    config = _make_vllm_config(tensor_parallel_size=2, decode_context_parallel_size=2)
    config.speculative_config = None

    offloading_config = build_offloading_config(
        config, _make_mamba_hybrid_kv_cache_config()
    )

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        32,
        16,
    )
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config),
        config,
        _make_mamba_hybrid_kv_cache_config(),
    )
    mamba_group = scheduler_config.kv_group_configs[1]
    assert mamba_group.alignment_chunk_count == 2
    assert [
        chunk_idx
        for chunk_idx in range(4)
        if is_store_reachable_swa_chunk(
            chunk_idx,
            4,
            mamba_group.alignment_chunk_count,
            mamba_group.sliding_window_size_in_chunks,
            mamba_group.is_eagle_group,
        )
    ] == [1, 3]


def test_preserves_data_parallel_config():
    config = _make_vllm_config()
    config.parallel_config.data_parallel_index = 2
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.data_parallel_rank_local = 1

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert offloading_config.parallel.data_parallel_index == 2
    assert offloading_config.parallel.data_parallel_size == 4
    assert offloading_config.parallel.data_parallel_rank_local == 1


def test_resolves_heterogeneous_hybrid_block_sizes():
    config = _make_vllm_config()
    config.cache_config.block_size = 4

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.tokens_per_hash == 4
    assert offloading_config.cache.blocks_per_chunk == 1


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replicated_layout_enabled_for_pure_mla_tp_mp_single_node(
    world_size: int,
    use_v2_model_runner: bool,
):
    assert _replicated_layout(
        _make_mla_kv_cache_config(),
        tensor_parallel_size=world_size,
        use_v2_model_runner=use_v2_model_runner,
    )


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["layer"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
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
                        size=_MLA_PAGE * 4,
                        layers=["layer"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
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
            # One group, two page sizes: layer1's run starts past layer0's region.
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=(_MLA_PAGE + _HALF_MLA_PAGE) * 4,
                        layers=["layer0"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    ),
                    KVCacheTensor(
                        size=(_MLA_PAGE + _HALF_MLA_PAGE) * 4,
                        layers=["layer1"],
                        layer_stride=_HALF_MLA_PAGE * 4,
                        block_stride=_HALF_MLA_PAGE,
                        offset=_MLA_PAGE * 4,
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
            # Overlaid groups with different page sizes: a block is a window of
            # the largest group's packing.
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_FULL_PAGE * 4,
                        layers=["mla"],
                        layer_stride=_MLA_PAGE,
                        block_stride=_FULL_PAGE,
                    ),
                    KVCacheTensor(
                        size=_FULL_PAGE * 4,
                        layers=["full"],
                        layer_stride=_FULL_PAGE,
                        block_stride=_FULL_PAGE,
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
                        size=_MLA_PAGE * 4,
                        layers=["mla"],
                        layer_stride=_MLA_PAGE,
                        block_stride=_MLA_PAGE,
                    ),
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["mamba"],
                        layer_stride=_MAMBA_PAGE,
                        block_stride=_MLA_PAGE,
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["mla"], _mla_spec()),
                    KVCacheGroupSpec(["mamba"], _MAMBA_SPEC),
                ],
            ),
            "mla-mamba-hybrid",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=[layer],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    )
                    for layer in ("layer0", "layer1")
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
    kv_cache_config: KVCacheConfig,
    case: str,
):
    assert not _replicated_layout(kv_cache_config), case


def test_replicated_layout_rejects_bare_mla_with_mixed_page_accounting():
    num_blocks = 4
    main_spec = _mla_spec(head_size=512)
    indexer_spec = _mla_spec(head_size=128, dtype=torch.uint8)
    main_layers = [f"main_{i}" for i in range(61)]
    indexer_layers = [f"indexer_{i}" for i in range(61)]
    # A DSA-style group: the main pages and the smaller indexer pages are packed one
    # after the other, so a block holds more than 61 MLA pages.
    main_bytes = main_spec.page_size_bytes * len(main_layers)
    indexer_bytes = indexer_spec.page_size_bytes * len(indexer_layers)
    size = (main_bytes + indexer_bytes) * num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=size,
                layers=main_layers,
                layer_stride=main_spec.page_size_bytes * num_blocks,
                block_stride=main_spec.page_size_bytes,
            ),
            KVCacheTensor(
                size=size,
                layers=indexer_layers,
                layer_stride=indexer_spec.page_size_bytes * num_blocks,
                block_stride=indexer_spec.page_size_bytes,
                offset=main_bytes * num_blocks,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(main_layers + indexer_layers, main_spec)],
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
    ],
)
def test_replicated_layout_parallel_gate(kwargs: dict[str, Any], case: str):
    assert not _replicated_layout(_make_mla_kv_cache_config(), **kwargs), case


def test_parallelism_agnostic_for_single_full_attention_group():
    assert _parallelism_agnostic([KVCacheGroupSpec(["l0"], _full_attention_spec())])


@pytest.mark.parametrize(
    "kv_cache_groups",
    [
        [KVCacheGroupSpec(["l0"], _mla_spec(head_size=576))],
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
        [
            KVCacheGroupSpec(["l0"], _full_attention_spec()),
            KVCacheGroupSpec(["l1"], _full_attention_spec()),
        ],
    ],
)
def test_parallelism_agnostic_excluded(kv_cache_groups: list[KVCacheGroupSpec]):
    assert not _parallelism_agnostic(kv_cache_groups)


def test_canonical_layout_widens_parallelism_agnostic_to_mla():
    """The canonical layout dedups the TP-replicated MLA latent into one
    portable copy, so the gate admits MLA — but only when canonical_layout
    is requested."""
    mla_groups = [KVCacheGroupSpec(["l0"], _mla_spec(head_size=576))]
    assert not _parallelism_agnostic(mla_groups)

    config = _make_vllm_config(extra_config={"canonical_layout": True})
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=mla_groups,
    )
    offloading_config = build_offloading_config(config, kv_cache_config)
    assert offloading_config.parallel.is_parallelism_agnostic
    assert offloading_config.canonical_layout

    # hybrid groupings stay out: their non-full-attention layers can only
    # derive opaque (exact-topology) mappings
    hybrid_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["l0"], _full_attention_spec()),
            KVCacheGroupSpec(["l1"], _full_attention_spec()),
        ],
    )
    assert not build_offloading_config(
        config, hybrid_config
    ).parallel.is_parallelism_agnostic


def test_canonical_layout_certifies_v2_model_runner():
    """Canonical bytes are certified per layer against live tensor strides at
    registration, so the static gate must not depend on the model-runner
    version — the v2 runner is the case the canonical layout exists for."""
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_attention_spec())],
    )

    config = _make_vllm_config()
    config.use_v2_model_runner = True
    assert not build_offloading_config(
        config, kv_cache_config
    ).parallel.is_parallelism_agnostic

    config = _make_vllm_config(extra_config={"canonical_layout": True})
    config.use_v2_model_runner = True
    assert build_offloading_config(
        config, kv_cache_config
    ).parallel.is_parallelism_agnostic


def test_parallelism_agnostic_disabled_on_v2_model_runner():
    config = _make_vllm_config()
    config.use_v2_model_runner = True
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_attention_spec())],
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert not offloading_config.parallel.is_parallelism_agnostic


def test_accepts_blocks_per_chunk_for_heterogeneous_groups():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 2})

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.blocks_per_chunk == 2


def test_block_size_and_blocks_per_chunk_are_mutually_exclusive():
    config = _make_vllm_config(extra_config={"block_size": 64, "blocks_per_chunk": 2})

    with pytest.raises(ValueError, match="Specify only one"):
        build_offloading_config(config, _make_kv_cache_config())


def test_blocks_per_chunk_must_be_positive():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 0})

    with pytest.raises(ValueError, match="greater than 0"):
        build_offloading_config(config, _make_kv_cache_config())
