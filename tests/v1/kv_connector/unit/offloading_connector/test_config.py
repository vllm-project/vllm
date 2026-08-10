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
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
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
    kv_tensor = KVCacheTensor(
        size=num_blocks * 8,
        shared_by=["layer"],
        block_stride=0,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
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


def _full_attention_spec(block_size: int = 16, **kwargs: Any) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
        **kwargs,
    )


def _mla_spec(
    block_size: int = 16,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        **kwargs,
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
        KVCacheTensor(
            size=spec.page_size_bytes * num_blocks,
            shared_by=[layer_name],
        )
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
            KVCacheGroupSpec(["full_layer"], _full_attention_spec(block_size=12)),
            KVCacheGroupSpec(["mla_layer"], _mla_spec()),
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


def _parallelism_agnostic(
    kv_cache_groups: list[KVCacheGroupSpec],
    *,
    canonical: bool = False,
    v2: bool = False,
) -> bool:
    config = _make_vllm_config(
        extra_config={"canonical_layout": True} if canonical else None
    )
    config.use_v2_model_runner = v2
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


def test_rejects_partially_packed_tensor_layout():
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.kv_cache_tensors[0].block_stride = 16

    with pytest.raises(AssertionError):
        build_offloading_config(_make_vllm_config(), kv_cache_config)


def test_zero_blocks_skips_tensor_layout_validation():
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.num_blocks = 0
    kv_cache_config.kv_cache_tensors[0].block_stride = 16

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


def test_preserves_data_parallel_index():
    config = _make_vllm_config()
    config.parallel_config.data_parallel_index = 2

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert offloading_config.parallel.data_parallel_index == 2


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
    assert _parallelism_agnostic(mla_groups, canonical=True)

    # Mamba hybrids stay out: their state layers can only derive opaque
    # (exact-topology) mappings
    mamba_groups = list(_make_mamba_hybrid_kv_cache_config().kv_cache_groups)
    assert not _parallelism_agnostic(mamba_groups, canonical=True)


def test_canonical_layout_certifies_attention_hybrids():
    """Hybrid attention models (full + sliding-window groups, e.g. gpt-oss)
    certify group by group under the canonical layout; every layer's bytes
    are head-shard fragments regardless of its window."""
    hybrid_groups = [
        KVCacheGroupSpec(["l0"], _full_attention_spec()),
        KVCacheGroupSpec(
            ["l1"],
            SlidingWindowSpec(
                block_size=16,
                num_kv_heads=4,
                head_size=128,
                dtype=torch.float32,
                sliding_window=128,
            ),
        ),
    ]
    assert not _parallelism_agnostic(hybrid_groups)
    assert _parallelism_agnostic(hybrid_groups, canonical=True)

    # DSv4-style sliding-window MLA is not a head-sharded layout; stays out
    swa_mla = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        sliding_window=128,
    )
    assert not _parallelism_agnostic(
        [KVCacheGroupSpec(["l0"], swa_mla)], canonical=True
    )


def test_canonical_layout_certifies_per_token_head_quant():
    """Per-token-head packed rows carry inline scales, so GQA groups certify
    under the canonical layout; the direct layout and the MLA latent path
    keep excluding it."""
    pth = KVQuantMode.FP8_PER_TOKEN_HEAD
    pth_groups = [KVCacheGroupSpec(["l0"], _full_attention_spec(kv_quant_mode=pth))]
    assert not _parallelism_agnostic(pth_groups)
    assert _parallelism_agnostic(pth_groups, canonical=True)

    mla_pth_groups = [
        KVCacheGroupSpec(["l0"], _mla_spec(head_size=576, kv_quant_mode=pth))
    ]
    assert not _parallelism_agnostic(mla_pth_groups, canonical=True)


def test_canonical_layout_certifies_uniform_type_wrapper():
    """GLM-5.2/DSv3.2-style groups wrap same-type specs with different page
    sizes (MLA plus its DSA indexer cache) in UniformTypeKVCacheSpecs; the
    gate must look through the wrapper like the mapping derivation does."""

    def uniform_groups(indexer_spec) -> list[KVCacheGroupSpec]:
        return [
            KVCacheGroupSpec(
                ["mla_layer", "indexer_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        "mla_layer": _mla_spec(head_size=576),
                        "indexer_layer": indexer_spec,
                    },
                ),
            )
        ]

    mla_wrapped = uniform_groups(_mla_spec(head_size=128))
    assert not _parallelism_agnostic(mla_wrapped)
    assert _parallelism_agnostic(mla_wrapped, canonical=True)

    # one uncertifiable inner spec poisons the wrapper
    swa_mla_wrapped = uniform_groups(
        SlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.float32,
            sliding_window=128,
        )
    )
    assert not _parallelism_agnostic(swa_mla_wrapped, canonical=True)


def test_canonical_layout_certifies_v2_model_runner():
    """Canonical bytes are certified per layer against live tensor strides at
    registration, so the static gate must not depend on the model-runner
    version — the v2 runner is the case the canonical layout exists for."""
    groups = [KVCacheGroupSpec(["l0"], _full_attention_spec())]
    assert not _parallelism_agnostic(groups, v2=True)
    assert _parallelism_agnostic(groups, canonical=True, v2=True)


def test_canonical_format_resolved_at_config_build():
    """The canonical format id needs the vLLM config context to resolve the
    KV cache layout; consumers like the scheduler-side FileMapper run outside
    that context, so build_offloading_config must resolve it eagerly."""
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_attention_spec())],
    )

    offloading_config = build_offloading_config(
        _make_vllm_config(extra_config={"canonical_layout": True}), kv_cache_config
    )
    assert offloading_config.canonical_format is not None
    assert offloading_config.canonical_format.startswith("v1-")

    no_canonical = build_offloading_config(_make_vllm_config(), kv_cache_config)
    assert no_canonical.canonical_format is None


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
