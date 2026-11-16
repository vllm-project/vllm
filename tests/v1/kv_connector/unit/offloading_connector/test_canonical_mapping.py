# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import make_kv_cache_group
from vllm.distributed.kv_transfer.kv_connector.v1.layer_layout import LayerSlot
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    _layer_mapping,
    _opaque_fallback_mapping,
    _RankContext,
    _verify_tiling,
    canonical_page_size,
    canonical_row_layouts,
    derive_canonical_mappings,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheStage,
    KVQuantMode,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

NUM_BLOCKS = 3


def _ctx(rank, tp=1, dcp=1, pcp=1, interleave=1, total=None, heads=2):
    return _RankContext(
        tp_size=tp,
        dcp_size=dcp,
        pcp_size=pcp,
        interleave=interleave,
        total_kv_heads=heads * tp if total is None else total,
        rank=rank,
    )


def _full_spec(num_kv_heads: int = 2, **kwargs) -> AttentionSpec:
    # block_size=4, head_dim=64, int8
    return TritonAttentionBackend.customize_spec(
        AttentionSpec(
            block_size=4,
            num_kv_heads=num_kv_heads,
            head_size=64,
            dtype=torch.int8,
            **kwargs,
        )
    )


def _mla_spec(**kwargs) -> MLAAttentionSpec:
    # page = 4 * 64 = 256B, one 64B latent row per token
    return MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=64, dtype=torch.int8, **kwargs
    )


def _split_nhd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    )


def _split_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.num_kv_heads,
        spec.block_size,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 1, 3, 2, 4)


def _packed_nhd_cache(spec) -> torch.Tensor:
    """Logical (num_blocks, heads, block_size, 2 * head_size) over an NHD
    physical layout — the FlashAttention/FlashInfer/Triton/Flex form."""
    return torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        spec.num_kv_heads,
        2 * spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3)


def _packed_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_kv_heads,
        spec.block_size,
        2 * spec.head_size,
        dtype=torch.int8,
    )


CACHE_BUILDERS = {
    "split_nhd": _split_nhd_cache,
    "split_hnd": _split_hnd_cache,
    "packed_nhd": _packed_nhd_cache,
    "packed_hnd": _packed_hnd_cache,
}


def _try_mapping(spec, kv_cache, ctx) -> CanonicalPageMapping | None:
    return _layer_mapping(spec, kv_cache, NUM_BLOCKS, ctx)


def _mapping(spec, kv_cache, ctx) -> CanonicalPageMapping:
    mapping = _try_mapping(spec, kv_cache, ctx)
    assert mapping is not None
    return mapping


def _triples(runs: tuple[CopyRun, ...]) -> list[tuple[int, int, int]]:
    """Expand runs to explicit (local_offset, canonical_offset, size) copies."""
    out = []
    for run in runs:
        for i in range(run.num_fragments):
            out.append(
                (
                    run.local_offset + i * run.local_stride,
                    run.canonical_offset + i * run.canonical_stride,
                    run.fragment_size,
                )
            )
    return out


# ---------------------------------------------------------------------------
# TP-only placement (byte-compatible with the uniform interleave layout)
# ---------------------------------------------------------------------------


def test_split_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _split_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert mapping.canonical_page_size_bytes == 4 * 1024
    assert mapping.parallelism_agnostic
    k_dst = [256, 768, 1280, 1792]
    assert _triples(mapping.runs) == [
        (local, canonical, 128)
        for local, canonical in zip(
            [0, 128, 256, 384, 512, 640, 768, 896],
            k_dst + [2048 + o for o in k_dst],
        )
    ]
    # Heads are sharded, not replicated: this rank writes every block
    assert mapping.num_writers == 1


def test_packed_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert _triples(mapping.runs) == [
        (0, 512, 256),
        (256, 1536, 256),
        (512, 2560, 256),
        (768, 3584, 256),
    ]


def test_packed_hnd_placement_rank1_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_hnd_cache(spec), _ctx(rank=1, tp=4))
    # Two heads, each a contiguous canonical head region of 4 tokens x 128B
    assert _triples(mapping.runs) == [(0, 1024, 1024)]


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_single_rank_coalesces_to_one_run(form):
    spec = _full_spec()
    mapping = _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank=0))
    assert mapping.canonical_page_size_bytes == 1024
    assert _triples(mapping.runs) == [(0, 0, 1024)]


# ---------------------------------------------------------------------------
# Replication and writer election
# ---------------------------------------------------------------------------


def test_gqa_replicated_heads_rotate_writer():
    # total 2 KV heads on tp=4: replication factor 2, head shard = rank // 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, total=2)  # noqa: E731
    rank2 = _mapping(spec, cache, ctx(2))
    rank3 = _mapping(spec, cache, ctx(3))
    assert rank2.canonical_page_size_bytes == 2 * 512
    # K region: head shard 1 at 64B offsets within 128B token rows
    assert _triples(rank2.runs)[:4] == [
        (0, 64, 64),
        (64, 192, 64),
        (128, 320, 64),
        (192, 448, 64),
    ]
    # Same head shard, so identical bytes: the two take alternate blocks
    assert rank3.runs == rank2.runs
    assert (rank2.is_writer(0), rank3.is_writer(0)) == (True, False)
    assert (rank2.is_writer(1), rank3.is_writer(1)) == (False, True)
    _verify_tiling("gqa", [_mapping(spec, cache, ctx(r)) for r in range(4)])


def test_mla_replicas_rotate_writer():
    spec = _mla_spec()
    rank0 = _mapping(spec, None, _ctx(rank=0, tp=2))
    rank1 = _mapping(spec, None, _ctx(rank=1, tp=2))
    # Latent pages are stored once per block, not once per rank
    assert rank0.canonical_page_size_bytes == 256
    assert _triples(rank0.runs) == [(0, 0, 256)]
    assert rank1.runs == rank0.runs
    assert (rank0.is_writer(0), rank1.is_writer(0)) == (True, False)
    assert (rank0.is_writer(1), rank1.is_writer(1)) == (False, True)


# ---------------------------------------------------------------------------
# DCP / PCP token sharding
# ---------------------------------------------------------------------------


def test_dcp_interleaves_tokens_within_replicas():
    # tp=4, dcp=2, total 2 KV heads: head shard = rank // 2, cp rank = rank % 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, dcp=2, total=2)  # noqa: E731
    per_rank = [_mapping(spec, cache, ctx(rank)) for rank in range(4)]
    assert all(m is not None for m in per_rank)
    # 8 canonical tokens x 2 heads x 64B per region
    assert per_rank[0].canonical_page_size_bytes == 2048
    assert not per_rank[0].parallelism_agnostic
    # rank 2 = head shard 1, cp rank 0: K tokens 0,2,4,6 at head offset 64
    assert _triples(per_rank[2].runs)[:4] == [
        (0, 64, 64),
        (64, 320, 64),
        (128, 576, 64),
        (192, 832, 64),
    ]
    # every rank contributes (dcp == replication: no residual replicas)
    assert all(m.num_writers == 1 for m in per_rank)
    _verify_tiling("dcp", per_rank)


def test_mla_dcp_shards_latent_tokens():
    spec = _mla_spec()
    ctx = lambda rank: _ctx(rank, tp=2, dcp=2)  # noqa: E731
    rank0 = _mapping(spec, None, ctx(0))
    rank1 = _mapping(spec, None, ctx(1))
    assert rank0.canonical_page_size_bytes == 512
    assert _triples(rank0.runs) == [(o, 2 * o, 64) for o in (0, 64, 128, 192)]
    assert _triples(rank1.runs) == [(o, 2 * o + 64, 64) for o in (0, 64, 128, 192)]
    _verify_tiling("mla-dcp", [rank0, rank1])


def test_pcp_tokens_and_tp_heads_compose():
    # tp=2 x pcp=2: rank = pcp_rank * 2 + tp_rank; 4 workers tile the page
    spec = _full_spec(num_kv_heads=1)
    cache = _packed_nhd_cache(spec)
    per_rank = [
        _mapping(spec, cache, _ctx(rank, tp=2, pcp=2, total=2)) for rank in range(4)
    ]
    assert all(m is not None and m.runs for m in per_rank)
    _verify_tiling("pcp", per_rank)


def test_interleave_chunks_stay_contiguous():
    # interleave=2: chunks of 2 tokens alternate between the 2 cp ranks and
    # coalesce into one contiguous fragment per chunk
    spec = _mla_spec()
    mapping = _mapping(spec, None, _ctx(rank=0, tp=2, dcp=2, interleave=2))
    assert _triples(mapping.runs) == [(0, 0, 128), (128, 256, 128)]
    _verify_tiling(
        "interleave",
        [
            _mapping(spec, None, _ctx(rank, tp=2, dcp=2, interleave=2))
            for rank in range(2)
        ],
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_all_ranks_tile_canonical_page(form):
    spec = _full_spec()
    per_rank = [
        _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank, tp=4))
        for rank in range(4)
    ]
    _verify_tiling("layer", per_rank)


# ---------------------------------------------------------------------------
# Byte-level round trips
# ---------------------------------------------------------------------------


def _store_all(mappings, pages, size: int, block_id: int = 0) -> bytes:
    buf = bytearray(size)
    for mapping, page in zip(mappings, pages):
        if not mapping.is_writer(block_id):
            continue
        for local, canonical, n in _triples(mapping.runs):
            buf[canonical : canonical + n] = page[local : local + n]
    return bytes(buf)


def _load_one(mapping, canonical_bytes: bytes) -> bytes:
    page = bytearray(mapping.local_page_size_bytes)
    for local, canonical, n in _triples(mapping.runs):
        page[local : local + n] = canonical_bytes[canonical : canonical + n]
    return bytes(page)


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_cross_tp_store_load(form):
    """Bytes stored under one TP size are the bytes another TP size loads."""
    total_heads, canonical_size = 8, 4096
    reference = bytes((7 + 31 * i) % 256 for i in range(canonical_size))

    def mappings_at(tp: int):
        spec = _full_spec(num_kv_heads=total_heads // tp)
        cache = CACHE_BUILDERS[form](spec)
        return [
            _mapping(spec, cache, _ctx(rank, tp=tp, total=total_heads))
            for rank in range(tp)
        ]

    for tp in (4, 2, 1):
        mappings = mappings_at(tp)
        assert all(m is not None for m in mappings)
        pages = [_load_one(m, reference) for m in mappings]
        assert _store_all(mappings, pages, canonical_size) == reference


def _reference_page(form, tokens, heads, head_size) -> bytes:
    """Serialize logical K/V, token, head, and dimension coordinates."""
    if form.endswith("nhd"):
        token_heads = [(token, head) for token in tokens for head in heads]
    else:
        token_heads = [(token, head) for head in heads for token in tokens]
    if form.startswith("split"):
        cells = ((kv, token, head) for kv in range(2) for token, head in token_heads)
    else:
        cells = ((kv, token, head) for token, head in token_heads for kv in range(2))
    return bytes(
        (32 * token + 8 * head + 4 * kv + dim) % 256
        for kv, token, head in cells
        for dim in range(head_size)
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
@pytest.mark.parametrize("interleave", [1, 2])
def test_tp2_pcp2_store_load_matches_reference(form, interleave):
    """TP head shards and PCP token shards preserve independent reference bytes."""
    spec = _full_spec()
    cache = CACHE_BUILDERS[form](spec)
    mappings = [
        _mapping(
            spec,
            cache,
            _ctx(rank, tp=2, dcp=1, pcp=2, interleave=interleave, total=4),
        )
        for rank in range(4)
    ]
    pcp_tokens = {
        1: ((0, 2, 4, 6), (1, 3, 5, 7)),
        2: ((0, 1, 4, 5), (2, 3, 6, 7)),
    }[interleave]
    # Rank order: PCP0/TP0, PCP0/TP1, PCP1/TP0, PCP1/TP1.
    rank_coordinates = (
        (pcp_tokens[0], (0, 1)),
        (pcp_tokens[0], (2, 3)),
        (pcp_tokens[1], (0, 1)),
        (pcp_tokens[1], (2, 3)),
    )
    pages = [
        _reference_page(form, tokens, heads, spec.head_size)
        for tokens, heads in rank_coordinates
    ]
    reference = _reference_page(form, range(8), range(4), spec.head_size)

    _verify_tiling("tp2-pcp2", mappings)
    assert _store_all(mappings, pages, len(reference)) == reference
    for rank, (mapping, page) in enumerate(zip(mappings, pages)):
        assert mapping.canonical_page_size_bytes == len(reference)
        assert mapping.local_page_size_bytes == len(page)
        assert not mapping.parallelism_agnostic
        assert _load_one(mapping, reference) == page, f"rank {rank}"


def test_cp_round_trip():
    # tp=4 / dcp=2 / 2 KV heads: 4 workers jointly hold one canonical page
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    mappings = [
        _mapping(spec, cache, _ctx(rank, tp=4, dcp=2, total=2)) for rank in range(4)
    ]
    reference = bytes((3 + 17 * i) % 256 for i in range(2048))
    pages = [_load_one(m, reference) for m in mappings]
    assert _store_all(mappings, pages, 2048) == reference


def test_replica_rotation_round_trip():
    """Whichever replica a block elects reproduces the same canonical page."""
    spec = _mla_spec()
    mappings = [_mapping(spec, None, _ctx(rank, tp=2)) for rank in range(2)]
    reference = bytes((5 + 11 * i) % 256 for i in range(256))
    pages = [_load_one(m, reference) for m in mappings]
    for block_id in (0, 1):
        assert _store_all(mappings, pages, 256, block_id) == reference


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_fail_closed_cases():
    spec = _full_spec()
    nhd = _split_nhd_cache(spec)
    # Spec heads inconsistent with total heads / tp
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, total=2)) is None
    # tp not divisible by total KV heads
    one_head = _full_spec(num_kv_heads=1)
    assert (
        _try_mapping(one_head, _split_nhd_cache(one_head), _ctx(0, tp=3, total=2))
        is None
    )
    # DCP wider than the KV replication factor (tokens would shard across
    # ranks holding different heads)
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, dcp=2)) is None
    # Interleave must divide the block size
    assert _try_mapping(spec, nhd, _ctx(0, tp=2, dcp=2, interleave=3, total=2)) is None
    # Per-token-head scales are packed with the data
    quant_spec = _full_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant_spec, _split_nhd_cache(quant_spec), _ctx(0, tp=4)) is None
    # Compressed MLA slots are not 1:1 with tokens
    assert (
        _try_mapping(_mla_spec(tokens_per_state=2), None, _ctx(0, tp=2, dcp=2)) is None
    )
    # Unrecognized physical layouts
    swapped = torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        2,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3, 4)
    assert _try_mapping(spec, swapped, _ctx(0, tp=4)) is None
    # Non-attention specs
    assert _try_mapping(KVCacheSpec(block_size=4), None, _ctx(0, tp=4, total=8)) is None


def test_opaque_fallback_places_page_whole():
    mapping = _opaque_fallback_mapping(1024, 4, 2)
    assert mapping.canonical_page_size_bytes == 4096
    assert not mapping.parallelism_agnostic
    assert _triples(mapping.runs) == [(0, 2048, 1024)]
    assert mapping.num_writers == 1
    _verify_tiling("opaque", [_opaque_fallback_mapping(1024, 4, r) for r in range(4)])


# ---------------------------------------------------------------------------
# derive_canonical_mappings end to end
# ---------------------------------------------------------------------------


def _vllm_config(tp=1, dcp=1, pcp=1, pp=1, interleave=1, total_kv_heads=2):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.parallel_config.pipeline_parallel_size = pp
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = 0
    config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return config


def _kv_cache_config(groups):
    config = MagicMock()
    config.kv_cache_groups = groups
    config.num_blocks = NUM_BLOCKS
    return config


def test_derive_mixed_model_with_dcp():
    attn_spec = _full_spec(num_kv_heads=1)
    mla_spec = _mla_spec()
    quant_spec = _full_spec(
        num_kv_heads=1, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD
    )
    kv_cache_config = _kv_cache_config(
        [
            make_kv_cache_group(["attn"], attn_spec),
            make_kv_cache_group(["mla"], mla_spec),
            make_kv_cache_group(["quant"], quant_spec),
        ]
    )
    kv_caches = {
        "attn": _split_nhd_cache(attn_spec),
        "quant": _split_nhd_cache(quant_spec),
    }
    mappings = derive_canonical_mappings(
        _vllm_config(tp=4, dcp=2, total_kv_heads=2), kv_cache_config, kv_caches
    )
    assert set(mappings) == {"attn", "mla", "quant"}
    assert not mappings["attn"].parallelism_agnostic
    assert mappings["attn"].runs
    # Uncertifiable layers degrade to an opaque page, never disappear
    assert not mappings["quant"].parallelism_agnostic
    assert (
        mappings["quant"].canonical_page_size_bytes
        == 4 * quant_spec.unpadded_page_size_bytes
    )


def test_derive_refuses_foreign_worker_groups():
    # The workers are not the TP x PCP grid of every pipeline stage
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config([make_kv_cache_group(["attn"], attn_spec)])
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    vllm_config = _vllm_config(tp=2, pp=2, total_kv_heads=4)
    vllm_config.parallel_config.world_size = 8
    assert derive_canonical_mappings(vllm_config, kv_cache_config, kv_caches) == {}


def test_derive_pipeline_stage_rank_maps_like_its_tp_rank():
    # Workers are ordered by PP rank, then TP rank: worker 3 of TP2 x PP2 is
    # TP rank 1 of the second stage.
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config([make_kv_cache_group(["attn"], attn_spec)])
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    pp_config = _vllm_config(tp=2, pp=2, total_kv_heads=4)
    pp_config.parallel_config.rank = 3
    tp_config = _vllm_config(tp=2, total_kv_heads=4)
    tp_config.parallel_config.rank = 1

    mappings = derive_canonical_mappings(pp_config, kv_cache_config, kv_caches)
    assert mappings == derive_canonical_mappings(tp_config, kv_cache_config, kv_caches)
    tp_config.parallel_config.rank = 0
    rank0 = derive_canonical_mappings(tp_config, kv_cache_config, kv_caches)
    assert mappings["attn"].runs != rank0["attn"].runs


# ---------------------------------------------------------------------------
# Canonical chunk rows
# ---------------------------------------------------------------------------


def _stage_kv_cache_config(partition: list[int], stage: int) -> KVCacheConfig:
    """The KV cache config of one pipeline stage of a model with an attention
    group and an MLA group, whose layers `partition` splits across stages."""
    bounds = [sum(partition[:i]) for i in range(len(partition) + 1)]
    stages = tuple(
        KVCacheStage(
            kv_cache_groups=(
                make_kv_cache_group(
                    [f"attn.{i}" for i in range(start, end)], _full_spec()
                ),
                make_kv_cache_group(
                    [f"mla.{i}" for i in range(start, end)], _mla_spec()
                ),
            ),
        )
        for start, end in zip(bounds, bounds[1:])
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=list(stages[stage].kv_cache_groups),
    )
    kv_cache_config.pipeline_stages = stages
    return kv_cache_config


@pytest.mark.parametrize("partition", [[4], [2, 2], [3, 1], [1, 1, 1, 1]])
def test_row_layouts_do_not_depend_on_pipeline_partition(partition):
    """Every stage lays out a chunk row from the layers of all stages, in
    model order, so the slots match those of a single-stage engine."""
    single_stage = canonical_row_layouts(
        _vllm_config(tp=2, total_kv_heads=4),
        _stage_kv_cache_config([4], 0),
        (0, 1),
        blocks_per_chunk=2,
    )
    vllm_config = _vllm_config(tp=2, pp=len(partition), total_kv_heads=4)
    for stage in range(len(partition)):
        layouts = canonical_row_layouts(
            vllm_config,
            _stage_kv_cache_config(partition, stage),
            (0, 1),
            blocks_per_chunk=2,
        )
        assert layouts == single_stage

    # The two TP head shards of an attention page, one replicated MLA page
    attn_page = canonical_page_size(vllm_config, _full_spec())
    assert attn_page == 2 * _full_spec().real_page_size_bytes
    mla_page = canonical_page_size(vllm_config, _mla_spec())
    assert mla_page == _mla_spec().real_page_size_bytes
    assert single_stage[0].slots["attn.3"] == LayerSlot(6 * attn_page, 2 * attn_page)
    assert single_stage[0].size == 8 * attn_page
    # A row holds one group's layers, so every group's slots start at 0
    assert single_stage[1].slots["mla.0"] == LayerSlot(0, 2 * mla_page)
    assert single_stage[1].size == 8 * mla_page


def test_uncertifiable_layer_gets_its_opaque_page_slot():
    spec = _full_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    vllm_config = _vllm_config(tp=4, total_kv_heads=8)
    assert canonical_page_size(vllm_config, spec) == 4 * spec.unpadded_page_size_bytes


def test_unrecognized_layout_of_certified_layer_fills_its_canonical_page():
    """Kernel blocks split the page, so its byte layout is not recognized,
    but the spec's KV head shards are: every rank's page goes whole to its
    head shard's position, and the page keeps the size of its row slot."""
    spec = _full_spec(num_kv_heads=1)
    kv_cache_config = _kv_cache_config([make_kv_cache_group(["attn"], spec)])
    kernel_blocks = torch.zeros(
        2 * NUM_BLOCKS, 2, spec.block_size // 2, 1, 64, dtype=torch.int8
    )
    vllm_config = _vllm_config(tp=4, total_kv_heads=2)
    per_rank = []
    for rank in range(4):
        vllm_config.parallel_config.rank = rank
        mappings = derive_canonical_mappings(
            vllm_config, kv_cache_config, {"attn": kernel_blocks}
        )
        per_rank.append(mappings["attn"])

    page = spec.real_page_size_bytes
    assert canonical_page_size(vllm_config, spec) == 2 * page
    assert all(m.canonical_page_size_bytes == 2 * page for m in per_rank)
    assert [_triples(m.runs) for m in per_rank] == [
        [(0, 0, page)],
        [(0, 0, page)],
        [(0, page, page)],
        [(0, page, page)],
    ]
    # The two ranks holding a head take turns writing it
    assert [(m.num_writers, m.writer_index) for m in per_rank] == [
        (2, 0),
        (2, 1),
        (2, 0),
        (2, 1),
    ]
    assert not any(m.parallelism_agnostic for m in per_rank)
