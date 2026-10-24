# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    _layer_mapping,
    _opaque_fallback_mapping,
    _RankContext,
    _verify_tiling,
    derive_canonical_mappings,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVQuantMode,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

NUM_BLOCKS = 3


@pytest.mark.parametrize(
    "change",
    [
        {"tp_replicated": True},
        {"shapes": ((3, 64), (8, 4, 2))},
        {"shapes": ((64, 3),)},
        {"dtypes": (torch.bfloat16,)},
        {"shapes": ((0, 3), (8, 4, 2))},
        {"shapes": ((65, 3), (8, 4, 2))},
        {"dtypes": (torch.float32, torch.float32)},
        {"dtypes": (torch.bfloat16, torch.float16)},
        {"num_prefill_checkpoint_blocks": 1},
        {"num_speculative_blocks": 1},
        {"page_size_padded": 4},
        {"mamba_cache_mode": "none"},
    ],
)
def test_gdn_rejects_unsupported_state_contract(change, monkeypatch):
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.recurrent import (
        derive_gdn_geometry,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    get_conv_state_layout.cache_clear()
    spec = MambaSpec(
        block_size=16,
        shapes=((64, 3), (8, 4, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    assert derive_gdn_geometry(replace(spec, **change), 1) is None
    get_conv_state_layout.cache_clear()


@pytest.mark.parametrize("producer_tp", [1, 2, 4])
@pytest.mark.parametrize("consumer_tp", [1, 2, 4])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("padding", [0, 32])
def test_gdn_cross_tp_bytes(producer_tp, consumer_tp, conv_dtype, padding, monkeypatch):
    """An independent projection/head oracle catches rank-major GDN corruption."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.recurrent import (
        derive_gdn_geometry,
        gdn_mapping,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    get_conv_state_layout.cache_clear()
    # Global Q/K widths 16, V width 32; temporal has eight (V=4,K=2) heads.
    sizes = (96, 96, 192, 256)
    components = [
        (
            torch.arange(size).repeat(NUM_BLOCKS, 1)
            + torch.arange(NUM_BLOCKS)[:, None] * 17
            + component * 41
        )
        .remainder(251)
        .to(torch.int8)
        for component, size in enumerate(sizes)
    ]
    expected = torch.cat(components, dim=1)

    def shard(tp, rank):
        return torch.cat(
            [
                value[:, rank * (size // tp) : (rank + 1) * (size // tp)]
                for size, value in zip(sizes, components)
            ],
            dim=1,
        )

    def page_and_mapping(tp, rank):
        local_bytes = sum(sizes) // tp
        physical_page = local_bytes + padding
        spec = MambaSpec(
            block_size=16,
            shapes=((64 // tp, 3), (8 // tp, 4, 2)),
            dtypes=(conv_dtype, torch.float32),
            page_size_padded=physical_page,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
        )
        backing = torch.full(
            (32 + NUM_BLOCKS * physical_page + 32,), -91, dtype=torch.int8
        )
        raw = backing.as_strided(
            (NUM_BLOCKS, 1, 1, local_bytes), (physical_page, 7, 5, 1), storage_offset=32
        )
        geometry = derive_gdn_geometry(spec, tp)
        assert geometry is not None
        mapping = gdn_mapping(spec, raw, NUM_BLOCKS, tp, rank)
        assert mapping.parallelism_agnostic
        return backing, raw.squeeze((1, 2)), mapping, physical_page

    canonical = torch.full_like(expected, -91)
    mappings = []
    for rank in reversed(range(producer_tp)):
        _, page, mapping, _ = page_and_mapping(producer_tp, rank)
        page.copy_(shard(producer_tp, rank))
        mappings.append(mapping)
        for lo, co, size in _triples(mapping.runs):
            canonical[:, co : co + size] = page[:, lo : lo + size]
    _verify_tiling("gdn", mappings)
    assert torch.equal(canonical, expected)
    for rank in range(consumer_tp):
        backing, page, mapping, physical_page = page_and_mapping(consumer_tp, rank)
        for lo, co, size in _triples(mapping.runs):
            page[:, lo : lo + size] = canonical[:, co : co + size]
        assert torch.equal(page, shard(consumer_tp, rank))
        assert torch.all(backing[:32] == -91)
        assert torch.all(backing[-32:] == -91)
        physical = backing[32:-32].reshape(NUM_BLOCKS, physical_page)
        assert torch.all(physical[:, mapping.local_page_size_bytes :] == -91)
    get_conv_state_layout.cache_clear()


def _ctx(rank, tp=1, dcp=1, pcp=1, interleave=1, total=None, heads=2):
    return _RankContext(
        tp_size=tp,
        dcp_size=dcp,
        pcp_size=pcp,
        interleave=interleave,
        total_kv_heads=heads * tp if total is None else total,
        rank=rank,
    )


def _full_spec(num_kv_heads: int = 2, **kwargs) -> FullAttentionSpec:
    # block_size=4, head_dim=64, int8
    return TritonAttentionBackend.customize_spec(
        FullAttentionSpec(
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
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=mla_spec),
            KVCacheGroupSpec(layer_names=["quant"], kv_cache_spec=quant_spec),
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
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec)]
    )
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    assert (
        derive_canonical_mappings(
            _vllm_config(tp=2, pp=2, total_kv_heads=4), kv_cache_config, kv_caches
        )
        == {}
    )
