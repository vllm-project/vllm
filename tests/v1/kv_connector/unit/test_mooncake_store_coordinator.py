# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import lcm

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    chunk_hashes_for_block_size,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)


def _make_coord(groups, hash_block_size, use_eagle=False, retention_interval=None):
    """Construct a coordinator using the natural LCM of group block sizes as
    the scheduler block size — mirrors ``resolve_kv_cache_block_sizes`` for
    the test fixtures."""
    block_sizes = [g.kv_cache_spec.block_size for g in groups]
    scheduler_block_size = lcm(*block_sizes)
    return MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
        retention_interval=retention_interval,
    )


# ----- ExternalCachedBlockPool -----


def test_external_cached_block_pool_tautological_returns_present_for_any_hash():
    cmap = ExternalCachedBlockPool()
    h = BlockHash(b"\xaa" * 4)
    res = cmap.get_cached_block(h, [0, 1])
    assert res is not None
    assert len(res) == 2
    assert res[0] is not cmap.null_block
    assert res[1] is not cmap.null_block


def test_external_cached_block_pool_hit_all_groups():
    h = BlockHash(b"\x11\x22\x33\x44")
    cmap = ExternalCachedBlockPool({(0, bytes(h)), (1, bytes(h))})
    res = cmap.get_cached_block(h, [0, 1])
    assert res is not None
    assert len(res) == 2
    assert res[0] is not cmap.null_block
    assert res[1] is not cmap.null_block


def test_external_cached_block_pool_miss_one_group():
    h = BlockHash(b"\x11\x22\x33\x44")
    cmap = ExternalCachedBlockPool({(0, bytes(h))})
    assert cmap.get_cached_block(h, [0, 1]) is None


def test_external_cached_block_pool_unknown_hash():
    h_known = BlockHash(b"\x01" * 4)
    h_unknown = BlockHash(b"\x02" * 4)
    cmap = ExternalCachedBlockPool({(0, bytes(h_known))})
    assert cmap.get_cached_block(h_unknown, [0]) is None


# ----- Helpers -----


def _full(block_size=16, sliding_window=None):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=sliding_window,
    )


def _swa(block_size=16, sliding_window=32):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=sliding_window,
    )


def _hashes(n: int) -> list[BlockHash]:
    return [BlockHash(bytes([i + 1]) * 4) for i in range(n)]


# ----- Single-group coordinator -----


def test_coordinator_single_full_attention_all_hits():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 64
    assert masks[0] == [True, True, True, True]


def test_coordinator_single_full_attention_partial_prefix():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(hs[0])), (0, bytes(hs[1]))})
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 32
    assert masks[0] == [True, True]


def test_coordinator_single_full_attention_no_hits():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool(set())
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 0
    assert masks[0] == []


def test_coordinator_single_swa_tautological_pool_masks_pre_window():
    """SWA tautological-pool: hit_length spans full prefix, mask is
    tail-window only."""
    groups = [KVCacheGroupSpec(["L0"], _swa(block_size=16, sliding_window=32))]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)  # 4 chunks * 16 tokens
    cmap = ExternalCachedBlockPool()
    masks, hit = coord.find_longest_cache_hit(hs, max_length=64, cached_block_pool=cmap)
    assert hit == 64
    # ceil((sw-1)/block_size) = ceil(31/16) = 2 tail blocks.
    assert masks[0][-2:] == [True, True]
    assert all(not m for m in masks[0][:-2])


# ----- Hybrid coordinator (single-group worker, multi-group coordinator) -----


def test_coordinator_hybrid_full_plus_swa_all_hit():
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(g, bytes(h)) for g in (0, 1) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64


def test_coordinator_hybrid_hole_in_full_clips_both():
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    exists = {(0, bytes(hs[0])), (0, bytes(hs[2])), (0, bytes(hs[3]))}
    exists |= {(1, bytes(h)) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 16


def test_coordinator_group_block_size_double_hash():
    """Group block_size=32 over hash_block_size=16 hashes: adjacent
    hashes merge before pool lookup."""
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _full(32)),
    ]
    coord = _make_coord(groups, hash_block_size=16)
    hs = _hashes(4)
    big_hashes = list(chunk_hashes_for_block_size(hs, 16, 32))
    exists = {(0, bytes(h)) for h in hs}
    exists |= {(1, bytes(bh)) for bh in big_hashes}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64
    assert hit % 32 == 0


# ----- store_mask -----


def test_store_mask_full_attention_all_true():
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16)
    masks = coord.store_mask(64)
    assert masks == (None,)


def test_store_mask_zero_aligned_returns_empty_per_group():
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups, hash_block_size=16)
    masks = coord.store_mask(0)
    assert masks == (None, None)


def test_store_mask_swa_only_window_around_each_lcm_boundary():
    """Hybrid full-attn(block=32) + SWA(block=8, sw=8). lcm=32. With
    aligned=64 the SWA group should mark exactly the blocks ending at 32
    and 64 (i.e. blocks 3 and 7 at block_size=8); the rest can never
    participate in any future hit."""
    full = _full(32)
    swa = _swa(block_size=8, sliding_window=8)
    groups = [KVCacheGroupSpec(["L0"], full), KVCacheGroupSpec(["L1"], swa)]
    coord = _make_coord(groups, hash_block_size=8)
    masks = coord.store_mask(64)
    # Full-attn: 2 chunks * 32 tokens.
    assert masks[0] is None
    # SWA: 8 chunks * 8 tokens. Only chunks ending at 32 and 64 are stored.
    assert masks[1] == [False, False, False, True, False, False, False, True]


def test_store_mask_swa_wider_window_covers_more_blocks_per_lcm():
    """Same hybrid layout but sliding_window=16 (= 2 SWA blocks). Each lcm
    boundary should now span two SWA tail blocks."""
    full = _full(32)
    swa = _swa(block_size=8, sliding_window=16)
    groups = [KVCacheGroupSpec(["L0"], full), KVCacheGroupSpec(["L1"], swa)]
    coord = _make_coord(groups, hash_block_size=8)
    masks = coord.store_mask(64)
    assert masks[0] is None
    # Boundary at 32: blocks ending in [16, 32) — chunks 2 and 3.
    # Boundary at 64: chunks 6 and 7. Others stay False.
    assert masks[1] == [False, False, True, True, False, False, True, True]


def test_store_mask_dsv4_5_groups_full_mla_plus_4_swa():
    """DSV4-shaped: full-MLA(B=256) + 4 SWA groups with B in {64, 64, 4, 8}
    and varied sliding windows. lcm=256, hash_block_size=4. Two lcm segments
    (aligned_len=512). Validates that the tile-once strategy produces the
    expected per-segment tail-window pattern, repeated."""
    full_mla = _full(block_size=256)
    swa_64_sw128 = _swa(block_size=64, sliding_window=128)
    swa_64_sw512 = _swa(block_size=64, sliding_window=512)
    swa_4_sw16 = _swa(block_size=4, sliding_window=16)
    swa_8_sw64 = _swa(block_size=8, sliding_window=64)
    groups = [
        KVCacheGroupSpec(["L0"], full_mla),
        KVCacheGroupSpec(["L1"], swa_64_sw128),
        KVCacheGroupSpec(["L2"], swa_64_sw512),
        KVCacheGroupSpec(["L3"], swa_4_sw16),
        KVCacheGroupSpec(["L4"], swa_8_sw64),
    ]
    coord = _make_coord(groups, hash_block_size=4)
    assert coord.lcm_block_size == 256
    masks = coord.store_mask(512)

    # Full-MLA: 2 chunks of 256, both stored.
    assert masks[0] is None
    # SWA(64, sw=128): tail = ceil(127/64) = 2; C = 256/64 = 4.
    # Per-segment template = [F,F,T,T]; tiled twice.
    assert masks[1] == [False, False, True, True] * 2
    # SWA(64, sw=512): tail = 8 >= C = 4 → entire segment True.
    assert masks[2] is None
    # SWA(4, sw=16): tail = ceil(15/4) = 4; C = 256/4 = 64.
    # Last 4 of each 64-chunk segment True.
    assert masks[3] == ([False] * 60 + [True] * 4) * 2
    # SWA(8, sw=64): tail = ceil(63/8) = 8; C = 256/8 = 32.
    # Last 8 of each 32-chunk segment True.
    assert masks[4] == ([False] * 24 + [True] * 8) * 2


def test_store_mask_fast_path_all_block_sizes_equal_lcm():
    """When every non-full-attn group already aligns to lcm_block_size, the
    fast path returns all-True without invoking find_longest_cache_hit."""
    full = _full(block_size=64)
    swa = _swa(block_size=64, sliding_window=128)
    groups = [KVCacheGroupSpec(["L0"], full), KVCacheGroupSpec(["L1"], swa)]
    coord = _make_coord(groups, hash_block_size=64)
    assert coord.lcm_block_size == 64
    masks = coord.store_mask(256)
    # Every block in every group is True — no sub-lcm filtering possible.
    assert masks == (None, None)


def test_store_mask_fast_path_single_attention_group():
    """Two groups sharing the same SWA spec collapse to one attention group;
    no lcm filter applies, every chunk is True."""
    swa = _swa(block_size=16, sliding_window=32)
    groups = [KVCacheGroupSpec(["L0"], swa), KVCacheGroupSpec(["L1"], swa)]
    coord = _make_coord(groups, hash_block_size=16)
    assert len(coord.attention_groups) == 1
    masks = coord.store_mask(64)
    assert masks == (None, None)


# ----- store_mask with retention_interval (DSV4 sparse SWA checkpointing) -----


def _retention_groups():
    """Hybrid full-attn(block=32) + SWA(block=8, sw=8); lcm=32. The SWA group
    densely keeps one tail block per 32-token boundary."""
    full = _full(32)
    swa = _swa(block_size=8, sliding_window=8)
    return [KVCacheGroupSpec(["L0"], full), KVCacheGroupSpec(["L1"], swa)]


def test_store_mask_dense_default_matches_every_lcm_boundary():
    """retention_interval=None (default) keeps the SWA tail at every lcm
    boundary: tokens 32/64/96/128 -> chunks 3/7/11/15."""
    coord = _make_coord(_retention_groups(), hash_block_size=8)
    masks = coord.store_mask(128)
    assert masks[0] is None
    assert masks[1] == [i % 4 == 3 for i in range(16)]


def test_store_mask_retention_interval_sparsifies_swa_tails():
    """retention_interval=64 keeps an SWA tail once per 64-token segment
    (chunks 7 and 15) instead of every 32 tokens, dropping the mid-segment
    boundaries at 32 and 96."""
    coord = _make_coord(_retention_groups(), hash_block_size=8, retention_interval=64)
    masks = coord.store_mask(128)
    assert masks[0] is None  # full attn unaffected
    assert masks[1] == [i in (7, 15) for i in range(16)]


def test_store_mask_retention_interval_zero_keeps_only_replay_boundary():
    """retention_interval=0 drops all segment tails; only the latest replay
    boundary (capped at num_prompt-1, aligned down to lcm) is retained."""
    coord = _make_coord(_retention_groups(), hash_block_size=8, retention_interval=0)
    # No replay info -> nothing reachable for the SWA group.
    assert coord.store_mask(128)[1] == [False] * 16
    # num_prompt=100 -> latest hit boundary = (100-1)//32*32 = 96 -> chunk 11.
    masks = coord.store_mask(128, num_prompt_tokens=100)
    assert masks[1] == [i == 11 for i in range(16)]


def test_store_mask_retention_interval_keeps_segment_and_replay_tails():
    """Sparse segment tails (interval=64 -> chunks 7,15) plus the replay
    boundary tail (num_prompt=100 -> chunk 11) coexist."""
    coord = _make_coord(_retention_groups(), hash_block_size=8, retention_interval=64)
    masks = coord.store_mask(128, num_prompt_tokens=100)
    assert masks[1] == [i in (7, 11, 15) for i in range(16)]


# ----- Eagle / MTP interaction with load_mask -----


def test_lookup_with_eagle_pops_last_full_attention_block():
    """Sanity: with use_eagle, find_longest_cache_hit drops the last block.
    Pairs with the load_mask test below to lock the round-trip contract."""
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16, use_eagle=True)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    # 4 blocks present, eagle pops 1 → 3 blocks = 48 tokens.
    assert hit == 48


def test_load_mask_with_eagle_does_not_double_prune_full_attention():
    """Regression for silent KV corruption with MTP/EAGLE-3.

    The recv side calls ``load_mask(block_hashes, token_len)`` where
    ``token_len`` is already the eagle-pruned hit length from ``lookup``.
    A second eagle pop here used to shorten the mask by one extra block;
    ``process_tokens`` then yielded a chunk past the mask, which the worker
    silently skipped — leaving the trailing block of the loaded prefix
    uninitialized in local KV.
    """
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16, use_eagle=True)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 48  # eagle popped 1 block

    masks = coord.load_mask(hs, token_len=hit)
    # Every chunk that process_tokens(token_len=48, ...) would yield must
    # have a corresponding mask slot. process_tokens emits chunk_id 0..2
    # (start=0, 16, 32), so the mask must be length 3, all True.
    assert masks[0] == [True, True, True]


def test_load_mask_with_eagle_hybrid_full_plus_swa():
    """Hybrid (FullAttn + SWA) with eagle: load_mask must cover every chunk
    in [0, token_len) for the FullAttn group; SWA group keeps its
    tail-window mask."""
    groups = [
        KVCacheGroupSpec(["L0"], _full(16)),
        KVCacheGroupSpec(["L1"], _swa(16, 32)),
    ]
    coord = _make_coord(groups, hash_block_size=16, use_eagle=True)
    hs = _hashes(4)
    exists = {(g, bytes(h)) for g in (0, 1) for h in hs}
    cmap = ExternalCachedBlockPool(exists)
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    # FullAttn dictates the convergence; eagle pops one block off it.
    assert hit == 48

    masks = coord.load_mask(hs, token_len=hit)
    # FullAttn: all chunks populated locally.
    assert masks[0] == [True, True, True]
    # SWA: tail-window only (ceil((32-1)/16) = 2 trailing blocks).
    assert masks[1][-2:] == [True, True]


def test_load_mask_without_eagle_unchanged():
    """Sanity: when eagle is off, load_mask is identical to the pre-fix path."""
    groups = [KVCacheGroupSpec(["L0"], _full(16))]
    coord = _make_coord(groups, hash_block_size=16, use_eagle=False)
    hs = _hashes(4)
    cmap = ExternalCachedBlockPool({(0, bytes(h)) for h in hs})
    _masks, hit = coord.find_longest_cache_hit(
        hs, max_length=64, cached_block_pool=cmap
    )
    assert hit == 64
    masks = coord.load_mask(hs, token_len=hit)
    assert masks[0] == [True, True, True, True]
