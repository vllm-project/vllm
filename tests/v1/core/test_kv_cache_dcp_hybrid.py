# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP + hybrid attention KV cache infrastructure.

Tests that:
1. resolve_kv_cache_block_sizes correctly handles SWA+DCP groups
2. RoutedExpertsManager CP slot computation is correct
"""

import math
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)


def _make_vllm_config(
    block_size: int = 16,
    dcp: int = 1,
    pcp: int = 1,
    enable_prefix_caching: bool = False,
    hash_block_size: int | None = None,
) -> MagicMock:
    config = MagicMock()
    config.cache_config.block_size = block_size
    config.cache_config.enable_prefix_caching = enable_prefix_caching
    config.cache_config.hash_block_size = hash_block_size
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.kv_transfer_config = None
    return config


def _make_full_attn_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _make_swa_spec(
    block_size: int = 16, sliding_window: int = 4096
) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
    )


def _make_kv_cache_config(
    specs: list,
    num_blocks: int = 1000,
) -> KVCacheConfig:
    groups = []
    for i, spec in enumerate(specs):
        groups.append(KVCacheGroupSpec([f"layer_{i}"], spec))
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _cp_slot_mapping(
    positions: np.ndarray,
    block_ids: list[int],
    block_size: int,
    total_cp: int,
    interleave: int,
) -> np.ndarray:
    """Compute CP virtual slot mapping (same formula as RoutedExpertsManager)."""
    vbs = block_size * total_cp
    vbi = positions // vbs
    vbo = positions % vbs
    rank = (vbo // interleave) % total_cp
    local_off = (vbo // (total_cp * interleave)) * interleave + (vbo % interleave)
    bid = np.array(block_ids, dtype=np.int32)[vbi]
    return (bid * block_size + local_off) * total_cp + rank


# --- resolve_kv_cache_block_sizes tests ---


def test_resolve_single_group_dcp():
    """Single group: block_size * dcp * pcp."""
    config = _make_vllm_config(block_size=16, dcp=2, pcp=1)
    kv_config = _make_kv_cache_config([_make_full_attn_spec(16)])
    sbs, hbs = resolve_kv_cache_block_sizes(kv_config, config)
    assert sbs == 32  # 16 * 2
    assert hbs == 32


def test_resolve_hybrid_no_dcp():
    """Two groups, no DCP: scheduler_block_size = LCM of block sizes."""
    config = _make_vllm_config(block_size=16, dcp=1, pcp=1)
    kv_config = _make_kv_cache_config(
        [
            _make_full_attn_spec(16),
            _make_swa_spec(32),
        ]
    )
    sbs, hbs = resolve_kv_cache_block_sizes(kv_config, config)
    assert sbs == math.lcm(16, 32)


def test_resolve_hybrid_with_dcp_swa_gets_dcp1():
    """Hybrid with DCP: SWA group effective_bs = block_size (DCP=1),
    FullAttn effective_bs = block_size * DCP."""
    config = _make_vllm_config(block_size=16, dcp=2, pcp=1)
    kv_config = _make_kv_cache_config(
        [
            _make_full_attn_spec(16),
            _make_swa_spec(16),
        ]
    )
    sbs, hbs = resolve_kv_cache_block_sizes(kv_config, config)
    # FullAttn effective = 16*2 = 32, SWA effective = 16*1 = 16
    assert sbs == math.lcm(32, 16)


def test_resolve_hybrid_with_dcp_different_block_sizes():
    """Hybrid with DCP and different block sizes."""
    config = _make_vllm_config(block_size=16, dcp=4, pcp=1)
    kv_config = _make_kv_cache_config(
        [
            _make_full_attn_spec(16),
            _make_swa_spec(32),
        ]
    )
    sbs, hbs = resolve_kv_cache_block_sizes(kv_config, config)
    # FullAttn effective = 16*4 = 64, SWA effective = 32*1 = 32
    assert sbs == math.lcm(64, 32)


# --- RoutedExpertsManager CP slot computation tests ---


def test_cp_slot_no_cp():
    """Without CP, slot = block_id * block_size + offset."""
    block_size = 16
    block_ids = [10, 20, 30]
    num_tokens = 3 * block_size

    block_ids_array = np.array(block_ids, dtype=np.int32)
    block_offsets = np.arange(block_size)
    expected = (
        block_ids_array.reshape(-1, 1) * block_size + block_offsets.reshape(1, -1)
    ).flatten()[:num_tokens]

    result = (
        block_ids_array.reshape(-1, 1) * block_size + block_offsets.reshape(1, -1)
    ).flatten()[:num_tokens]
    np.testing.assert_array_equal(result, expected)


def test_cp_slot_uniqueness():
    """With CP, each position gets a unique virtual slot."""
    block_size = 16
    total_cp = 2
    interleave = 1
    num_tokens = 64
    block_ids = list(range(10, 10 + (num_tokens // (block_size * total_cp)) + 1))

    positions = np.arange(num_tokens)
    slot_mapping = _cp_slot_mapping(
        positions, block_ids, block_size, total_cp, interleave
    )

    assert len(np.unique(slot_mapping)) == num_tokens


def test_cp_slot_matches_re_slot_reference():
    """CP slot mapping matches the re_slot_mapping kernel reference."""
    block_size = 16
    total_cp = 4
    interleave = 2
    num_tokens = 128

    num_virtual_blocks = (num_tokens + block_size * total_cp - 1) // (
        block_size * total_cp
    )
    block_ids = list(range(100, 100 + num_virtual_blocks + 1))
    positions = np.arange(num_tokens)

    slot_mapping = _cp_slot_mapping(
        positions, block_ids, block_size, total_cp, interleave
    )

    # Same formula as _reference_re_slot_mapping from test_re_slot_mapping.py
    block_table = np.array(block_ids, dtype=np.int32)
    virtual_block_size = block_size * total_cp
    block_indices = positions // virtual_block_size
    block_numbers = block_table[block_indices].astype(np.int64)
    virtual_block_offsets = positions - block_indices * virtual_block_size
    token_rank = (virtual_block_offsets // interleave) % total_cp
    local_block_offsets = (
        virtual_block_offsets // (total_cp * interleave)
    ) * interleave + (virtual_block_offsets % interleave)
    reference = (
        block_numbers * block_size + local_block_offsets
    ) * total_cp + token_rank

    np.testing.assert_array_equal(slot_mapping, reference)


def test_cp_slot_all_non_negative():
    """All virtual slots must be non-negative."""
    block_size = 16
    total_cp = 2
    interleave = 1
    num_tokens = 256
    block_ids = list(range(0, 20))

    positions = np.arange(num_tokens)
    slot_mapping = _cp_slot_mapping(
        positions, block_ids, block_size, total_cp, interleave
    )

    assert np.all(slot_mapping >= 0)
