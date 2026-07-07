# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP + hybrid attention KV cache infrastructure.

Tests that resolve_kv_cache_block_sizes correctly handles SWA+DCP groups.
"""

import math
from unittest.mock import MagicMock

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
