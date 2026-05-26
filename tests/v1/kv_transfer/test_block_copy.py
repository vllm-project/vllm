# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for copy_kv_blocks with Triton attention backend layout.

Triton attention uses a blocks-first KV cache shape:
    [num_blocks, 2, block_size, num_kv_heads, head_dim]
where blocks are in dimension 0.

The NIXL connector determines the correct block dimension via
AttentionBackend.get_kv_cache_block_dim() and passes it to
copy_kv_blocks so that block copying works for all layouts.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM = 16, 4, 64
NUM_BLOCKS = 32


def _make_triton_kv_cache(num_blocks=NUM_BLOCKS, dtype=torch.float16):
    """Create a KV cache tensor in Triton blocks-first layout."""
    return torch.randn(num_blocks, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)


def test_copy_kv_blocks_h2d_triton():
    """NIXL h2d copy works for Triton blocks-first layout."""
    block_dim = TritonAttentionBackend.get_kv_cache_block_dim(
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_DIM
    )
    assert block_dim == 0

    src = _make_triton_kv_cache()
    dst = torch.zeros_like(src)

    src_ids = [3, 7, 15]
    dst_ids = [0, 1, 2]

    copy_kv_blocks(
        {"layer": src},
        {"layer": dst},
        src_ids,
        dst_ids,
        "h2d",
        block_dim=block_dim,
    )

    for s, d in zip(src_ids, dst_ids):
        assert torch.equal(dst[d], src[s])


def test_copy_kv_blocks_d2h_triton():
    """NIXL d2h copy works for Triton blocks-first layout."""
    block_dim = TritonAttentionBackend.get_kv_cache_block_dim(
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_DIM
    )
    assert block_dim == 0

    src = _make_triton_kv_cache()
    dst = torch.zeros_like(src)

    src_ids = [5, 10, 20]
    dst_ids = [0, 1, 2]

    copy_kv_blocks(
        {"layer": src},
        {"layer": dst},
        src_ids,
        dst_ids,
        "d2h",
        block_dim=block_dim,
    )

    for s, d in zip(src_ids, dst_ids):
        assert torch.equal(dst[d], src[s])
