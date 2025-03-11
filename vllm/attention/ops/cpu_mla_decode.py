# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm import _custom_ops as ops


def mla_decode_kvcache_cpu(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, num_heads, qk_head_dim)
        kv_cache: (num_blocks, block_size, qk_head_dim + v_head_dim)
        block-table: (batch_size, max_num_blocks_per_seq)
        seq_lens: (batch_size,)
        softmax_scale: float
    """
    bsize, num_heads, qk_head_dim = q.shape
    v_head_dim = kv_cache.shape[2] - qk_head_dim

    if softmax_scale is None:
        softmax_scale = qk_head_dim**(-0.5)

    out = q.new_empty(bsize, num_heads, v_head_dim)
    ops.mla_decode_kvcache_cpu(out, q, kv_cache, softmax_scale, block_tables,
                               seq_lens)
    return out
