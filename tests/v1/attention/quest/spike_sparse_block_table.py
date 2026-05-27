# SPDX-License-Identifier: Apache-2.0
"""Spike: does FlashAttention paged kernel accept a sparse block_table?

Run manually on a GPU host:
    pytest tests/v1/attention/quest/spike_sparse_block_table.py::run \
        -v -s --no-header

This is NOT a Phase A exit gate. It informs Phase B metadata design.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("Spike requires CUDA", allow_module_level=True)


def run():
    """Build a tiny KV cache, populate two block_tables (full vs sparse),
    invoke flash_attn_varlen_func twice, and print whether outputs differ
    in the documented way (sparse should drop the unselected blocks).
    """
    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
        )
    except ImportError:
        pytest.skip("flash_attn_varlen_func not available")

    block_size = 16
    num_kv_heads = 2
    head_size = 64
    num_blocks = 8
    seqlen = num_blocks * block_size

    kv_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda",
    )

    q = torch.randn(1, num_kv_heads, head_size, dtype=torch.float16, device="cuda")
    full_block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    sparse_block_table = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device="cuda")

    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_lens_full = torch.tensor([seqlen], dtype=torch.int32, device="cuda")
    seq_lens_sparse = torch.tensor(
        [sparse_block_table.numel() * block_size],
        dtype=torch.int32,
        device="cuda",
    )

    out_full = flash_attn_varlen_func(
        q=q.unsqueeze(0),
        k=None, v=None,
        kv_cache=kv_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_full,
        max_seqlen_k=seqlen,
        causal=True,
        block_table=full_block_table.unsqueeze(0),
    )
    out_sparse = flash_attn_varlen_func(
        q=q.unsqueeze(0),
        k=None, v=None,
        kv_cache=kv_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens_sparse,
        max_seqlen_k=seq_lens_sparse.item(),
        causal=True,
        block_table=sparse_block_table.unsqueeze(0),
    )
    differ = not torch.allclose(out_full, out_sparse, atol=1e-3)
    print(f"sparse_block_table accepted={out_sparse is not None}, "
          f"differs_from_full={differ}")
