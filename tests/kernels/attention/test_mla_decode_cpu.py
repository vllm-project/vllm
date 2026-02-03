# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def ref_mla(
    out: Tensor,  # (bs, num_heads, v_head_dim)
    query: Tensor,  # (bs, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("mean_seq_len", [256])
@pytest.mark.parametrize("h_q", [16])
@pytest.mark.parametrize("d", [576])
@pytest.mark.parametrize("dv", [512])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.cpu_model
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")
def test_mla_decode_cpu(
    bs: int,
    mean_seq_len: int,
    h_q: int,
    d: int,
    dv: int,
    block_size: int,
    dtype: torch.dtype,
    varlen: bool,
):
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    scale = d ** (-0.5)
    if varlen:
        seq_lens = torch.empty(bs).normal_(mean_seq_len, mean_seq_len / 2)
        seq_lens = seq_lens.clip(2).to(torch.int32)
    else:
        seq_lens = torch.full((bs,), mean_seq_len, dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    seqlen_pad = cdiv(max_seq_len, 256) * 256  # is this necessary?

    q = torch.randn(bs, h_q, d)
    block_table = torch.arange(bs * seqlen_pad // block_size, dtype=torch.int32)
    block_table = block_table.view(bs, seqlen_pad // block_size)

    kv_cache = torch.randn(block_table.numel(), block_size, d)
    for i, seq_len in enumerate(seq_lens.tolist()):
        kv_cache.view(bs, seqlen_pad, d)[i, seq_len:] = float("nan")

    out_mla = q.new_zeros(bs, h_q, dv)
    ops.mla_decode_kvcache_cpu(out_mla, q, kv_cache, scale, block_table, seq_lens)

    out_ref = q.new_zeros(bs, h_q, dv)
    ref_mla(out_ref, q, kv_cache, scale, block_table, seq_lens)

    assert not out_mla.isnan().any(), "Likely read out of bounds"
    torch.testing.assert_close(out_mla, out_ref)
