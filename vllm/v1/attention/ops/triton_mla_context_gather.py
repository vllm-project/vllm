# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather of the MLA prefill context from an fp8 paged latent cache.

The FP8 prefill attention needs the context latent in two forms: kv_c in bf16
for the kv_b projection, and k_pe in fp8 for the attention kernel. This kernel
writes both in one pass, so no separate cast runs after the gather.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_T = 16


@triton.jit
def _gather_mla_context_fp8_kernel(
    cache_ptr,
    kv_c_ptr,
    k_pe_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    starts_ptr,
    token_to_seq_ptr,
    k_scale_ptr,
    num_tokens,
    block_size,
    stride_cache_b,
    stride_cache_t,
    stride_table,
    stride_kv_c,
    stride_k_pe,
    KV_LORA: tl.constexpr,
    PE_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    ok = tok < num_tokens
    seq = tl.load(token_to_seq_ptr + tok, mask=ok, other=0)
    pos = (
        tl.load(starts_ptr + seq, mask=ok, other=0)
        + tok
        - tl.load(cu_seq_lens_ptr + seq, mask=ok, other=0)
    )
    block = tl.load(
        block_table_ptr + seq.to(tl.int64) * stride_table + pos // block_size,
        mask=ok,
        other=0,
    )
    row = (
        cache_ptr
        + block.to(tl.int64) * stride_cache_b
        + (pos % block_size).to(tl.int64) * stride_cache_t
    )
    scale = tl.load(k_scale_ptr)

    offs_c = tl.arange(0, KV_LORA)
    kv_c = tl.load(row[:, None] + offs_c[None, :], mask=ok[:, None])
    tl.store(
        kv_c_ptr + tok.to(tl.int64)[:, None] * stride_kv_c + offs_c[None, :],
        (kv_c.to(tl.float32) * scale).to(kv_c_ptr.dtype.element_ty),
        mask=ok[:, None],
    )
    offs_pe = tl.arange(0, PE_DIM)
    k_pe = tl.load(row[:, None] + KV_LORA + offs_pe[None, :], mask=ok[:, None])
    tl.store(
        k_pe_ptr + tok.to(tl.int64)[:, None] * stride_k_pe + offs_pe[None, :],
        (k_pe.to(tl.float32) * scale).to(k_pe_ptr.dtype.element_ty),
        mask=ok[:, None],
    )


def gather_mla_context_fp8(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    starts: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    k_scale: torch.Tensor,
    kv_lora_rank: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gathers the context tokens of one chunk from an fp8 MLA cache.

    cache: [num_blocks, block_size, kv_lora_rank + pe_dim] fp8 (or its uint8 view).
    block_table: [num_reqs, max_blocks] int32 rows of the chunk's requests.
    cu_seq_lens: [num_reqs + 1] int32 token offsets of the requests in the chunk.
    starts: [num_reqs] int32 position of each request's first gathered token.
    token_to_seq: [num_tokens] int32 request of each gathered token.
    Returns kv_c [num_tokens, kv_lora_rank] in bf16 and k_pe
    [num_tokens, 1, pe_dim] in ``fp8_dtype``, both dequantized with k_scale.
    """
    fp8 = fp8_dtype
    if cache.dtype != fp8:
        cache = cache.view(fp8)
    pe_dim = cache.shape[-1] - kv_lora_rank
    assert cache.stride(-1) == 1 and block_table.stride(-1) == 1
    kv_c = torch.empty(
        (num_tokens, kv_lora_rank), dtype=torch.bfloat16, device=cache.device
    )
    k_pe = torch.empty((num_tokens, 1, pe_dim), dtype=fp8, device=cache.device)
    if num_tokens == 0:
        return kv_c, k_pe
    _gather_mla_context_fp8_kernel[(triton.cdiv(num_tokens, _BLOCK_T),)](
        cache,
        kv_c,
        k_pe,
        block_table,
        cu_seq_lens,
        starts,
        token_to_seq,
        k_scale,
        num_tokens,
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        block_table.stride(0),
        kv_c.stride(0),
        k_pe.stride(0),
        KV_LORA=kv_lora_rank,
        PE_DIM=pe_dim,
        BLOCK_T=_BLOCK_T,
    )
    return kv_c, k_pe
