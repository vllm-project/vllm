# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton MQA logits kernels."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton MQA logits kernels require CUDA/ROCm",
)


def _quantize_k_per_row(
    k_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    amax = k_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = amax / 448.0
    k_fp8 = (k_bf16.float() / sf).to(torch.float8_e4m3fn)
    return k_fp8, sf.squeeze(-1)


def _pack_paged_kv(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack BF16 KV into the layout produced by `indexer_k_quant_and_cache`.

    Physical bytes per block are segregated: all `block_size * head_dim` fp8 K
    bytes first, then `block_size * 4` fp32 scale bytes. The outer
    `[NB, BS, 1, D+4]` shape matches the production cache allocation.
    """
    num_blocks, block_size, head_dim = kv_bf16.shape
    amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    sf = (amax / 448.0).to(torch.float32)
    k_fp8 = (kv_bf16.float() / sf).to(torch.float8_e4m3fn)
    packed = torch.empty(
        (num_blocks, block_size, 1, head_dim + 4),
        dtype=torch.uint8,
        device=kv_bf16.device,
    )
    flat = packed.view(num_blocks, -1)
    k_end = block_size * head_dim
    flat[:, :k_end] = k_fp8.reshape(num_blocks, -1).view(torch.uint8)
    flat[:, k_end:] = sf.reshape(num_blocks, -1).view(torch.uint8)
    return packed


# References adapted from DeepGEMM (test_attention.py) — used as the spec
# the Triton kernels must agree with.
def _fp8_mqa_logits_ref(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    arange = torch.arange(0, seq_len_kv, device=q.device)[None, :]
    mask = (arange >= cu_seqlen_ks[:, None]) & (arange < cu_seqlen_ke[:, None])
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _fp8_paged_mqa_logits_ref(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    flat = kv_cache.view(num_blocks, -1)
    k_end = block_size * dim
    kv_data = (
        flat[:, :k_end].reshape(num_blocks, block_size, 1, dim).view(fp8_dtype).float()
    )
    scale = flat[:, k_end:].view(torch.float32).reshape(num_blocks, block_size, 1, 1)
    q = q.float()
    kv_data = kv_data * scale
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_data[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device=q.device,
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = (torch.relu(s) * weight_slice[..., None]).sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


# Looser tolerance to accommodate FP8 rounding and the paged torch
# reference using fp32 matmul while the triton kernel uses bf16 matmul
# (with an fp32 accumulator, matching the DeepGEMM path).
_ATOL = 1.0
_RTOL = 0.2


@pytest.mark.parametrize("M,N", [(64, 64), (128, 256), (256, 512)])
@pytest.mark.parametrize("num_heads", [16, 32])
@pytest.mark.parametrize("partial_mask", [False, True])
def test_fp8_mqa_logits_triton_matches_torch(M, N, num_heads, partial_mask):
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"

    q_bf16 = torch.randn(M, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(N, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(M, num_heads, dtype=torch.float32, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scales = _quantize_k_per_row(k_bf16)

    if partial_mask:
        # Exercise the non-trivial ks/ke masking branch (chunked prefill).
        ks = torch.arange(M, dtype=torch.int32, device=device) % (N // 4)
        ke = ks + torch.randint(1, N // 2, (M,), dtype=torch.int32, device=device)
        ke = torch.minimum(ke, torch.tensor(N, dtype=torch.int32, device=device))
    else:
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), N, dtype=torch.int32, device=device)

    out_torch = _fp8_mqa_logits_ref(q_fp8, (k_fp8, k_scales), weights, ks, ke)
    out_triton = fp8_mqa_logits_triton(q_fp8, (k_fp8, k_scales), weights, ks, ke)

    assert torch.equal(
        torch.isinf(out_torch) & (out_torch < 0),
        torch.isinf(out_triton) & (out_triton < 0),
    )
    finite = ~torch.isinf(out_torch)
    if finite.any():
        torch.testing.assert_close(
            out_triton[finite],
            out_torch[finite],
            atol=_ATOL,
            rtol=_RTOL,
        )


@pytest.mark.parametrize(
    "batch_size,next_n,context_len",
    [
        (1, 1, 128),
        (1, 1, 512),
        (2, 1, 256),
        (1, 4, 512),  # speculative decoding with next_n=4
    ],
)
@pytest.mark.parametrize("num_heads", [16, 32])
def test_fp8_paged_mqa_logits_triton_matches_torch(
    batch_size, next_n, context_len, num_heads
):
    torch.manual_seed(0)
    head_dim = 128
    block_size = 64
    device = "cuda"

    total_blocks = 64
    max_blocks = (context_len + block_size - 1) // block_size + 4

    kv_bf16 = torch.randn(
        total_blocks, block_size, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_packed = _pack_paged_kv(kv_bf16)

    q_fp8 = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )

    context_lens = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    block_tables = torch.randint(
        0,
        total_blocks,
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )

    max_model_len = max_blocks * block_size

    out_torch = _fp8_paged_mqa_logits_ref(
        q_fp8,
        kv_packed,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kv_packed,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    inf_torch = torch.isinf(out_torch) & (out_torch < 0)
    inf_triton = torch.isinf(out_triton) & (out_triton < 0)
    assert torch.equal(inf_torch, inf_triton)
    finite = ~inf_torch
    if finite.any():
        torch.testing.assert_close(
            out_triton[finite],
            out_torch[finite],
            atol=_ATOL,
            rtol=_RTOL,
        )
