# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check AITER's fused indexer prologue against the kernels it replaces."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import aiter_indexer_qk_fused_kernel

N_HEAD = 32
HEAD_DIM = 128
ROPE_DIM = 64
HALF = ROPE_DIM // 2
EPS = 1e-6
MAX_POSITION = 512

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or aiter_indexer_qk_fused_kernel() is None,
    reason="requires an AITER build exporting indexer_qk_rope_quant_and_cache",
)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool):
    """Rotate the leading ``ROPE_DIM`` lanes, leaving the rest untouched."""
    xf = x.float()
    if is_neox:
        x0, x1 = xf[..., :HALF], xf[..., HALF:ROPE_DIM]
    else:
        x0, x1 = xf[..., 0:ROPE_DIM:2], xf[..., 1:ROPE_DIM:2]
    r0 = (x0 * cos - x1 * sin).bfloat16().float()
    r1 = (x1 * cos + x0 * sin).bfloat16().float()

    out = xf.clone()
    if is_neox:
        out[..., :HALF], out[..., HALF:ROPE_DIM] = r0, r1
    else:
        out[..., 0:ROPE_DIM:2], out[..., 1:ROPE_DIM:2] = r0, r1
    return out


def _ref_k(k, weight, bias, cos, sin, is_neox) -> torch.Tensor:
    """Eager LayerNorm + RoPE, matching ``LayerNorm`` then ``get_rope``."""
    normed = torch.nn.functional.layer_norm(
        k.float(), (HEAD_DIM,), weight, bias, EPS
    ).type_as(k)
    return _rope(normed, cos, sin, is_neox).type_as(k).contiguous()


def _ref_q(q, weights, cos, sin, is_neox, softmax_scale, head_scale):
    """Eager spelling of ``_fused_indexer_q_rope_quant_kernel``."""
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    roped = _rope(q, cos.unsqueeze(1), sin.unsqueeze(1), is_neox)

    amax = roped.abs().amax(dim=-1)
    q_scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-10) / fp8_max)))
    q_fp8 = (roped / q_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q_fp8, weights.float() * q_scale * softmax_scale * head_scale


@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_tokens", [1, 37, 256])
def test_fused_qk_matches_separate_kernels(
    is_neox: bool, block_size: int, num_tokens: int
):
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_blocks = num_tokens // block_size + 4
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5

    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=torch.bfloat16, device=device)
    # Strided, like the caller's fused wk + weights_proj GEMM output.
    kw = torch.randn(num_tokens, HEAD_DIM + N_HEAD, dtype=torch.bfloat16, device=device)
    k, weights = kw[:, :HEAD_DIM], kw[:, HEAD_DIM:]
    norm_weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=device)
    norm_bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=device)
    cos_sin_cache = torch.randn(
        MAX_POSITION, ROPE_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, MAX_POSITION, (num_tokens,), dtype=torch.int64, device=device
    )
    slot_mapping = torch.randperm(num_blocks * block_size, device=device)[
        :num_tokens
    ].contiguous()
    cos = cos_sin_cache[positions, :HALF].float()
    sin = cos_sin_cache[positions, HALF:].float()

    cache_shape = (num_blocks, block_size, HEAD_DIM + 4)
    ref_cache = torch.zeros(cache_shape, dtype=torch.uint8, device=device)
    fused_cache = torch.zeros_like(ref_cache)

    ref_q, ref_weights = _ref_q(
        q, weights, cos, sin, is_neox, softmax_scale, head_scale
    )
    indexer_k_quant_and_cache_triton(
        _ref_k(k, norm_weight, norm_bias, cos, sin, is_neox),
        ref_cache,
        slot_mapping,
        HEAD_DIM,
        "ue8m0",
    )

    fused_q = torch.empty(q.shape, dtype=current_platform.fp8_dtype(), device=device)
    fused_weights = torch.empty(weights.shape, dtype=torch.float32, device=device)
    aiter_indexer_qk_fused_kernel()(
        q,
        fused_q,
        weights,
        fused_weights,
        k,
        fused_cache,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_sin_cache[:, :HALF],
        cos_sin_cache[:, HALF:],
        EPS,
        HEAD_DIM,
        "ue8m0",
        softmax_scale * head_scale,
        block_size > 1,
        is_neox,
    )

    assert (fused_cache != 0).any(), "fused kernel wrote no cache"
    torch.testing.assert_close(fused_cache, ref_cache, rtol=0, atol=0)
    torch.testing.assert_close(fused_weights, ref_weights, rtol=1e-5, atol=0)
    # One fp8 e4m3 code of drift, from the fp32 reduction order.
    torch.testing.assert_close(fused_q.float(), ref_q.float(), rtol=0.13, atol=1e-3)
