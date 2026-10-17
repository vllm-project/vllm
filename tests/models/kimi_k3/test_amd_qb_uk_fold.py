# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The folded q_b_proj + W_UK decode-query weight of the AMD Kimi-K3 MLA wrapper."""

import pytest
import torch

from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper
from vllm.platforms import current_platform

compute_w_fold = KimiK3MultiHeadLatentAttentionWrapper._compute_w_fold

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Kimi-K3 AMD MLA fold requires ROCm"
)

# Kimi-K3 MLA dims: q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim.
LQ, L, P, R = 1536, 512, 128, 64


def _weights(num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    w_qb = torch.randn(num_heads * (P + R), LQ, device="cuda", generator=gen) * 0.02
    w_uk = torch.randn(L, num_heads, P, device="cuda", generator=gen) * 0.02
    return w_qb.to(torch.bfloat16), w_uk.to(torch.bfloat16)


def _reference(
    x: torch.Tensor, w_qb: torch.Tensor, w_uk: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unfused path: q_b_proj, then the per-head W_UK absorb, in fp32."""
    q = (x.float() @ w_qb.float().T).view(-1, num_heads, P + R)
    q_nope, q_pe = q.split([P, R], dim=-1)
    return torch.einsum("bnp,lnp->bnl", q_nope, w_uk.float()), q_pe


def _split_fold_output(
    y: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The views _fused_decode takes on the fold GEMM output."""
    b = y.shape[0]
    return (
        y[:, : num_heads * L].view(b, num_heads, L),
        y[:, num_heads * L :].view(b, num_heads, R),
    )


@pytest.mark.parametrize("num_heads", [12, 24, 96])
def test_fold_row_layout(num_heads: int) -> None:
    w_qb, w_uk = _weights(num_heads)
    w_fold = compute_w_fold(w_qb, w_uk, num_heads, P, R)
    assert w_fold.shape == (num_heads * (L + R), LQ)
    assert w_fold.dtype == w_qb.dtype and w_fold.is_contiguous()
    w = w_qb.view(num_heads, P + R, LQ)
    # pe rows are the q_b_proj pe rows, head-major, unchanged
    torch.testing.assert_close(
        w_fold[num_heads * L :].view(num_heads, R, LQ), w[:, P:], rtol=0, atol=0
    )
    # nope rows of head h are W_UK[:, h] @ W_qb_nope[h] (fp32 product, one bf16
    # rounding; matmul vs einsum accumulation order can flip the last bf16 bit)
    h = num_heads - 1
    expect = w_uk[:, h].float() @ w[h, :P].float()
    torch.testing.assert_close(
        w_fold[h * L : (h + 1) * L].float(), expect, rtol=2**-7, atol=1e-4
    )


@pytest.mark.parametrize("num_heads", [12, 96])
@pytest.mark.parametrize("num_tokens", [1, 56, 112])
def test_fold_matches_unfused_projection(num_heads: int, num_tokens: int) -> None:
    w_qb, w_uk = _weights(num_heads)
    x = torch.randn(num_tokens, LQ, device="cuda").to(torch.bfloat16)
    ql_nope_ref, q_pe_ref = _reference(x, w_qb, w_uk, num_heads)
    w_fold = compute_w_fold(w_qb, w_uk, num_heads, P, R)
    ql_nope, q_pe = _split_fold_output(x.float() @ w_fold.float().T, num_heads)
    # the only extra rounding is W_fold's bf16 cast of the fp32 product; q_pe
    # uses the same bf16 weights, so it differs only by fp32 summation order
    torch.testing.assert_close(ql_nope, ql_nope_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q_pe, q_pe_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_tokens", [1, 56, 112])
def test_fold_through_aiter_gemm(num_tokens: int) -> None:
    """The kernel the wrapper calls on the folded weight agrees with torch."""
    pytest.importorskip("aiter")
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    num_heads = 12
    w_qb, w_uk = _weights(num_heads)
    x = torch.randn(num_tokens, LQ, device="cuda").to(torch.bfloat16)
    w_fold = compute_w_fold(w_qb, w_uk, num_heads, P, R)
    y = gemm_a16w16(x, w_fold)
    assert y.shape == (num_tokens, num_heads * (L + R)) and y.dtype == torch.bfloat16
    ql_nope, q_pe = _split_fold_output(y, num_heads)
    ql_nope_ref, q_pe_ref = _split_fold_output(x.float() @ w_fold.float().T, num_heads)
    torch.testing.assert_close(ql_nope.float(), ql_nope_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(q_pe.float(), q_pe_ref, rtol=2e-2, atol=2e-2)
