# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the aiter fused_qk_norm_mrope_kvcache wrapper.

The wrapper (registered as torch.ops.vllm.aiter_fused_qk_norm_mrope_kvcache)
combines per-head Q-RMSNorm, K-RMSNorm and 3D MRotaryEmbedding into one
HIP launch, and produces materialized Q/K/V outputs.  These tests compare
its output against an explicit decomposition (RMSNorm + RMSNorm +
MRotaryEmbedding.forward_native).
"""

import pytest
import torch

# Import side-effect: registers torch.ops.vllm.aiter_fused_qk_norm_mrope_kvcache
from vllm._aiter_ops import check_aiter_fused_qk_norm_mrope_kvcache  # noqa: F401
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and check_aiter_fused_qk_norm_mrope_kvcache()),
    reason=(
        "aiter_fused_qk_norm_mrope_kvcache requires ROCm + an aiter build "
        "that exposes fused_qk_norm_mrope_3d_cache_pts_quant_shuffle"
    ),
)


def _reference_qk_norm_mrope(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    mrope: MRotaryEmbedding,
    positions: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_size: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: per-head RMSNorm on Q and K, then MRotary on (Q, K)."""
    num_tokens = qkv.size(0)
    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_norm = RMSNorm(head_size, eps=eps).to(qkv.device, qkv.dtype)
    k_norm = RMSNorm(head_size, eps=eps).to(qkv.device, qkv.dtype)
    with torch.no_grad():
        q_norm.weight.copy_(q_norm_weight)
        k_norm.weight.copy_(k_norm_weight)

    q_by_head = q.view(num_tokens, num_heads_q, head_size)
    q_normed = q_norm(q_by_head).view(num_tokens, q_size)
    k_by_head = k.view(num_tokens, num_heads_kv, head_size)
    k_normed = k_norm(k_by_head).view(num_tokens, kv_size)

    q_out, k_out = mrope.forward_native(positions, q_normed, k_normed)
    return (
        q_out.view(num_tokens, num_heads_q, head_size),
        k_out.view(num_tokens, num_heads_kv, head_size),
        v.view(num_tokens, num_heads_kv, head_size),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("mrope_interleaved", [True, False])
def test_aiter_fused_qk_norm_mrope_matches_decomposition(
    dtype: torch.dtype, num_tokens: int, mrope_interleaved: bool
):
    """Fused output must match RMSNorm + RMSNorm + MRotary decomposition."""
    torch.manual_seed(0)
    device = torch.device("cuda")

    num_heads_q = 8
    num_heads_kv = 2
    head_size = 128  # rotary_dim = head_size for Qwen3-Omni-style attention
    eps = 1e-6
    # mrope_section must sum to rotary_dim // 2 = 64
    mrope_section = [24, 20, 20]

    mrope = MRotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=True,
        dtype=dtype,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    ).to(device)

    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)

    # 3D mrope positions [3, num_tokens] (T, H, W).
    positions = torch.stack(
        [torch.arange(num_tokens, device=device, dtype=torch.long)] * 3, dim=0
    )

    # Reference: use a fresh copy of qkv because forward_native is non-inplace
    # but we want to be sure the fused path doesn't mutate the input either.
    q_ref, k_ref, v_ref = _reference_qk_norm_mrope(
        qkv.clone(),
        q_weight,
        k_weight,
        mrope,
        positions,
        num_heads_q,
        num_heads_kv,
        head_size,
        eps,
    )

    cos_sin_cache = mrope._match_cos_sin_cache_dtype(qkv)

    q_out = torch.empty(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k_out = torch.empty(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v_out = torch.empty(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    torch.ops.vllm.aiter_fused_qk_norm_mrope_kvcache(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_out,
        k_out,
        v_out,
        num_heads_q,
        num_heads_kv,
        num_heads_kv,
        head_size,
        True,  # is_neox_style
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        eps,
    )

    # V is a pure passthrough (no RMSNorm, no rotation): must be bit-exact.
    torch.testing.assert_close(v_out, v_ref, atol=0, rtol=0)

    # Q and K go through fp32 RMSNorm + fp32 rotation then cast back to dtype.
    # The kernel's reduction and trig order differs slightly from PyTorch's
    # native path, so allow a small absolute tolerance.
    tol = 1e-2 if dtype is torch.float16 else 2e-2
    torch.testing.assert_close(q_out, q_ref, atol=tol, rtol=0)
    torch.testing.assert_close(k_out, k_ref, atol=tol, rtol=0)


def test_aiter_fused_qk_norm_mrope_fake_impl():
    """The fake impl must be registered for torch.compile compatibility."""
    num_tokens = 2
    num_heads_q = 8
    num_heads_kv = 2
    head_size = 128
    q_size = num_heads_q * head_size
    kv_size = num_heads_kv * head_size
    dtype = torch.float16
    device = torch.device("cuda")

    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    cos_sin_cache = torch.randn(4096, head_size, dtype=dtype, device=device)
    positions = torch.zeros(3, num_tokens, dtype=torch.long, device=device)
    q_out = torch.empty(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    k_out = torch.empty(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)
    v_out = torch.empty(num_tokens, num_heads_kv, head_size, dtype=dtype, device=device)

    torch.library.opcheck(
        torch.ops.vllm.aiter_fused_qk_norm_mrope_kvcache,
        (
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q_out,
            k_out,
            v_out,
            num_heads_q,
            num_heads_kv,
            num_heads_kv,
            head_size,
            True,  # is_neox_style
            24,
            20,
            20,
            True,  # is_interleaved
            1e-6,
        ),
        test_utils=("test_faketensor",),
    )
