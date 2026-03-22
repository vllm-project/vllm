# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused_gdn_prefill_post_conv kernel.

Verifies that the fused kernel matches the reference:
  split → rearrange → contiguous → l2norm → gating
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)


def reference_post_conv(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    H: int,
    K: int,
    V: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
):
    """Reference implementation using individual ops."""
    L = conv_output.shape[0]
    HV = A_log.shape[0]

    # Split
    q_flat, k_flat, v_flat = torch.split(conv_output, [H * K, H * K, HV * V], dim=-1)

    # Rearrange + contiguous
    q = q_flat.view(L, H, K).contiguous()
    k = k_flat.view(L, H, K).contiguous()
    v = v_flat.view(L, HV, V).contiguous()

    # L2 norm
    if apply_l2norm:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)

    # Gating
    x = a.float() + dt_bias.float()
    sp = F.softplus(x, beta=1.0, threshold=20.0)
    g = -torch.exp(A_log.float()) * sp

    if output_g_exp:
        g = torch.exp(g)

    beta_out = torch.sigmoid(b.float())

    return q, k, v, g, beta_out


# Qwen3.5-35B config: H=16, HV=32, K=128, V=128
# Qwen3.5-397B config: H=16, HV=64, K=128, V=128
@pytest.mark.parametrize(
    "H, HV, K, V",
    [
        (16, 32, 128, 128),  # 35B
        (16, 64, 128, 128),  # 397B
        (4, 8, 64, 64),  # small
    ],
)
@pytest.mark.parametrize("L", [1, 16, 128, 512, 2048])
@pytest.mark.parametrize("apply_l2norm", [True, False])
@pytest.mark.parametrize("output_g_exp", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_post_conv_correctness(H, HV, K, V, L, apply_l2norm, output_g_exp, dtype):
    """Test fused kernel matches reference for all configs."""
    torch.manual_seed(42)
    device = "cuda"
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.randn(L, qkv_dim, dtype=dtype, device=device)
    a = torch.randn(L, HV, dtype=dtype, device=device)
    b = torch.randn(L, HV, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1

    # Reference
    ref_q, ref_k, ref_v, ref_g, ref_beta = reference_post_conv(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        H,
        K,
        V,
        apply_l2norm,
        output_g_exp,
    )

    # Fused kernel
    fused_q, fused_k, fused_v, fused_g, fused_beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=apply_l2norm,
        output_g_exp=output_g_exp,
    )

    # Check shapes
    assert fused_q.shape == (L, H, K), f"q shape: {fused_q.shape}"
    assert fused_k.shape == (L, H, K), f"k shape: {fused_k.shape}"
    assert fused_v.shape == (L, HV, V), f"v shape: {fused_v.shape}"
    assert fused_g.shape == (L, HV), f"g shape: {fused_g.shape}"
    assert fused_beta.shape == (L, HV), f"beta shape: {fused_beta.shape}"

    # Check dtypes
    assert fused_q.dtype == dtype
    assert fused_k.dtype == dtype
    assert fused_v.dtype == dtype
    assert fused_g.dtype == torch.float32
    assert fused_beta.dtype == torch.float32

    # Check contiguity
    assert fused_q.is_contiguous()
    assert fused_k.is_contiguous()
    assert fused_v.is_contiguous()

    # Check values
    atol_qkv = 1e-2 if apply_l2norm else 1e-3
    rtol_qkv = 1e-2 if apply_l2norm else 1e-3

    torch.testing.assert_close(fused_q, ref_q, atol=atol_qkv, rtol=rtol_qkv)
    torch.testing.assert_close(fused_k, ref_k, atol=atol_qkv, rtol=rtol_qkv)
    torch.testing.assert_close(fused_v, ref_v, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fused_g, ref_g, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fused_beta, ref_beta, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("L", [1, 64, 256])
def test_fused_post_conv_empty_and_edge(L):
    """Test edge cases."""
    torch.manual_seed(0)
    device = "cuda"
    H, HV, K, V = 16, 32, 128, 128
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.randn(L, qkv_dim, dtype=torch.bfloat16, device=device)
    a = torch.randn(L, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(L, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
    )

    # Basic sanity
    assert not torch.isnan(q).any(), "NaN in q"
    assert not torch.isnan(k).any(), "NaN in k"
    assert not torch.isnan(v).any(), "NaN in v"
    assert not torch.isnan(g).any(), "NaN in g"
    assert not torch.isnan(beta).any(), "NaN in beta"

    # L2 norm check: each head vector should have unit norm
    q_norms = torch.norm(q.float(), dim=-1)
    k_norms = torch.norm(k.float(), dim=-1)
    torch.testing.assert_close(q_norms, torch.ones_like(q_norms), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k_norms, torch.ones_like(k_norms), atol=1e-3, rtol=1e-3)

    # Beta should be in (0, 1)
    assert (beta >= 0).all() and (beta <= 1).all(), "beta out of range"


def test_fused_post_conv_l0():
    """Test L=0 edge case."""
    device = "cuda"
    H, HV, K, V = 16, 32, 128, 128
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.empty(0, qkv_dim, dtype=torch.bfloat16, device=device)
    a = torch.empty(0, HV, dtype=torch.bfloat16, device=device)
    b = torch.empty(0, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
    )
    assert q.shape == (0, H, K)
    assert g.shape == (0, HV)
