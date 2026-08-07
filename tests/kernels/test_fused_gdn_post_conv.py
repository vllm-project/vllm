# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDN post-convolution kernels.

The prefill tests cover preparation and the MTP tests cover recurrent state
updates plus output normalization and gating.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    rmsnorm_fn,
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
def test_fused_post_conv_sanity(L):
    """Sanity checks: no NaN, unit-norm q/k, beta in (0,1)."""
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


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("query_lengths", [(4, 4), (4, 2, 0), (8,)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_ratio8(
    tp_size: int,
    query_lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    norm_dtype: torch.dtype,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused GDN decode MTP requires SM100")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    torch.manual_seed(0)
    device = "cuda"
    H = 16 // tp_size
    HV = 128 // tp_size
    K = V = 128
    num_reqs = len(query_lengths)
    state_width = max(query_lengths)
    num_tokens = sum(query_lengths)
    num_slots = num_reqs * state_width + 1
    scale = K**-0.5
    eps = 1e-6

    mixed_qkv = torch.randn(
        num_tokens,
        2 * H * K + HV * V,
        dtype=torch.bfloat16,
        device=device,
    )
    query, key, value = torch.split(
        mixed_qkv,
        [H * K, H * K, HV * V],
        dim=-1,
    )
    query = query.view(1, num_tokens, H, K)
    key = key.view(1, num_tokens, H, K)
    value = value.view(1, num_tokens, HV, V)
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    assert not a.is_contiguous()
    assert not b.is_contiguous()
    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    output_gate = torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(V, dtype=norm_dtype, device=device)
    state_ref = (
        0.01 * torch.randn(num_slots, HV, V, K, dtype=torch.float32, device=device)
    ).to(state_dtype)
    state_actual = state_ref.clone()
    state_indices = torch.arange(1, num_slots, dtype=torch.int32, device=device).view(
        num_reqs, state_width
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.tensor(
        [1, *[state_width] * (num_reqs - 1)],
        dtype=torch.int32,
        device=device,
    )
    if query_lengths[-1] == 0:
        state_indices[-1].zero_()
        num_accepted_tokens[-1] = 1

    for step in range(3):
        raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state_ref,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        expected = rmsnorm_fn(
            raw_ref.squeeze(0),
            norm_weight,
            None,
            z=output_gate,
            eps=eps,
            norm_before_gate=True,
            activation="silu",
        )
        actual = ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            state=state_actual,
            output_gate=output_gate,
            norm_weight=norm_weight,
            out=torch.empty_like(output_gate),
            scale=scale,
            norm_eps=eps,
        )

        output_error = (actual.float() - expected.float()).norm()
        output_relative_l2 = output_error / expected.float().norm().clamp_min(1e-20)
        assert output_relative_l2 < 5e-4, (
            f"MTP output relative L2 mismatch at step {step}: "
            f"{output_relative_l2.item():.6g}"
        )

    torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)
