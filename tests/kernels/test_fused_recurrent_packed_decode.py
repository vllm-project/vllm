# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)


def _bf16_l2norm(value: torch.Tensor) -> torch.Tensor:
    inverse_norm = torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + 1e-6)
    return value * inverse_norm


def _promoted_l2norm(value: torch.Tensor) -> torch.Tensor:
    value_f32 = value.float()
    inverse_norm = torch.rsqrt((value_f32 * value_f32).sum(dim=-1, keepdim=True) + 1e-6)
    return value_f32 * inverse_norm


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    return (
        torch.linalg.vector_norm(actual_f32 - expected_f32)
        / torch.linalg.vector_norm(expected_f32)
    ).item()


def test_fused_recurrent_packed_decode_uses_bf16_semantics_by_default():
    """Packed decode uses input-dtype normalization for BF16 by default."""
    torch.manual_seed(51779)
    device = torch.device("cuda")
    B, H, HV, K, V = 8, 2, 4, 128, 32
    qkv_dim = 2 * H * K + HV * V

    mixed_qkv = torch.randn(B, qkv_dim, dtype=torch.bfloat16, device=device)
    a = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    state_indices = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    initial_state = torch.zeros(B + 1, HV, V, K, dtype=torch.bfloat16, device=device)

    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, 1, H, K)
    k = k.view(B, 1, H, K)
    v = v.view(B, 1, HV, V)
    g = (-A_log.exp() * F.softplus(a.float() + dt_bias)).unsqueeze(1)
    beta = b.sigmoid().unsqueeze(1)
    reference_out, reference_state = fused_recurrent_gated_delta_rule(
        q=_bf16_l2norm(q),
        k=_bf16_l2norm(k),
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state.clone(),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
    )

    promoted_out, promoted_state = fused_recurrent_gated_delta_rule(
        q=_promoted_l2norm(q),
        k=_promoted_l2norm(k),
        v=v,
        g=g,
        beta=b.float().sigmoid().unsqueeze(1),
        scale=K**-0.5,
        initial_state=initial_state.clone(),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
    )
    promoted_errors = (
        _relative_l2(promoted_out.to(reference_out.dtype), reference_out),
        _relative_l2(promoted_state[state_indices], reference_state[state_indices]),
    )

    out = torch.empty(B, 1, HV, V, dtype=torch.bfloat16, device=device)
    _, state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=initial_state.clone(),
        out=out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    bf16_errors = (
        _relative_l2(out, reference_out),
        _relative_l2(state[state_indices], reference_state[state_indices]),
    )

    assert bf16_errors[0] < promoted_errors[0]
    assert bf16_errors[1] < promoted_errors[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device("cuda")

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = b.sigmoid().unsqueeze(1)

    q = _bf16_l2norm(q)
    k = _bf16_l2norm(k)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2
    rtol = 1e-2
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device("cuda")
    gates = torch.empty((B, HV), device=device, dtype=torch.bfloat16)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device, dtype=torch.bfloat16)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty(
            (B, 2 * H * K + HV * V), device=device, dtype=torch.bfloat16
        ),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device, dtype=torch.bfloat16),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("invalid_input", ["mixed_qkv", "a", "b", "out"])
def test_fused_recurrent_packed_decode_rejects_non_bf16_activations(
    invalid_input: str,
):
    inputs = {
        "mixed_qkv": torch.empty(1, 3, dtype=torch.bfloat16),
        "a": torch.empty(1, 1, dtype=torch.bfloat16),
        "b": torch.empty(1, 1, dtype=torch.bfloat16),
        "out": torch.empty(1, 1, 1, 1, dtype=torch.bfloat16),
    }
    inputs[invalid_input] = inputs[invalid_input].float()

    with pytest.raises(AssertionError, match="only supports BF16 activations"):
        fused_recurrent_gated_delta_rule_packed_decode(
            **inputs,
            A_log=torch.empty(1),
            dt_bias=torch.empty(1),
            scale=1.0,
            initial_state=torch.empty(1, 1, 1, 1, dtype=torch.bfloat16),
            ssm_state_indices=torch.ones(1, dtype=torch.int32),
        )
