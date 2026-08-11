# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


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


def test_fused_sigmoid_gating_uses_bf16_semantics() -> None:
    """The fused decode update is closer to the input-dtype reference."""
    torch.set_default_device(DEVICE)
    set_random_seed(51779)
    B, T, H, HV, K, V = 4, 1, 2, 4, 128, 32

    q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
    a = torch.randn(B * T, HV, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(HV, dtype=torch.float32) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
    state_indices = torch.arange(1, B + 1, dtype=torch.int32)
    initial_state = torch.zeros(B + 1, HV, V, K, dtype=torch.bfloat16)

    g = (-A_log.exp() * F.softplus(a.float() + dt_bias)).unsqueeze(1)
    beta = b.sigmoid().unsqueeze(1)
    reference_out, reference_state = fused_recurrent_gated_delta_rule(
        q=_bf16_l2norm(q),
        k=_bf16_l2norm(k),
        v=v,
        g=g,
        beta=beta,
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
        initial_state=initial_state.clone(),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
    )
    promoted_errors = (
        _relative_l2(promoted_out.to(reference_out.dtype), reference_out),
        _relative_l2(promoted_state[state_indices], reference_state[state_indices]),
    )

    out, state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=initial_state.clone(),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    bf16_errors = (
        _relative_l2(out, reference_out),
        _relative_l2(state[state_indices], reference_state[state_indices]),
    )

    assert bf16_errors[0] < promoted_errors[0]
    assert bf16_errors[1] < promoted_errors[1]


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    seq_len = 1  # seq_len is 1 for decode
    num_tokens = num_reqs * seq_len
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    reference_query = _bf16_l2norm(query)
    reference_key = _bf16_l2norm(key)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=reference_query,
        k=reference_key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    num_speculative_tokens: int,
    dtype: torch.dtype,
) -> None:
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * (num_speculative_tokens + 1)
    total_entries = num_tokens * 2

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    # Entry 0 is reserved as NULL_BLOCK_ID (CUDA graph padding), so valid
    # state indices start at 1.
    ssm_state = torch.rand(
        total_entries + 1, num_v_heads, head_k_dim, head_v_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[
        :num_tokens
    ].view(num_reqs, num_speculative_tokens + 1)
    num_accepted_tokens = torch.randint(
        1, num_speculative_tokens + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(
        0, num_tokens + 1, num_speculative_tokens + 1, dtype=torch.int32
    )

    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    reference_query = _bf16_l2norm(query)
    reference_key = _bf16_l2norm(key)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=reference_query,
        k=reference_key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=False,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(core_attn_out, core_attn_out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        last_recurrent_state, last_recurrent_state_ref, atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("invalid_input", ["a", "b", "q", "k", "v"])
def test_fused_sigmoid_gating_rejects_non_bf16_activations(invalid_input: str):
    inputs = {
        "a": torch.empty(1, 1, dtype=torch.bfloat16, device="cpu"),
        "b": torch.empty(1, 1, dtype=torch.bfloat16, device="cpu"),
        "q": torch.empty(1, 1, 1, 1, dtype=torch.bfloat16, device="cpu"),
        "k": torch.empty(1, 1, 1, 1, dtype=torch.bfloat16, device="cpu"),
        "v": torch.empty(1, 1, 1, 1, dtype=torch.bfloat16, device="cpu"),
    }
    inputs[invalid_input] = inputs[invalid_input].float()

    with pytest.raises(AssertionError, match="only supports BF16 activations"):
        fused_sigmoid_gating_delta_rule_update(
            **inputs,
            A_log=torch.empty(1, device="cpu"),
            dt_bias=torch.empty(1, device="cpu"),
            initial_state=torch.empty(1, 1, 1, 1, dtype=torch.bfloat16, device="cpu"),
        )
