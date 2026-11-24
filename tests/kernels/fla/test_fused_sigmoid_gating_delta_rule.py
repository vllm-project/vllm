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


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize(
    "head_k_dim,head_v_dim,beta_scale", [(128, 128, 1.0), (96, 192, 2.0)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_non_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    beta_scale: float,
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
        total_entries + 1, num_v_heads, head_v_dim, head_k_dim, dtype=dtype
    )
    state_indices = (torch.randperm(total_entries, dtype=torch.int32) + 1)[:num_tokens]
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)

    beta = b.float().sigmoid() * beta_scale
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        beta_scale=beta_scale,
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


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 7])
def test_gdn_spec_window_matches_single_token_steps(
    state_dtype: torch.dtype, width: int
) -> None:
    """Verification must carry the same state as a cache reload at each step."""
    set_random_seed(907)
    num_reqs, num_k_heads, num_v_heads, head_dim = 4, 2, 16, 128
    options = {"device": DEVICE, "dtype": torch.bfloat16}
    q = torch.randn(num_reqs, width + 1, num_k_heads, head_dim, **options)
    k = torch.randn_like(q)
    v = torch.randn(num_reqs, width + 1, num_v_heads, head_dim, **options)
    a = torch.randn(num_reqs, width + 1, num_v_heads, **options)
    b = torch.randn_like(a)
    initial = (
        torch.randn(
            num_reqs, num_v_heads, head_dim, head_dim, device=DEVICE, dtype=state_dtype
        )
        * 0.05
    )
    A_log = torch.randn(num_v_heads, device=DEVICE) * 0.2
    dt_bias = torch.randn_like(A_log) * 0.2
    first_token = torch.ones(num_reqs, dtype=torch.int32, device=DEVICE)

    def verify(start: int, length: int, initial_state: torch.Tensor):
        tokens = slice(start, start + length)
        total = num_reqs * length
        slots = torch.arange(1, total + 1, device=DEVICE, dtype=torch.int32)
        slots = slots.view(num_reqs, length)
        state = initial_state.new_zeros(total + 1, num_v_heads, head_dim, head_dim)
        state[slots[:, 0].long()] = initial_state
        out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a[:, tokens].flatten(0, 1),
            b=b[:, tokens].flatten(0, 1),
            dt_bias=dt_bias,
            q=q[:, tokens].flatten(0, 1).unsqueeze(0),
            k=k[:, tokens].flatten(0, 1).unsqueeze(0),
            v=v[:, tokens].flatten(0, 1).unsqueeze(0),
            initial_state=state,
            cu_seqlens=torch.arange(
                0, total + 1, length, device=DEVICE, dtype=torch.int32
            ),
            ssm_state_indices=slots,
            num_accepted_tokens=first_token,
            use_qk_l2norm_in_kernel=True,
        )
        return out.view(num_reqs, length, num_v_heads, head_dim), state[slots.long()]

    out, states = verify(0, width, initial)
    serial_out, serial_states = [], []
    carry = initial
    for step in range(width):
        step_out, step_state = verify(step, 1, carry)
        serial_out.append(step_out)
        serial_states.append(step_state)
        carry = step_state[:, 0]

    expected_states = torch.cat(serial_states, dim=1)
    torch.testing.assert_close(out, torch.cat(serial_out, dim=1), atol=0, rtol=0)
    torch.testing.assert_close(states, expected_states, atol=0, rtol=0)

    rows = torch.arange(num_reqs, device=DEVICE)
    accepted = (rows + 1) % width
    next_out, _ = verify(width, 1, states[rows, accepted])
    expected_next_out, _ = verify(width, 1, expected_states[rows, accepted])
    torch.testing.assert_close(next_out, expected_next_out, atol=0, rtol=0)


@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize(
    "head_k_dim,head_v_dim,beta_scale", [(128, 128, 1.0), (96, 192, 2.0)]
)
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sigmoid_gating_delta_rule_update_spec(
    tp_size: int,
    num_reqs: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    beta_scale: float,
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
        total_entries + 1, num_v_heads, head_v_dim, head_k_dim, dtype=dtype
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

    beta = b.float().sigmoid() * beta_scale
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    core_attn_out_ref, last_recurrent_state_ref = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=ssm_state.clone(),
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out, last_recurrent_state = fused_sigmoid_gating_delta_rule_update(
        beta_scale=beta_scale,
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
