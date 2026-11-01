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
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
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


@pytest.mark.parametrize("num_reqs", [1, 5])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 7])
def test_fused_sigmoid_gating_delta_rule_update_defer_spec_state(
    num_reqs: int,
    num_speculative_tokens: int,
) -> None:
    """Deferred spec states give the same outputs, step after step, as states
    stored after every token."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    max_query_len = num_speculative_tokens + 1
    A_log = torch.randn(num_v_heads) * 0.5
    dt_bias = torch.randn(num_v_heads)
    num_entries = num_reqs * max_query_len
    ssm_state = 0.1 * torch.randn(
        num_entries + 1, num_v_heads, head_dim, head_dim, dtype=torch.float32
    )
    ssm_state_defer = ssm_state.clone()
    state_indices = (torch.randperm(num_entries, dtype=torch.int32) + 1).view(
        num_reqs, max_query_len
    )
    query_lens = [1] * num_reqs
    for step in range(8):
        # Accept a prefix of the previous step's tokens, then run a new step
        # of random length.
        num_accepted_tokens = torch.tensor(
            [int(torch.randint(1, n + 1, ())) for n in query_lens],
            dtype=torch.int32,
        )
        if step == 0:
            num_accepted_tokens.fill_(1)
        query_lens = torch.randint(1, max_query_len + 1, (num_reqs,)).tolist()
        num_tokens = sum(query_lens)
        cu_seqlens = torch.tensor([0] + query_lens).cumsum(0).to(torch.int32)
        q = torch.randn(1, num_tokens, num_k_heads, head_dim, dtype=torch.bfloat16)
        k = torch.randn(1, num_tokens, num_k_heads, head_dim, dtype=torch.bfloat16)
        v = torch.randn(1, num_tokens, num_v_heads, head_dim, dtype=torch.bfloat16)
        a = torch.randn(num_tokens, num_v_heads, dtype=torch.bfloat16)
        b = torch.randn(num_tokens, num_v_heads, dtype=torch.bfloat16)
        outs = [
            fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                initial_state=state,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
                defer_spec_state=defer,
            )[0]
            for state, defer in ((ssm_state, False), (ssm_state_defer, True))
        ]
        # The modes differ only in launch configuration, hence in the order
        # of their reductions.
        torch.testing.assert_close(outs[1], outs[0], atol=2e-3, rtol=2e-3)
        # Slot 0 holds the state after the first token in both modes.
        first_slots = state_indices[:, 0].long()
        torch.testing.assert_close(
            ssm_state_defer[first_slots],
            ssm_state[first_slots],
            atol=1e-4,
            rtol=1e-4,
        )
