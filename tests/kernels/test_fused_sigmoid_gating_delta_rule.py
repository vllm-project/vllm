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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA/ROCm GPU"
)
def test_fused_sigmoid_gating_readout_finite_under_large_fp32_state():
    """Readout must stay finite when the fp32 SSM state is large.

    The gated-delta-net recurrent state is an *unnormalized* accumulator that
    grows over long, low-decay context (it is normalized only at the readout by
    RMSNormGated). With an fp32 SSM state cache (``mamba_ssm_cache_dtype=float32``)
    the readout ``o = state @ q`` can exceed fp16's ~65504 range. If the output
    is materialized in the fp16 activation dtype it overflows to +/-inf and, after
    normalization, becomes NaN -> the model degenerates to a single repeated
    token. Regression for that overflow: fails when the output is downcast to
    fp16, passes when it honors the fp32 state precision.
    """
    set_random_seed(0)
    B, T, H, HV, K, V = 1, 1, 4, 4, 64, 128
    dt = torch.float16
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=dt)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=dt)
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=dt)
    a = torch.randn(B, T, HV, device=DEVICE, dtype=dt)
    b = torch.randn(B, T, HV, device=DEVICE, dtype=dt)
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEVICE, dtype=torch.float32)
    # fp32 state at a magnitude reachable over long context; 1e6 > fp16 max.
    state = torch.randn(B, HV, V, K, device=DEVICE, dtype=torch.float32) * 1e6

    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=state.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    n_bad = int((~torch.isfinite(o)).sum())
    assert n_bad == 0, (
        f"readout has {n_bad} non-finite (inf/NaN) entries under a large fp32 "
        f"state -- fp16 overflow; max|o|={o.float().abs().max().item():.3e}"
    )
    # Carries the true large magnitude, not a saturated/clamped value.
    assert o.float().abs().max().item() > 1e4
