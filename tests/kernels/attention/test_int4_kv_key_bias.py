# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RoPE key-bias score correction for the int4 per-token-head cache."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.int4_per_token_head import (
    Int4KeyBias,
    key_bias_query_coeffs,
    reshape_and_cache_int4,
    unified_attention_int4,
)


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_key_bias_coefficients_reproduce_rotated_key_score(
    is_neox_style: bool,
) -> None:
    """The coefficient dot product equals direct RoPE for GQA heads."""
    torch.manual_seed(1)
    rotary_dim = 64
    q = torch.randn(2, 4, rotary_dim)
    bias = torch.randn(2, rotary_dim) * 100
    angles = torch.randn(7, rotary_dim // 2)
    cos, sin = angles.cos(), angles.sin()
    cache = torch.cat((cos, sin), dim=-1)
    key_bias = Int4KeyBias(bias, cache, rotary_dim, is_neox_style)

    coeffs = key_bias_query_coeffs(q, key_bias)
    actual = torch.einsum("thd,pd->thp", coeffs, cache)

    grouped_bias = bias.repeat_interleave(2, dim=0)
    if is_neox_style:
        b0, b1 = grouped_bias.chunk(2, dim=-1)
        rotated = torch.cat(
            (
                b0[None] * cos[:, None] - b1[None] * sin[:, None],
                b0[None] * sin[:, None] + b1[None] * cos[:, None],
            ),
            dim=-1,
        )
    else:
        b0, b1 = grouped_bias[..., 0::2], grouped_bias[..., 1::2]
        rotated = torch.stack(
            (
                b0[None] * cos[:, None] - b1[None] * sin[:, None],
                b0[None] * sin[:, None] + b1[None] * cos[:, None],
            ),
            dim=-1,
        ).flatten(-2)
    expected = torch.einsum("thd,phd->thp", q, rotated)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_int4_attention_restores_position_dependent_key_bias() -> None:
    """Key-bias correction restores a num_keys× weight over the uniform baseline.

    Absolute outputs wobble under int4 + RHT on a constant value row, so this
    asserts the relative effect of the correction instead of 1.0 / 0.25 targets.
    """
    device = "cuda"
    head_size = 64
    block_size = 16
    num_keys = 4
    key = torch.zeros(num_keys, 1, head_size, device=device, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    value[0] = 1
    key_cache = torch.zeros(
        1, block_size, 1, head_size // 2, device=device, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    k_scale_cache = torch.zeros(1, block_size, 1, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)
    reshape_and_cache_int4(
        key,
        value,
        key_cache,
        value_cache,
        torch.arange(num_keys, device=device, dtype=torch.long),
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
    )

    q = torch.zeros(1, 1, head_size, device=device, dtype=torch.bfloat16)
    q[..., 0] = 16
    bias = torch.zeros(1, head_size, device=device, dtype=torch.bfloat16)
    bias[..., 0] = 16
    positions = torch.arange(num_keys, device=device, dtype=torch.float32)
    cos = torch.ones(num_keys, head_size // 2, device=device)
    sin = torch.zeros_like(cos)
    cos[:, 0] = positions.cos()
    sin[:, 0] = positions.sin()
    key_bias = Int4KeyBias(bias, torch.cat((cos, sin), dim=-1), head_size, True)

    def run(key_bias_arg: Int4KeyBias | None) -> torch.Tensor:
        out = torch.empty_like(q)
        unified_attention_int4(
            q,
            key_cache,
            value_cache,
            out,
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor([num_keys], device=device, dtype=torch.int32),
            max_seqlen_k=num_keys,
            softmax_scale=head_size**-0.5,
            window_size=(-1, -1),
            block_table=torch.tensor([[0]], device=device, dtype=torch.int32),
            softcap=0,
            sinks=None,
            alibi_slopes=None,
            use_alibi_sqrt=False,
            qq_bias=None,
            output_scale=None,
            mm_prefix_range=None,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            key_bias=key_bias_arg,
        )
        return out.float()

    corrected = run(key_bias)
    uncorrected = run(None)
    # Identical zero keys → uniform 1/num_keys weights without correction.
    # Bias restoration puts all mass on key 0, so corrected/uncorrected == num_keys.
    ratio = corrected / uncorrected
    torch.testing.assert_close(
        ratio,
        torch.full_like(ratio, float(num_keys)),
        atol=1e-3,
        rtol=0,
    )
