# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's merge_attn_states Triton operator.

merge_attn_states combines two partial attention results (the split-KV case):
given per-split outputs and their log-sum-exp values, it produces the softmax
weighted combination. Implements Section 2.2 of https://arxiv.org/pdf/2501.01005.

Compared against a naive PyTorch reference. Adapted from the upstream
tests/kernels/attention/test_merge_attn_states.py, which compares the Triton
kernel against a custom CUDA op (unavailable on XPU); here the PyTorch reference
is the oracle so the test is device-agnostic.

Source: vllm/v1/attention/ops/triton_merge_attn_states.py
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)

DEVICE = current_platform.device_type


def merge_attn_states_torch(
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
):
    """Naive float32 reference (Section 2.2 of arxiv 2501.01005)."""
    p_lse = prefix_lse.clone()
    s_lse = suffix_lse.clone()
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_exp = torch.exp(p_lse)
    s_exp = torch.exp(s_lse)
    out_se = p_exp + s_exp

    output_lse = torch.log(out_se) + max_lse
    p_scale = (p_exp / out_se).transpose(0, 1).unsqueeze(2)  # [TOK, HEADS, 1]
    s_scale = (s_exp / out_se).transpose(0, 1).unsqueeze(2)
    output = prefix_output.float() * p_scale + suffix_output.float() * s_scale
    return output, output_lse


def _make_inputs(num_tokens, num_heads, head_size, dtype):
    """Random inputs; ~10% of lse entries are +inf (a split that saw no keys)."""
    prefix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE
    )
    suffix_output = torch.randn(
        num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE
    )
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device=DEVICE)
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device=DEVICE)

    # A position must not be inf in both splits at once.
    mask_p = torch.rand(num_heads, num_tokens) < 0.1
    mask_s = torch.rand(num_heads, num_tokens) < 0.1
    both = mask_p & mask_s
    mask_p &= ~both
    mask_s &= ~both
    prefix_lse[mask_p] = float("inf")
    suffix_lse[mask_s] = float("inf")
    return prefix_output, prefix_lse, suffix_output, suffix_lse


NUM_TOKENS = [256, 512, 1024]
NUM_HEADS = [8, 16, 32]
HEAD_SIZES = [64, 128]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_merge_attn_states(num_tokens, num_heads, head_size, dtype):
    """merge_attn_states must match the PyTorch reference (output and lse)."""
    torch.manual_seed(0)
    prefix_output, prefix_lse, suffix_output, suffix_lse = _make_inputs(
        num_tokens, num_heads, head_size, dtype
    )

    output = torch.zeros(num_tokens, num_heads, head_size, dtype=dtype, device=DEVICE)
    output_lse = torch.zeros(num_heads, num_tokens, dtype=torch.float32, device=DEVICE)
    merge_attn_states_triton(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
    )

    ref_output, ref_lse = merge_attn_states_torch(
        prefix_output, prefix_lse, suffix_output, suffix_lse
    )

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(output.float(), ref_output, atol=1e-3, rtol=rtol)
    torch.testing.assert_close(output_lse, ref_lse, atol=1e-3, rtol=rtol)
