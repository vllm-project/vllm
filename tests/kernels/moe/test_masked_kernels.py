# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for masked utility kernels. 
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.masked_kernels import (
    invoke_masked_silu_and_mul, masked_per_token_group_quant_fp8)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform


def ref_silu_mul(x, out, valid_tokens_array):

    valid_tokens_array = valid_tokens_array.to("cpu")
    batch_size = x.size(0)
    for b in range(batch_size):
        # num valid tokens
        n = valid_tokens_array[b]
        if n == 0:
            continue
        torch.ops._C.silu_and_mul(out[b, :n, :], x[b, :n, :])


def ref_per_token_group_quant(
        x: torch.Tensor, x_q: torch.Tensor, valid_tokens_array: torch.Tensor,
        group_size: int,
        column_major_scales: bool) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape == x_q.shape

    # make scales tensor
    B, NUM_TOKENS, HIDDEN_SIZE = x.shape
    x_q_s = torch.empty((B, NUM_TOKENS, HIDDEN_SIZE // group_size),
                        device="cuda",
                        dtype=torch.float32)

    valid_tokens_array = valid_tokens_array.to("cpu")
    batch_size = x.size(0)
    for b in range(batch_size):
        # num valid tokens
        n = valid_tokens_array[b]
        if n == 0:
            continue
        x_slice = x[b, :n, :]
        xq_slice, xqs_slice = per_token_group_quant_fp8(
            x_slice, group_size, column_major_scales=column_major_scales)
        x_q[b, :n, :].copy_(xq_slice)
        x_q_s[b, :n, :].copy_(xqs_slice)

    return x_q, x_q_s


BATCH_SIZES = [1, 13, 26, 32]
NUM_TOKENS = [7, 37, 64, 4096]

## Tests for masked per_token_group_quant_fp8  ####

HIDDEN_SIZES = [128, 256, 384, 512, 1024]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("column_major_scales", [True])
def test_masked_per_token_group_quant_fp8(batch_size: int, num_tokens: int,
                                          hidden_size: int, dtype: torch.dtype,
                                          column_major_scales: bool):

    DEEPGEMM_BLOCK_SIZE = 128

    input = torch.randn(
        (batch_size, num_tokens, hidden_size), device="cuda",
        dtype=dtype) / 10.0

    out_q = torch.randn((batch_size, num_tokens, hidden_size), device="cuda")
    out_q = out_q.to(dtype=current_platform.fp8_dtype())

    ref_out_q = torch.empty_like(out_q)
    ref_out_q.copy_(out_q)

    # valid num_tokens per batch
    valid_num_tokens = torch.randint(low=0,
                                     high=num_tokens + 1,
                                     size=(batch_size, ),
                                     device="cuda").to(torch.int32)

    # Reference
    ref_out_q, ref_out_scales = ref_per_token_group_quant(
        x=input,
        x_q=ref_out_q,
        valid_tokens_array=valid_num_tokens,
        group_size=DEEPGEMM_BLOCK_SIZE,
        column_major_scales=column_major_scales)

    # Impl
    out_q, out_scales = masked_per_token_group_quant_fp8(
        x=input,
        x_q=out_q,
        valid_tokens_array=valid_num_tokens,
        group_size=DEEPGEMM_BLOCK_SIZE,
        column_major_scales=column_major_scales)

    torch.testing.assert_close(ref_out_q, out_q)

    valid_num_tokens_cpu = valid_num_tokens.to(device="cpu")
    for b in range(valid_num_tokens_cpu.size(0)):
        n = valid_num_tokens_cpu[b]
        torch.testing.assert_close(ref_out_scales[b, :n, :],
                                   out_scales[b, :n, :])


## Tests for masked silu_and_mul ####

HIDDEN_SIZES = [124, 1024, 2176, 2816]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_masked_silu_mul(batch_size: int, num_tokens: int, hidden_size: int,
                         dtype: torch.dtype):

    input = torch.randn(
        (batch_size, num_tokens, hidden_size), device="cuda",
        dtype=dtype) / 10.0

    out = torch.empty((batch_size, num_tokens, hidden_size // 2),
                      device="cuda",
                      dtype=dtype)

    ref_out = torch.empty_like(out)
    ref_out.copy_(out)

    # valid num_tokens per batch
    valid_num_tokens = torch.randint(low=0,
                                     high=num_tokens + 1,
                                     size=(batch_size, ),
                                     device="cuda").to(torch.int32)

    # reference
    ref_silu_mul(input, ref_out, valid_num_tokens)

    # impl
    invoke_masked_silu_and_mul(out, input, valid_num_tokens)

    torch.testing.assert_close(ref_out, out)
