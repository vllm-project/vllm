# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for masked utility kernels. 
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.masked_kernels import (
    invoke_masked_silu_and_mul)


def ref_silu_mul(x, out, valid_tokens_array):

    valid_tokens_array = valid_tokens_array.to("cpu")
    batch_size = x.size(0)
    for b in range(batch_size):
        # num valid tokens
        n = valid_tokens_array[b]
        torch.ops._C.silu_and_mul(out[b, :n, :], x[b, :n, :])


## Tests for masked silu_and_mul ####
@pytest.mark.parametrize("batch_size", [1, 13, 26, 32, 64])
@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
def test_masked_silu_mul(batch_size: int, num_tokens: int, hidden_size: int,
                         dtype: torch.dtype):

    input = torch.randn((batch_size, num_tokens, hidden_size),
                        device="cuda",
                        dtype=dtype)

    out = torch.empty((batch_size, num_tokens, hidden_size // 2),
                      device="cuda",
                      dtype=dtype)

    ref_out = torch.empty_like(out)
    ref_out.copy_(out)

    # valid num_tokens per batch
    valid_num_tokens = torch.randint(low=0,
                                     high=batch_size + 1,
                                     size=(batch_size, ),
                                     device="cuda")

    # reference
    ref_silu_mul(input, ref_out, valid_num_tokens)

    # impl
    invoke_masked_silu_and_mul(out, input, valid_num_tokens)

    torch.testing.assert_close(ref_out, out, atol=1e-3, rtol=1e-2)
