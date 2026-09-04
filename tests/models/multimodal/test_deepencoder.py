# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.deepencoder import (
    _flex_attention_with_decomposed_rel_pos,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_attention_matches_dense_decomposed_bias(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, heads, height, width, dim = 1, 2, 4, 4, 16
    tokens = height * width
    q = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rel_h = torch.randn(batch, heads, tokens, height, 1, device="cuda", dtype=dtype)
    rel_w = torch.randn(batch, heads, tokens, 1, width, device="cuda", dtype=dtype)
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=(rel_h + rel_w).flatten(-2)
    )

    actual = _flex_attention_with_decomposed_rel_pos(q, k, v, rel_h, rel_w, width)

    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
