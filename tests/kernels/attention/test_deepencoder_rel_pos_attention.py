# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.deepencoder import deepencoder_rel_pos_attention
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="The DeepEncoder rel-pos attention kernel requires CUDA or ROCm.",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("height,width", [(8, 8), (14, 14), (16, 12)])
def test_deepencoder_rel_pos_attention(dtype: torch.dtype, height: int, width: int):
    torch.manual_seed(0)
    batch, heads, head_dim = 2, 3, 64
    num_tokens = height * width
    shape = (batch, heads, num_tokens, head_dim)
    q, k, v = (torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3))
    rel_h = torch.randn(batch, heads, num_tokens, height, dtype=dtype, device="cuda")
    rel_w = torch.randn(batch, heads, num_tokens, width, dtype=dtype, device="cuda")
    scale = head_dim**-0.5

    out = deepencoder_rel_pos_attention(q, k, v, rel_h, rel_w, width, scale)

    # The non-Triton path of DeepEncoder's attention: key (h, w) gets the bias
    # rel_h[..., h] + rel_w[..., w].
    attn_bias = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(
        batch, heads, num_tokens, num_tokens
    )
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=scale)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
