# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-row lm_head padding (GB10 GEMV avoidance) must not change logits."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import vocab_parallel_embedding as vpe
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


@pytest.mark.parametrize("num_rows", [1, 2, 5])
def test_single_row_pad_matches_gemm(monkeypatch, num_rows):
    monkeypatch.setattr(vpe, "_pad_single_row_lm_head", lambda: True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(32064, 2048, device="cuda", dtype=torch.bfloat16) * 0.02,
        requires_grad=False,
    )
    x = torch.randn(num_rows, 2048, device="cuda", dtype=torch.bfloat16)

    out = vpe.UnquantizedEmbeddingMethod().apply(layer, x)

    assert out.shape == (num_rows, 32064) and out.dtype == torch.bfloat16
    if num_rows == 1:
        # Same kernel as a two-row GEMM, so the padded row is bit-identical to it.
        two = F.linear(torch.cat([x, torch.zeros_like(x)]), layer.weight)
        torch.testing.assert_close(out, two[:1], rtol=0, atol=0)
        # Against the unpadded GEMV only the summation order differs.
        torch.testing.assert_close(out, F.linear(x, layer.weight))
    else:
        torch.testing.assert_close(out, F.linear(x, layer.weight), rtol=0, atol=0)
