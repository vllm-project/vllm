# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Folding indexer scales must preserve the unfused projection rounding."""

import pytest
import torch

from vllm.models.kimi_k3.amd.dsa import _index_weights_kernel
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm kernel")


@pytest.mark.parametrize("rows", [1, 96, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [32, 64])
def test_index_weights_preserve_projection_rounding(rows, dtype, heads):
    torch.manual_seed(7)
    values = torch.randn(rows, heads + 16, dtype=dtype, device="cuda")[:, :heads]
    scales = torch.exp2(torch.randint(-12, 5, (rows, heads), device="cuda").float())
    head_scale, softmax_scale = heads**-0.5, 128**-0.5
    expected = (values * head_scale).float() * scales * softmax_scale
    actual = torch.empty_like(expected)
    _index_weights_kernel[(rows,)](
        values,
        scales,
        actual,
        values.stride(0),
        head_scale,
        softmax_scale,
        NUM_HEADS=heads,
        BLOCK=heads,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
