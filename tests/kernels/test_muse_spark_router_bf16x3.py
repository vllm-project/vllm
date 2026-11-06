# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm fp32-exact router GEMM over three BF16 weight planes."""

import pytest
import torch

from vllm.models.muse_spark.amd.ops.router_bf16x3 import (
    router_bf16x3_gemm,
    split_fp32_to_bf16x3,
    supports_router_bf16x3,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("num_tokens", [1, 33, 300, 1100, 5000])
def test_router_bf16x3_gemm_is_fp32_exact(num_tokens):
    """Three bf16 planes carry the fp32 router weight exactly, at every size."""
    torch.manual_seed(num_tokens)
    w = torch.randn(256, 8192, device="cuda") * 0.02
    x = torch.randn(num_tokens, 8192, device="cuda", dtype=torch.bfloat16)
    assert supports_router_bf16x3(256, 8192)
    planes = split_fp32_to_bf16x3(w)
    assert torch.equal(planes.float().sum(0), w)

    ref = x.double() @ w.double().t()
    out = router_bf16x3_gemm(x, planes)
    assert out.dtype == torch.float32 and out.shape == (num_tokens, 256)
    assert ((out.double() - ref).abs().max() / ref.abs().max()).item() < 2e-6
