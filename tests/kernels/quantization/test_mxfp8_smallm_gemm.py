# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the Triton small-M MXFP8 GEMM (mxfp8_smallm_gemm).

The kernel computes, per 32-wide block,
    out[m, n] = sum_k in_fp8[m, k] * w_fp8[n, k] * 2^(in_sf + w_sf - 254)
which is exactly a dequantize-then-matmul. We check it against that
reference over the small-M (M < 128) range it is used for.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton MXFP8 small-M GEMM requires a CUDA-like device",
)


@pytest.mark.parametrize("M", [1, 16, 17, 32, 64, 127])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("K", [128, 256])
def test_mxfp8_smallm_gemm_matches_reference(M, N, K):
    from vllm.model_executor.kernels.linear.mxfp8.triton_smallm import (
        mxfp8_smallm_gemm,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    device = current_platform.device_type
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    x_fp8, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    w_fp8, w_scale = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)

    out = mxfp8_smallm_gemm(x_fp8, x_scale, w_fp8, w_scale)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16

    x_deq = dequant_mxfp8_to_bf16(x_fp8, x_scale)
    w_deq = dequant_mxfp8_to_bf16(w_fp8, w_scale)
    ref = x_deq.float() @ w_deq.float().t()

    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
