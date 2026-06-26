# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the opt-in small-M BF16 fallback in the FlashInfer
CUTLASS MXFP8 linear kernel (VLLM_MXFP8_BF16_FALLBACK_SMALL_M).

At M < 128 the kernel skips FlashInfer's mm_mxfp8 (which pads M up to a
128-row tile) and instead does a plain BF16 matmul against a dequantized
copy of the weight. This checks the fallback output matches the reference
dequantize-then-matmul it is meant to reproduce.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FlashInfer CUTLASS MXFP8 kernel requires sm_100 (Blackwell)",
)


class _Layer:
    """Minimal stand-in for the linear layer the kernel mutates."""


@pytest.mark.parametrize("M", [1, 16, 17, 32, 64, 127])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("K", [256])
def test_bf16_fallback_matches_dequant_matmul(monkeypatch, M, N, K):
    monkeypatch.setenv("VLLM_MXFP8_BF16_FALLBACK_SMALL_M", "1")

    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    w_fp8, w_scale = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)

    layer = _Layer()
    layer.weight = w_fp8
    layer.weight_scale = w_scale

    kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    assert hasattr(layer, "weight_bf16"), "fallback weight was not cached"

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = kernel.apply_weights(layer, x)

    ref = x @ dequant_mxfp8_to_bf16(w_fp8, w_scale).t()
    assert out.shape == (M, N)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
