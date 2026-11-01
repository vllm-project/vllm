# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 emulation after load-time dequantization runs the platform GEMM. On
ROCm the cases below reach wvSplitK (1-5 tokens), LLMM1 (the N=8 shape at one
token) and hipBLASLt (the rest)."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform

# (N, K) of the DeepSeek-V4.1-Flash MXFP8 linears at TP=4.
V41_SHAPES = [
    (1792, 5120),
    (8192, 1280),
    (2048, 4096),
    (5120, 2048),
    (1152, 5120),
    (5120, 576),
]
# Not a model shape: N <= 8 fails the wvSplitK predicate (N == 1 or N > 8), so
# on ROCm one token takes LLMM1 instead.
LLMM1_SHAPE = (8, 1024)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs a GPU")
@pytest.mark.parametrize("n,k", V41_SHAPES + [LLMM1_SHAPE])
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 7, 32, 256])
@torch.inference_mode()
def test_mxfp8_emulation_matches_bf16_linear(monkeypatch, n, k, num_tokens):
    """Matches F.linear on the dequantized weight at the token counts that
    select different ROCm GEMM kernels."""
    monkeypatch.setenv("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "1")
    torch.manual_seed(num_tokens)
    weight, scale = _mxfp8_e4m3_quantize_torch(
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    )
    expected_weight = dequant_mxfp8_to_bf16(weight, scale).float()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    kernel = EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)
    expected = x.float() @ expected_weight.T
    rel = (out.float() - expected).norm() / expected.norm()
    assert out.dtype == x.dtype and rel < 5e-3, rel
