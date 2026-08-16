# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear import (
    AiterA16Wfp4LinearKernel,
    MxFp4LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950
else:

    def on_gfx950() -> bool:
        return False


pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]


def test_fused_a16wfp4_matches_dynamic_mxfp4_qdq():
    pytest.importorskip("aiter.ops.flydsl.gemm_a16wfp4")
    torch.manual_seed(20260816)
    n = k = 256
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(
        120,
        130,
        (n, k // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    layer.register_parameter("weight", torch.nn.Parameter(weight.clone(), False))
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(scale.clone(), False),
    )
    kernel = AiterA16Wfp4LinearKernel(
        MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic)
    )
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    x = torch.randn((2, 3, k), dtype=torch.bfloat16, device="cuda")

    with patch.object(envs, "VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD", True):
        kernel.process_weights_after_loading(layer)
    actual = kernel.apply_weights(layer, x, bias)
    expected = torch.nn.functional.linear(
        quant_dequant_mxfp4(x),
        dequant_mxfp4(weight, scale, torch.bfloat16),
        bias,
    )

    assert layer._aiter_a16wfp4_prepared
    assert actual.shape == (2, 3, n)
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
