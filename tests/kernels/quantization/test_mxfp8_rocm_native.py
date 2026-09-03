# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
    _mxfp8_dot_scaled_linear,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_e4m3_quantize_torch,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform


def test_mxfp8_torch_quantization_zero_blocks() -> None:
    x = torch.zeros((2, MXFP8_BLOCK_SIZE), dtype=torch.bfloat16)

    quantized, scales = _mxfp8_e4m3_quantize_torch(x)

    torch.testing.assert_close(quantized.float(), torch.zeros_like(x).float())
    assert torch.equal(scales, torch.full_like(scales, 127))


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_mx(),
    reason="Native MXFP8 linear requires ROCm CDNA4",
)
def test_mxfp8_native_linear_zero_input_is_finite() -> None:
    device = torch.device("cuda")
    m, n, k = 64, 128, 128
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
    weight_source = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    weight, weight_scale = mxfp8_e4m3_quantize(weight_source)

    output = _mxfp8_dot_scaled_linear(x, weight, weight_scale)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))
