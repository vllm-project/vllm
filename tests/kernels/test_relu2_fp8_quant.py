# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fusion.relu2_fp8_quant import (
    Bf16ReLUSquaredStaticFp8Quant,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("shape", [(1, 16), (17, 5120), (128, 10240)])
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant(default_vllm_config, shape) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.03125], device="cuda", dtype=torch.float32)

    activated = ReLUSquaredActivation()(x)
    expected, _ = ops.scaled_fp8_quant(activated, scale)
    actual = Bf16ReLUSquaredStaticFp8Quant()(x, scale)

    assert actual.dtype == current_platform.fp8_dtype()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
