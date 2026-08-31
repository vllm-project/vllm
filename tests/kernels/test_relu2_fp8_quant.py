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

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test requires CUDA"
)


def _assert_fp8_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == current_platform.fp8_dtype()
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 16), (17, 5120), (128, 10240)])
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant(default_vllm_config, shape) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.1495535671710968, device="cuda", dtype=torch.float32)

    activated = ReLUSquaredActivation()(x)
    expected, _ = ops.scaled_fp8_quant(activated, scale)
    fused = Bf16ReLUSquaredStaticFp8Quant()

    _assert_fp8_bitwise_equal(fused(x, scale), expected)
    _assert_fp8_bitwise_equal(fused.forward_native(x, scale), expected)


@requires_cuda
@pytest.mark.parametrize(
    "scale_value", [0.00435965, 0.03125, 0.1495535671710968, 6.03571415]
)
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_all_bf16_values(
    default_vllm_config, scale_value
) -> None:
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16).reshape(1, -1).to("cuda")
    scale = torch.tensor(scale_value, device="cuda", dtype=torch.float32)

    activated = ReLUSquaredActivation()(x)
    expected, _ = ops.scaled_fp8_quant(activated, scale)
    fused = Bf16ReLUSquaredStaticFp8Quant()

    _assert_fp8_bitwise_equal(fused(x, scale), expected)
    _assert_fp8_bitwise_equal(fused.forward_native(x, scale), expected)


@requires_cuda
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_empty(default_vllm_config) -> None:
    x = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)

    actual = Bf16ReLUSquaredStaticFp8Quant()(x, scale)

    assert actual.shape == x.shape
    assert actual.dtype == current_platform.fp8_dtype()


@requires_cuda
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_torch_compile(default_vllm_config) -> None:
    torch.manual_seed(1)
    x = torch.randn((17, 5120), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.00435965, device="cuda", dtype=torch.float32)
    activated = ReLUSquaredActivation()(x)
    expected, _ = ops.scaled_fp8_quant(activated, scale)

    fused = torch.compile(Bf16ReLUSquaredStaticFp8Quant(), fullgraph=True)

    _assert_fp8_bitwise_equal(fused(x, scale), expected)
