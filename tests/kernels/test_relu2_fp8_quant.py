# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.vllm import OptimizationLevel
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fusion.relu2_fp8_quant import (
    Bf16ReLUSquaredStaticFp8Quant,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test requires CUDA"
)


def _assert_fp8_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == current_platform.fp8_dtype()
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


@pytest.fixture(scope="module")
def o2_relu2_fp8_ops():
    config = VllmConfig(
        optimization_level=OptimizationLevel.O2,
        compilation_config=CompilationConfig(custom_ops=["none"]),
    )
    with set_current_vllm_config(config):
        yield (
            ReLUSquaredActivation(),
            QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR),
            Bf16ReLUSquaredStaticFp8Quant(),
        )


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 16), (17, 5120), (128, 10240)])
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant(o2_relu2_fp8_ops, shape) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.1495535671710968, device="cuda", dtype=torch.float32)

    relu2, quant_fp8, fused = o2_relu2_fp8_ops
    expected, _ = quant_fp8(relu2(x), scale)

    _assert_fp8_bitwise_equal(fused(x, scale), expected)
    _assert_fp8_bitwise_equal(fused.forward_native(x, scale), expected)


@requires_cuda
@pytest.mark.parametrize(
    "scale_value",
    [
        0.00435965,
        0.007149832788854837,
        0.03125,
        0.0714285746216774,
        0.1495535671710968,
        0.1517857164144516,
        6.03571415,
    ],
)
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_all_bf16_values(
    o2_relu2_fp8_ops, scale_value
) -> None:
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16).reshape(1, -1).to("cuda")
    scale = torch.tensor(scale_value, device="cuda", dtype=torch.float32)

    relu2, quant_fp8, fused = o2_relu2_fp8_ops
    expected, _ = quant_fp8(relu2(x), scale)

    _assert_fp8_bitwise_equal(fused(x, scale), expected)


@requires_cuda
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_empty(o2_relu2_fp8_ops) -> None:
    x = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)

    _, _, fused = o2_relu2_fp8_ops
    actual = fused(x, scale)

    assert actual.shape == x.shape
    assert actual.dtype == current_platform.fp8_dtype()


@requires_cuda
@torch.inference_mode()
def test_bf16_relu2_static_fp8_quant_torch_compile(o2_relu2_fp8_ops) -> None:
    torch.manual_seed(1)
    x = torch.randn((17, 5120), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.00435965, device="cuda", dtype=torch.float32)
    relu2, quant_fp8, fused = o2_relu2_fp8_ops
    expected, _ = quant_fp8(relu2(x), scale)

    compiled_fused = torch.compile(fused, fullgraph=True)

    _assert_fp8_bitwise_equal(compiled_fused(x, scale), expected)
