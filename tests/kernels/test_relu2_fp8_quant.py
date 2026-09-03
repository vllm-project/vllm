# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.vllm import OptimizationLevel
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

requires_sm90 = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="This test requires SM90 or newer",
)


class _StaticFp8Linear(torch.nn.Module):
    def __init__(self, input_scale: torch.Tensor) -> None:
        super().__init__()
        self.input_quant_key = kFp8StaticTensorSym
        self.input_scale = input_scale


def _assert_fp8_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == current_platform.fp8_dtype()
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def _assert_quantized_activation(
    result: QuantizedActivation, x: torch.Tensor, scale: torch.Tensor
) -> None:
    assert result.scale is scale
    assert result.orig_dtype == x.dtype
    assert result.orig_shape == x.shape
    assert result.quant_key == kFp8StaticTensorSym


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
        )


@requires_sm90
@pytest.mark.parametrize("shape", [(1, 1), (1, 16), (17, 5120), (128, 10240)])
@torch.inference_mode()
def test_relu2_static_fp8_quant(o2_relu2_fp8_ops, shape) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.1495535671710968, device="cuda", dtype=torch.float32)

    relu2, quant_fp8 = o2_relu2_fp8_ops
    expected, _ = quant_fp8(relu2(x), scale)
    result = maybe_fused_act_quant(relu2, x, _StaticFp8Linear(scale))

    assert isinstance(result, QuantizedActivation)
    _assert_quantized_activation(result, x, scale)
    _assert_fp8_bitwise_equal(result.data, expected)


@requires_sm90
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
def test_relu2_static_fp8_quant_all_bf16_values(o2_relu2_fp8_ops, scale_value) -> None:
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16).reshape(1, -1).to("cuda")
    scale = torch.tensor(scale_value, device="cuda", dtype=torch.float32)

    relu2, quant_fp8 = o2_relu2_fp8_ops
    expected, _ = quant_fp8(relu2(x), scale)
    result = maybe_fused_act_quant(relu2, x, _StaticFp8Linear(scale))

    assert isinstance(result, QuantizedActivation)
    _assert_quantized_activation(result, x, scale)
    _assert_fp8_bitwise_equal(result.data, expected)


@requires_sm90
@torch.inference_mode()
def test_relu2_static_fp8_quant_empty(o2_relu2_fp8_ops) -> None:
    x = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)

    relu2, _ = o2_relu2_fp8_ops
    result = maybe_fused_act_quant(relu2, x, _StaticFp8Linear(scale))

    assert isinstance(result, QuantizedActivation)
    _assert_quantized_activation(result, x, scale)
    assert result.data.shape == x.shape
    assert result.data.dtype == current_platform.fp8_dtype()


@requires_sm90
@pytest.mark.parametrize("unsupported", ["dtype", "layout", "scale"])
@torch.inference_mode()
def test_relu2_static_fp8_quant_falls_back(o2_relu2_fp8_ops, unsupported: str) -> None:
    x = torch.randn((17, 32), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)
    if unsupported == "dtype":
        x = x.to(torch.float16)
    elif unsupported == "layout":
        x = x.T
    else:
        scale = scale.to(torch.float16)

    relu2, _ = o2_relu2_fp8_ops
    expected = relu2(x)
    result = maybe_fused_act_quant(relu2, x, _StaticFp8Linear(scale))

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, QuantizedActivation)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@requires_sm90
@torch.inference_mode()
def test_relu2_static_fp8_quant_torch_compile_dynamic(
    o2_relu2_fp8_ops,
) -> None:
    torch.manual_seed(1)
    scale = torch.tensor(0.00435965, device="cuda", dtype=torch.float32)
    relu2, quant_fp8 = o2_relu2_fp8_ops
    linear = _StaticFp8Linear(scale)

    def fused(x: torch.Tensor) -> torch.Tensor:
        result = maybe_fused_act_quant(relu2, x, linear)
        assert isinstance(result, QuantizedActivation)
        return result.data

    compile_count = 0

    def counting_inductor_backend(gm, example_inputs):
        nonlocal compile_count
        compile_count += 1
        return torch._inductor.compile(gm, example_inputs)

    torch._dynamo.reset()
    try:
        x = torch.randn((17, 5120), device="cuda", dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        compiled_fused = torch.compile(
            fused, backend=counting_inductor_backend, fullgraph=True
        )

        for candidate in (
            x,
            torch.randn((31, 5120), device="cuda", dtype=torch.bfloat16),
            torch.randn((63, 5120), device="cuda", dtype=torch.bfloat16),
        ):
            expected, _ = quant_fp8(relu2(candidate), scale)
            _assert_fp8_bitwise_equal(compiled_fused(candidate), expected)

        assert compile_count == 1
    finally:
        torch._dynamo.reset()
