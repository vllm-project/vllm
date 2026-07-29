# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast
from unittest.mock import Mock

import torch

from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils import nvfp4_aot


def _marlin_kernel() -> MarlinNvFp4LinearKernel:
    return MarlinNvFp4LinearKernel.__new__(MarlinNvFp4LinearKernel)


def test_aot_dequant_delegates_to_unquantized_linear(monkeypatch):
    monkeypatch.setattr(
        nvfp4_aot.envs,
        "VLLM_NVFP4_DEQUANT_AT_LOAD",
        True,
    )
    monkeypatch.setattr(
        nvfp4_aot,
        "dequantize_to_dtype",
        Mock(return_value=torch.ones((1, 16), dtype=torch.bfloat16)),
    )

    runtime = nvfp4_aot.NvFp4LinearRuntime(_marlin_kernel())
    assert runtime.unquantized_method is not None
    runtime.unquantized_method.process_weights_after_loading = Mock()
    runtime.unquantized_method.apply = Mock(return_value=torch.tensor([1.0]))

    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(
        torch.zeros((1, 8), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    layer.weight_global_scale = torch.tensor(1.0)
    packed_weight = layer.weight

    runtime.process_weights_after_loading(layer)
    output = runtime.apply(layer, torch.tensor([1.0]))

    assert layer.weight.dtype == torch.bfloat16
    dequant_args = nvfp4_aot.dequantize_to_dtype.call_args
    assert dequant_args.args[0] is packed_weight
    assert dequant_args.args[1] is layer.weight_scale
    assert dequant_args.args[2] is layer.weight_global_scale
    assert dequant_args.args[3] == torch.bfloat16
    assert dequant_args.kwargs == {"block_size": 16, "swizzle": False}
    runtime.unquantized_method.process_weights_after_loading.assert_called_once_with(
        layer
    )
    runtime.unquantized_method.apply.assert_called_once()
    assert torch.equal(output, torch.tensor([1.0]))


def test_aot_dequant_disabled_keeps_nvfp4_kernel(monkeypatch):
    monkeypatch.setattr(
        nvfp4_aot.envs,
        "VLLM_NVFP4_DEQUANT_AT_LOAD",
        False,
    )

    kernel = _marlin_kernel()
    kernel.process_weights_after_loading = Mock()
    kernel.apply_weights = Mock(return_value=torch.tensor([2.0]))
    runtime = nvfp4_aot.NvFp4LinearRuntime(kernel)

    layer = torch.nn.Module()
    runtime.process_weights_after_loading(layer)
    output = runtime.apply(layer, torch.tensor([1.0]))

    assert not runtime.uses_unquantized_linear
    kernel.process_weights_after_loading.assert_called_once_with(layer)
    kernel.apply_weights.assert_called_once()
    assert torch.equal(output, torch.tensor([2.0]))


def test_aot_dequant_only_replaces_marlin(monkeypatch):
    monkeypatch.setattr(
        nvfp4_aot.envs,
        "VLLM_NVFP4_DEQUANT_AT_LOAD",
        True,
    )

    non_marlin_kernel = cast(NvFp4LinearKernel, object())
    runtime = nvfp4_aot.NvFp4LinearRuntime(non_marlin_kernel)

    assert not runtime.uses_unquantized_linear
