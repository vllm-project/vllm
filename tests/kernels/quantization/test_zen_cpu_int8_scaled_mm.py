# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-scale contract for ZentorchInt8ScaledMMLinearKernel."""

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.zentorch import (
    ZentorchInt8ScaledMMLinearKernel,
)
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

_LAYER_PARAMS = [
    "weight",
    "weight_scale",
    "input_scale",
    "input_zero_point",
    "azp_adj",
]


@pytest.fixture
def kernel(monkeypatch):
    """The kernel, without requiring a Zen CPU or a registered zentorch op.

    process_weights_after_loading only rewrites parameters. is_supported()
    gates construction on the platform and the op, which this contract does
    not use.
    """
    monkeypatch.setattr(
        ZentorchInt8ScaledMMLinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    return ZentorchInt8ScaledMMLinearKernel(config, layer_param_names=_LAYER_PARAMS)


@pytest.mark.parametrize("trailing_one", [False, True])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
def test_zentorch_int8_weight_scale_is_f32(kernel, trailing_one, scale_dtype):
    """Per-channel scales are stored as fp32 (N,), matching the MoE experts."""
    n, k = 32, 16
    # Small integers are exact in both fp32 and bf16.
    values = torch.arange(1, n + 1, dtype=torch.float32)
    scale = values.to(scale_dtype)
    if trailing_one:
        scale = scale.unsqueeze(-1)

    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.randint(-128, 127, (n, k), dtype=torch.int8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(scale, requires_grad=False),
    )

    kernel.process_weights_after_loading(layer)

    weight_scale = layer.weight_scale
    assert weight_scale.dtype == torch.float32
    assert tuple(weight_scale.shape) == (n,)
    assert weight_scale.is_contiguous()
    torch.testing.assert_close(weight_scale, values)
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight.shape) == (n, k)
    assert layer.weight.is_contiguous()
