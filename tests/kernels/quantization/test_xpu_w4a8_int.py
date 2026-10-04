# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.lora.layers.utils import _get_lora_device
from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
    XPUW4A8IntLinearKernel,
)


def test_xpu_w4a8_removes_unpacked_weight_after_repacking() -> None:
    layer = torch.nn.Module()
    weight = torch.nn.Parameter(
        torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8),
        requires_grad=False,
    )
    layer.register_parameter("weight", weight)
    layer.register_parameter("weight_packed", weight)
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(2, 1), requires_grad=False),
    )

    kernel = object.__new__(XPUW4A8IntLinearKernel)
    kernel.w_q_name = "weight_packed"
    kernel.process_weights_after_loading(layer)

    assert _get_lora_device(layer) == layer.weight_packed.device
    assert not hasattr(layer, "weight")
    assert layer.weight_packed.shape == (2, 1)
