# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.utils import TestFP8Layer
from vllm.model_executor.kernels.linear import ChannelWiseTorchFP8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not ChannelWiseTorchFP8ScaledMMLinearKernel.is_supported()[0],
    reason="requires FP8 torch._scaled_mm",
)
def test_channelwise_unit_scale_is_persistent(default_vllm_config, monkeypatch):
    # torch._scaled_mm must get one persistent unit scale, across calls and weight
    # reloads, instead of a per-call allocation.
    layer = TestFP8Layer(
        weight_shape=(256, 256),
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticChannelSym,
        input_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        force_kernel=ChannelWiseTorchFP8ScaledMMLinearKernel,
    )
    assert type(layer.kernel) is ChannelWiseTorchFP8ScaledMMLinearKernel

    scales: list[torch.Tensor] = []
    scaled_mm = torch._scaled_mm

    def spy(*args, **kwargs):
        scales.extend([kwargs["scale_a"], kwargs["scale_b"]])
        return scaled_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", spy)
    x = torch.rand(32, 256, dtype=torch.bfloat16, device="cuda")
    layer(x)
    layer.kernel.process_weights_after_loading(layer)  # as a weight reload does
    layer(x)
    assert len(scales) == 4
    assert all(s is scales[0] for s in scales)
