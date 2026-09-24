# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.conv import Conv3dLayer


@pytest.fixture
def enabled_conv3d():
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["+conv3d"]))
    with set_current_vllm_config(config):
        yield


@pytest.mark.parametrize(
    "input_shape,kernel_size,out_channels,bias",
    [
        pytest.param((4, 3, 2, 14, 14), (2, 14, 14), 1280, False, id="minimax_m3"),
        pytest.param((4, 3, 2, 14, 14), (2, 14, 14), 1152, False, id="qwen2_vl"),
        pytest.param((4, 3, 2, 14, 14), (2, 14, 14), 1152, True, id="qwen3_vl"),
        pytest.param((4, 3, 1, 14, 14), (1, 14, 14), 1536, True, id="glm_vl"),
        pytest.param((1, 3, 4, 28, 28), (2, 14, 14), 32, True, id="multiple_patches"),
    ],
)
def test_conv3d_patch_embedding_matches_torch(
    enabled_conv3d, input_shape, kernel_size, out_channels, bias
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    reference = nn.Conv3d(
        input_shape[1], out_channels, kernel_size, stride=kernel_size, bias=bias
    ).to(device=device, dtype=dtype)
    layer = Conv3dLayer(
        input_shape[1],
        out_channels,
        kernel_size,
        stride=kernel_size,
        bias=bias,
        params_dtype=dtype,
    ).to(device)
    layer.load_state_dict(reference.state_dict(), strict=True)
    assert layer.enabled() and layer.enable_linear

    x = torch.randn(input_shape, device=device, dtype=dtype)
    expected = reference(x)
    atol = rtol = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(layer(x), expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(layer.forward_native(x), expected, atol=atol, rtol=rtol)


def test_conv3d_overlapping_grouped_convolution_matches_torch(enabled_conv3d):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = nn.Conv3d(
        4, 8, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), groups=2
    ).to(device)
    layer = Conv3dLayer(
        4,
        8,
        (2, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
        groups=2,
    ).to(device)
    layer.load_state_dict(reference.state_dict(), strict=True)
    assert layer.enabled() and not layer.enable_linear

    x = torch.randn((2, 4, 3, 7, 7), device=device)
    torch.testing.assert_close(layer(x), reference(x))
