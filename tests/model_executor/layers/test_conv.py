# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer


@pytest.mark.parametrize(
    "input_shape,bias",
    [
        pytest.param((256, 3, 14, 14), False, id="packed_patches"),
        pytest.param((2, 3, 28, 28), True, id="multiple_spatial_patches"),
    ],
)
def test_conv2d_patch_embedding_correctness(default_vllm_config, input_shape, bias):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    reference = nn.Conv2d(3, 1024, 14, stride=14, bias=bias).to(
        device=device, dtype=dtype
    )
    layer = Conv2dLayer(3, 1024, 14, stride=14, bias=bias, params_dtype=dtype).to(
        device
    )
    layer.load_state_dict(reference.state_dict(), strict=True)
    assert layer.enable_linear

    x = torch.randn(input_shape, device=device, dtype=dtype)
    expected = reference(x)
    atol = rtol = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(layer.forward_native(x), expected, atol=atol, rtol=rtol)


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
def test_conv3d_patch_embedding_correctness(
    default_vllm_config, input_shape, kernel_size, out_channels, bias
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
    assert layer.enable_linear

    x = torch.randn(input_shape, device=device, dtype=dtype)
    expected = reference(x)
    atol = rtol = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(layer(x), expected, atol=atol, rtol=rtol)
