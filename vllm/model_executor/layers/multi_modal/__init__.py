# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-Modal Layers."""

from .conv import CausalConv2dLayer, Conv2dLayer, Conv3dLayer


def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    enable_bias: bool,
    padding_mode: str,
    conv_type: str,
):
    assert in_channels and out_channels and kernel_size

    if conv_type == "conv2d":
        return Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            enable_bias=enable_bias,
            padding_mode=padding_mode,
        )
    elif conv_type == "causal_conv2d":
        return CausalConv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            enable_bias=enable_bias,
            padding_mode=padding_mode,
        )
    elif conv_type == "conv3d":
        return Conv3dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            enable_bias=enable_bias,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"Unknown conv layer type {conv_type}.")
