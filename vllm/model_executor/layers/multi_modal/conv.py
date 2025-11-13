# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv Layer Class."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
import torch.nn.parameter as Parameter

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.utils import set_weight_attrs


@CustomOp.register("conv2d")
class Conv2dLayer(CustomOp):
    """Conv2D layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        enable_bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding: int = padding
        self.dilation = dilation
        self.groups = groups
        self.enable_bias = enable_bias
        self.padding_mode = padding_mode

        self.enable_linear: bool = False
        if _enable_linear(kernel_size[0], stride, padding):
            self.enable_linear = True

        _create_conv_weights(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            groups=self.groups,
            enable_bias=self.enable_bias,
            enable_linear=self.enable_linear,
            weight_loader=self.weight_loader,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if self.enable_linear:
            loaded_weight = _convert_conv_to_linear_weight(loaded_weight)
        param.data.copy_(loaded_weight)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            x = F.linear(x, self.weight, self.bias)
        else:
            x = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"bias={self.bias}, "
        return s


@CustomOp.register("causal_conv2d")
class CausalConv2dLayer(CustomOp):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would
    have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be
    set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        enable_bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.enable_bias = enable_bias
        self.padding_mode = padding_mode

        if padding is not None:
            raise ValueError(
                "Argument padding should be set to None for CausalConv2dLayer."
            )
        self._left_padding: int = kernel_size[0] - 1
        self._right_padding: int = stride - 1
        self.padding: int = 0

        self.enable_linear: bool = False
        if _enable_linear(kernel_size[0], stride, padding):
            self.enable_linear = True

        _create_conv_weights(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            groups=self.groups,
            enable_bias=self.enable_bias,
            enable_linear=self.enable_linear,
            weight_loader=self.weight_loader,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if self.enable_linear:
            loaded_weight = _convert_conv_to_linear_weight(loaded_weight)
        param.data.copy_(loaded_weight)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            x = F.linear(x, self.weight, self.bias)
        else:
            x = F.pad(x, pad=(self._left_padding, self._right_padding, 0, 0))
            x = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"bias={self.bias}, "
        return s


@CustomOp.register("conv3d")
class Conv3dLayer(CustomOp):
    """Conv3D layer with linear weight."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        enable_bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        assert isinstance(kernel_size, tuple) and len(kernel_size) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding: int = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.use_linear = False

        self.enable_linear: bool = False
        if _enable_linear(kernel_size[0], stride, padding):
            self.enable_linear = True

        _create_conv_weights(
            self,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            groups=self.groups,
            enable_bias=self.enable_bias,
            enable_linear=self.enable_linear,
            weight_loader=self.weight_loader,
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if self.enable_linear:
            loaded_weight = _convert_conv_to_linear_weight(loaded_weight)
        param.data.copy_(loaded_weight)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            x = F.linear(x, self.weight, self.bias)
        else:
            x = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def extra_repr(self) -> str:
        s = f"in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"bias={self.bias}, "
        return s


def _enable_linear(
    kernel_size: int,
    stride: int,
    padding: int,
) -> bool:
    assert isinstance(kernel_size, int) and isinstance(stride, int)
    return kernel_size == stride and padding == 0


def _create_conv_weights(
    layer: torch.nn.Module,
    in_channels: int,
    out_channels: int,
    kernel_size: tuple,
    groups: int,
    enable_bias: bool,
    enable_linear: bool,
    weight_loader: Callable,
) -> None:
    if enable_linear:
        # Use linear computation for better performance.
        weight = Parameter(
            torch.empty((out_channels, in_channels * math.prod(kernel_size)))
        )
    else:
        # Use normal Conv2D computation.
        weight = Parameter(
            torch.empty((out_channels, in_channels // groups, *kernel_size))
        )
    layer.register_parameter("weight", weight)
    set_weight_attrs(weight, {"weight_loader": weight_loader})

    if enable_bias:
        bias = Parameter(torch.empty(out_channels))
        layer.register_parameter("bias", bias)
        set_weight_attrs(bias, {"weight_loader": weight_loader})
    else:
        layer.register_parameter("bias", None)


# Due to a performance regression with Conv3D in PyTorch2.9, we reshape
# Conv3D weights to Linear weights for better performance.
# See: https://github.com/vllm-project/vllm/issues/27406
# and https://github.com/pytorch/pytorch/issues/166122
# FIXME(Isotr0py): Revert the PR introduces this workaround
# (https://github.com/vllm-project/vllm/pull/27418),
# once the performance issue is resolved in PyTorch.
def _convert_conv_to_linear_weight(conv_weight: torch.Tensor) -> torch.Tensor:
    """
    Reshape Conv2D or Conv3D weight to Linear weight.
    Only work when kernel_size==stride.
    """
    out_channels = conv_weight.shape[0]
    linear_weight = conv_weight.reshape(out_channels, -1)
    return linear_weight
