# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_tensor_strategy,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


@dataclass
class ScaledMMLinearLayerConfig:
    pass


@dataclass
class Int8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    # TODO: Change to QuantKey like FP8ScaledMMLinearLayerConfig
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool


@dataclass
class FP8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None


@dataclass
class FP8W8A8LinearLayerConfig(FP8ScaledMMLinearLayerConfig):
    pass


@dataclass
class FP8W8A16LinearLayerConfig(FP8ScaledMMLinearLayerConfig):
    pass


_FP8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_scale_ub,
]
_Int8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_zp
    torch.Tensor | None,  # azp_adj
]

_ParamsT = TypeVar("_ParamsT", _Int8ParamsT, _FP8ParamsT)
_ConfigT = TypeVar("_ConfigT", bound=ScaledMMLinearLayerConfig)


class ScaledMMLinearKernel(Generic[_ConfigT, _ParamsT], ABC):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, c: _ConfigT, layer_param_names: Sequence[str]) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer) -> _ParamsT:
        raise NotImplementedError


class FP8ScaledMMLinearKernel(
    ScaledMMLinearKernel[FP8ScaledMMLinearLayerConfig, _FP8ParamsT], ABC
):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer) -> _FP8ParamsT:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class FP8W8A8LinearKernel(FP8ScaledMMLinearKernel):
    def __init__(
        self, c: FP8W8A8LinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        act_scale_descriptor = c.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )
        self.fp8_dtype = current_platform.fp8_dtype()
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, w_s, i_s, _, _ = self._get_layer_params(layer)

        weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
            w_q,
            w_s,
            layer.logical_widths,
            i_s,
        )
        if self.config.activation_quant_key.scale.static:
            assert input_scale is not None
            input_scale = input_scale.max()

        weight = weight.t()
        # Update layer with new values.

        w_q_name, w_s_name, i_s_name, _, _ = self.layer_param_names

        replace_parameter(layer, w_q_name, weight.data)
        replace_parameter(layer, w_s_name, weight_scale.data)
        if input_scale is not None:
            replace_parameter(layer, i_s_name, input_scale)
        else:
            layer.input_scale = None

    def _get_layer_params(self, layer) -> _FP8ParamsT:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp8_dtype = self.fp8_dtype
        maybe_out_dtype = self.config.out_dtype
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        #   ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_s computed from x.
        #   If static, layer.input_scale is scalar and x_s is input_scale.
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], w.shape[1]]
        out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != fp8_dtype:
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
                x_s_ub,
            )
        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )

    @abstractmethod
    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        return None


class FP8W8A16LinearKernel(FP8ScaledMMLinearKernel):
    """
    FP8W8A16LinearKernel provides a kernel implementation for scenarios where GPUs lack native FP8 hardware support.

    This kernel leverages the Marlin kernel for efficient weight-only FP8 quantization, enabling fast inference on hardware that does not natively support FP8 operations.
    Unlike FP8W8A8LinearKernel, which is designed for platforms with full FP8 support, FP8W8A16LinearKernel uses FP8 quantized weights but processes activations in FP16, making it suitable for a broader range of GPUs.
    Intended usage: select this kernel when deploying on platforms without FP8 hardware acceleration, or when higher activation precision is desired.
    Supported platforms: GPUs without FP8 hardware support; for platforms with FP8 support, prefer FP8W8A8LinearKernel for optimal performance.
    """

    def __init__(
        self, c: FP8W8A16LinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        act_scale_descriptor = c.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )
        self.fp8_dtype = current_platform.fp8_dtype()
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Int8ScaledMMLinearKernel(
    ScaledMMLinearKernel[Int8ScaledMMLinearLayerConfig, _Int8ParamsT], ABC
):
    def _get_layer_params(self, layer) -> _Int8ParamsT:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s, None),
            getattr(layer, i_zp, None),
            getattr(layer, azp_adj, None),
        )
