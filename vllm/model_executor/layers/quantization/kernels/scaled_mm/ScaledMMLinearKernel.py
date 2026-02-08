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
class FP4ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    """Configuration for FP4 (NVFP4) scaled matrix multiplication layer."""

    group_size: int | None  # Group size for activation quantization
    is_checkpoint_fp4_serialized: (
        bool  # Whether the weights are serialized in checkpoint
    )
    out_dtype: torch.dtype | None  # Output data type


_FP4ParamsT = tuple[
    torch.Tensor,  # weight (packed uint8, 2 fp4 values per byte)
    torch.Tensor,  # weight_scale (fp8 block scales)
    torch.Tensor,  # weight_scale_2 (global weight scale, fp32)
    torch.Tensor,  # input_scale_inv (1/input_scale, fp32)
    torch.Tensor,  # alpha (input_scale * weight_scale_2, fp32)
]
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

_ParamsT = TypeVar("_ParamsT", _Int8ParamsT, _FP8ParamsT, _FP4ParamsT)
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


class FP4ScaledMMLinearKernel(
    ScaledMMLinearKernel[FP4ScaledMMLinearLayerConfig, _FP4ParamsT], ABC
):
    """Base class for FP4 scaled matrix multiplication kernels.

    FP4 kernels implement W4A8/W4A16 quantization where weights are
    quantized to 4-bit and activations are either 8-bit or 16-bit.
    """

    def __init__(
        self, c: FP4ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer) -> _FP4ParamsT:
        """Extract FP4 quantization parameters from layer."""
        w, w_s, w_s2, i_s_inv, alpha = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, w_s2),
            getattr(layer, i_s_inv),
            getattr(layer, alpha),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply FP4 quantized weights to input tensor."""
        weight, weight_scale, weight_global_scale, input_scale_inv, alpha = (
            self._get_layer_params(layer)
        )

        # Derive output shape from weight tensor dimensions
        # Weight shape is (out_features, in_features / 2) due to FP4 packing
        output_size = [*x.shape[:-1], weight.shape[0]]

        # Delegate to the scaled matrix multiplication implementation
        return self.apply_fp4_mm(
            x=x,
            weight=weight,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            input_scale_inv=input_scale_inv,
            alpha=alpha,
            bias=bias,
            output_shape=output_size,
            layer=layer,
        )

    @abstractmethod
    def apply_fp4_mm(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        input_scale_inv: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list[int],
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Backend-specific FP4 matrix multiplication.

        Subclasses must implement this to perform the actual computation.
        """
        raise NotImplementedError


class FP8ScaledMMLinearKernel(
    ScaledMMLinearKernel[FP8ScaledMMLinearLayerConfig, _FP8ParamsT], ABC
):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
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
