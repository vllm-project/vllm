# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


class ScaledMMLinearQuantStrategy(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    BLOCK = "block"


QUANT_STRATEGY_MAP = {
    QuantizationStrategy.TENSOR: ScaledMMLinearQuantStrategy.TENSOR,
    QuantizationStrategy.CHANNEL: ScaledMMLinearQuantStrategy.CHANNEL,
}


@dataclass
class ScaledMMLinearLayerConfig:
    is_static_input_scheme: bool


@dataclass
class Int8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    is_channelwise: bool
    input_symmetric: bool


@dataclass
class FP8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    weight_quant_strategy: ScaledMMLinearQuantStrategy
    activation_group_shape: GroupShape
    out_dtype: torch.dtype | None


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
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, c: _ConfigT, layer_param_names: Sequence[str]) -> None:
        assert self.can_implement(c)
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
        self.quant_fp8 = QuantFP8(
            static=c.is_static_input_scheme,
            group_shape=c.activation_group_shape,
            num_token_padding=self.get_ouput_padding(),
        )
        super().__init__(c, layer_param_names)

    @abstractmethod
    def get_ouput_padding(self) -> int | None:
        raise NotImplementedError

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer) -> _FP8ParamsT:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s),
            getattr(layer, x_s_ub),
        )


class Int8ScaledMMLinearKernel(
    ScaledMMLinearKernel[Int8ScaledMMLinearLayerConfig, _Int8ParamsT], ABC
):
    def _get_layer_params(self, layer) -> _Int8ParamsT:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s),
            getattr(layer, i_zp),
            getattr(layer, azp_adj),
        )
