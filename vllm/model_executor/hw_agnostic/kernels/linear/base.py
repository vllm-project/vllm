# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from typing_extensions import Self

from vllm.model_executor.hw_agnostic.quantization.quant_keys import QuantKey


@dataclass
class MMLinearLayerConfig: ...


@dataclass
class Params:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None

    WEIGHT: ClassVar[str] = "weight"
    WEIGHT_SCALE: ClassVar[str] = "weight_scale"
    INPUT_SCALE: ClassVar[str] = "input_scale"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
        )


@dataclass
class FP8Params(Params):
    input_scale_ub: torch.Tensor | None

    INPUT_SCALE_UB: ClassVar[str] = "input_scale_ub"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "FP8Params":
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


_ParamsT = TypeVar("_ParamsT", bound=Params)
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)


class MMLinearKernel(ABC, Generic[_ConfigT, _ParamsT]):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, config: _ConfigT) -> None:
        self.config = config

    def input_quant_key(self) -> QuantKey | None:
        return None

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_layer_params(self, layer: torch.nn.Module, **kwargs: Any) -> _ParamsT:
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        return None

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
