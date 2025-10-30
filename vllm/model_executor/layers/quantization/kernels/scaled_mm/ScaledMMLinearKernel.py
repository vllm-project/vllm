# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


class ScaledMMLinearQuantStrategy(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    BLOCK = "block"

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


@dataclass
class ScaledMMLinearLayerConfig:
    pass

@dataclass
class Int8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    is_channelwise: bool
    is_static_input_scheme: bool
    input_symmetric: bool

@dataclass
class FP8ScaledMMLinearLayerConfig(ScaledMMLinearLayerConfig):
    weight_quant_strategy: ScaledMMLinearQuantStrategy
    activation_group_shape: GroupShape
    out_dtype: torch.dtype


ConfigT = TypeVar('ConfigT', bound=ScaledMMLinearLayerConfig)


class ScaledMMLinearKernel(Generic[ConfigT], ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self, c: ConfigT, layer_mapping_function: Callable
    ) -> None:
        assert self.can_implement(c)
        self.config = c
        self.layer_mapping_function = layer_mapping_function

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