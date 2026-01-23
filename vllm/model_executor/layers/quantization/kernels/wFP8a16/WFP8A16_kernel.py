# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)


@dataclass
class Wfp8A16LinearLayerConfig:
    pass


@dataclass
class FP8WoQLinearLayerConfig(Wfp8A16LinearLayerConfig):
    weight_quant_key: QuantKey
    input_dtype: torch.dtype
    is_block_quant: bool = False


_FP8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor,  # activation
]


class FP8WoQLinearKernel(FP8WoQLinearLayerConfig, _FP8ParamsT, ABC):
    """
    FP8 WoQ kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.
    """

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: FP8WoQLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: FP8WoQLinearLayerConfig,
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c

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
