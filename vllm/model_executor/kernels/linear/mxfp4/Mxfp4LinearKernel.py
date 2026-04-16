# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class Mxfp4LinearLayerConfig:
    """Configuration for an MXFP4 linear layer.

    All MXFP4 layers share the same structure: FP4-E2M1 weights with
    uint8 (E8M0) per-block scales at block size 32.
    """

    pass


class Mxfp4LinearKernel(ABC):
    def __init__(self, c: Mxfp4LinearLayerConfig) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: Mxfp4LinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

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
