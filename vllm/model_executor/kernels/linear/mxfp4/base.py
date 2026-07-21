# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class MxFp4LinearLayerConfig:
    """Configuration for an MXFP4 linear layer.

    All MXFP4 layers share the same structure: packed uint8 weights (2 FP4 values per
    byte) and per-block weight scales (group size 32).
    """

    pass


class MxFp4LinearKernel(ABC):
    """Base class for MXFP4 quantized linear kernels.

    Each subclass implements a specific GEMM backend (CUTLASS, Marlin, etc).
    The kernel selection mechanism iterates over registered subclasses in
    priority order,calling ``is_supported`` and ``can_implement`` to find the best
    match for the current hardware.
    """

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Return whether this kernel can run on the current platform."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        """Return whether this kernel can handle *config*."""
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transform weights into the format required by this kernel.

        Called once after checkpoint weights have been loaded onto the
        device.  Implementations should repack / swizzle / pad weights
        and scales in-place on *layer*.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the quantized GEMM."""
        raise NotImplementedError
