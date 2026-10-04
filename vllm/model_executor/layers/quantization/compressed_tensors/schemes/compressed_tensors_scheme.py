# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

from vllm.model_executor.utils import held_tensors

__all__ = ["CompressedTensorsScheme"]


class CompressedTensorsScheme(ABC):
    """Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by CompressedTensors.
    """

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Get minimum device capability."""
        raise NotImplementedError()

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """Weight creation for the particular scheme. Inputs to this function."""
        raise NotImplementedError()

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ):
        """Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.

        Args:
            layer: torch.nn.Module with the registered weights and
                other parameters relevant to the particular scheme.
            x: input to the layer
            bias: bias parameter

        """
        raise NotImplementedError()

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        """Device tensors held by this scheme's kernels."""
        return held_tensors(self)
