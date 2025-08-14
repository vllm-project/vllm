# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Optional

import torch

from vllm.platforms import current_platform

__all__ = ["CompressedTensorsScheme"]


class CompressedTensorsScheme(ABC):
    """
    Abstract class used to describe the weight creation and forward pass 
    of different quantization schemes supported by CompressedTensors.
    """

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        raise NotImplementedError

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """
        Weight creation for the particular scheme. Inputs to this function 

        """
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]):
        """
        Run the forward pass for the particular scheme. This is where 
        scheme-specific dequant/quant steps/kernels should be applied.

        :param layer: torch.nn.Module with the registered weights and 
            other parameters relevant to the particular scheme. 
        :param x: input to the layer
        :param bias: bias parameter

        """
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError

    # TODO: make this call on init
    @classmethod
    def check_scheme_supported(cls,
                               min_capability: Optional[int] = None,
                               error: bool = True,
                               match_exact: bool = False) -> bool:
        min_capability = min_capability or cls.get_min_capability()
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            if match_exact:
                supported = capability == min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        "the current GPU. Required capability: ",
                        f"{min_capability}. Current capability: {capability}.")
            else:
                supported = capability >= min_capability
                if error and not supported:
                    raise RuntimeError(
                        "Quantization scheme is not supported for ",
                        f"the current GPU. Min capability: {min_capability}. ",
                        f"Current capability: {capability}.")
            return supported
        else:
            return False
