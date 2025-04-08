# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise)
from vllm.platforms import current_platform


@dataclass
class ScaledMMLinearLayerConfig:
    is_channelwise: bool
    is_static_input_scheme: bool
    input_symmetric: bool


class ScaledMMLinearKernel(ABC):

    @classmethod
    def is_supported(
        cls,
        compute_capability: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns true if this kernel is supported on the current platform.
        By default, a kernel is supported if the min_capability is reached
        (it still has to override the get_min_capability method).
        Kernels can also override this method for custom support checking.
        """
        return cls._current_capability_supported(compute_capability)

    @classmethod
    def get_min_capability(cls) -> int:
        """
        :return: minimum capability required for this kernel.
        Override is_supported if min_capability is irrelevant.
        """
        raise NotImplementedError(
            "Either implement get_min_capability or override is_supported")

    @classmethod
    def _current_capability_supported(
        cls,
        compute_capability: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        if compute_capability is None:
            _cc = current_platform.get_device_capability()
            if _cc is not None:
                compute_capability = _cc.major * 10 + _cc.minor

        # If the current platform uses compute_capability,
        # make sure the kernel supports the compute capability.
        if compute_capability is None:
            raise ValueError(
                f"Cannot determine if kernel {cls.__name__} is supported on "
                f"platform {current_platform} as compute capability is not "
                f"supported. Please override is_supported or remove the "
                f"kernel from the list of kernels for the platform.")

        kernel_min_capability = cls.get_min_capability()
        if (kernel_min_capability > compute_capability):
            return (False,
                    f"compute capability >={kernel_min_capability} required, "
                    f"{compute_capability} current")

        return True, None

    @classmethod
    @abstractmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def __init__(self, c: ScaledMMLinearLayerConfig, w_q_param_name: str,
                 w_s_param_name: str, i_s_param_name: str,
                 i_zp_param_name: str, azp_adj_param_name: str) -> None:
        assert self.can_implement(c)
        assert self.is_supported()
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        self.i_s_name = i_s_param_name
        self.i_zp_name = i_zp_param_name
        self.azp_adj_name = azp_adj_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _get_weight_params(
            self, layer: torch.nn.Module) -> Tuple[
                torch.Tensor,  # weight
                torch.Tensor,  # weight_scale
                Optional[torch.Tensor],  # input_scale,
                Optional[torch.Tensor],  # input_zp
                Optional[torch.Tensor],  # azp_adj
            ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.i_s_name, None),
            getattr(layer, self.i_zp_name, None),
            getattr(layer, self.azp_adj_name, None),
        )

    def replace_parameter(self, layer: torch.nn.Module, name: str,
                          param: torch.nn.Parameter):
        """
        This utility can replace a parameter with the new value.
        """

        # Call free util function
        replace_parameter(layer, name,
                          torch.nn.Parameter(param.data, requires_grad=False))

    def maybe_unfuse_weight_scale(self, layer: torch.nn.Module,
                                  weight_scale_param: torch.nn.Parameter):
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1

        if is_fused_module and not self.config.is_channelwise:
            weight_scale_param = convert_to_channelwise(
                weight_scale_param, layer.logical_widths)

        return weight_scale_param

    def fuse_asymmetric_params(
        self, input_scale_param: torch.nn.Parameter,
        input_zp_param: torch.nn.Parameter
    ) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        # reconstruct the ranges
        int8_traits = torch.iinfo(torch.int8)
        azps = input_zp_param.to(dtype=torch.int32)
        range_max = (input_scale_param * (int8_traits.max - azps)).max()
        range_min = (input_scale_param * (int8_traits.min - azps)).min()

        scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)

        # AZP loaded as int8 but used as int32
        azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)

        return scale, azp
