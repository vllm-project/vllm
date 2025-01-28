from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import ScalarType


@dataclass
class MPLinearLayerConfig:
    full_weight_shape: Tuple[int, int]  # [in, out]
    partition_weight_shape: Tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool


class MPLinearKernel(ABC):

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def __init__(self,
                 c: MPLinearLayerConfig,
                 w_q_param_name: str,
                 w_s_param_name: str,
                 w_zp_param_name: Optional[str] = None,
                 w_gidx_param_name: Optional[str] = None) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        if c.zero_points:
            assert w_zp_param_name is not None
        if c.has_g_idx:
            assert w_gidx_param_name is not None
        self.w_zp_name = w_zp_param_name
        self.w_gidx_name = w_gidx_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _transform_param(self, layer: torch.nn.Module, name: Optional[str],
                         fn: Callable) -> None:
        if name is not None and getattr(layer, name, None) is not None:

            old_param = getattr(layer, name)
            new_param = fn(old_param)
            # replace the parameter with torch.nn.Parameter for TorchDynamo
            # compatibility
            replace_parameter(
                layer, name,
                torch.nn.Parameter(new_param.data, requires_grad=False))

    def _get_weight_params(
            self, layer: torch.nn.Module) -> Tuple[
                torch.Tensor,  # w_q
                torch.Tensor,  # w_s
                Optional[torch.Tensor],  # w_zp, 
                Optional[torch.Tensor]  # w_gidx
            ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
            getattr(layer, self.w_gidx_name or "", None),
        )
