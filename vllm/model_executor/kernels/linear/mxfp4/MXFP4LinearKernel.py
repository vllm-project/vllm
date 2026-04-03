# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter


@dataclass
class MXFP4LinearLayerConfig:
    partition_weight_shape: tuple[int, int]
    act_type: torch.dtype | None


class MXFP4LinearKernel(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: MXFP4LinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: MXFP4LinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
    ) -> None:
        assert self.can_implement(c)
        self.config = c
        self.act_type = (
            c.act_type if c.act_type is not None else torch.get_default_dtype()
        )
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name

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

    def _transform_param(
        self, layer: torch.nn.Module, name: str | None, fn: Callable
    ) -> None:
        if name is not None and getattr(layer, name, None) is not None:
            old_param = getattr(layer, name)
            new_param = fn(old_param)
            # replace the parameter with torch.nn.Parameter for TorchDynamo
            # compatibility
            replace_parameter(
                layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
            )

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,  # w_q
        torch.Tensor,  # w_s
    ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
        )
