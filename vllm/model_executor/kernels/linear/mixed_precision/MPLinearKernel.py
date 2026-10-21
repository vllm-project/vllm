# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import ScalarType


def _shallow_copy(param: torch.Tensor) -> torch.Tensor:
    """A new object of ``param``'s class sharing its data, with its own copy of
    the attributes (e.g. dim tags) for a transform to retag."""
    copy = torch.Tensor._make_subclass(type(param), param.data, param.requires_grad)
    copy.__dict__.update(param.__dict__)
    return copy


@dataclass
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    out_type: torch.dtype | None = None


class MPLinearKernel(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: str | None = None,
    ) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        if c.zero_points:
            assert w_zp_param_name is not None
        self.w_zp_name: str | None = w_zp_param_name

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

    def _release_unused_zp(self, layer: torch.nn.Module) -> None:
        name = self.w_zp_name
        if name is not None and getattr(layer, name, None) is not None:
            # Loaded but unused: keep the param, free its data.
            replace_parameter(layer, name, getattr(layer, name).new_empty(0))
        self.w_zp_name = None

    def _transform_param(
        self, layer: torch.nn.Module, name: str | None, fn: Callable
    ) -> None:
        if name is not None and getattr(layer, name, None) is not None:
            # `fn` may retag its input; the kept loader param keeps its own dims.
            new_param = fn(_shallow_copy(getattr(layer, name)))
            replace_parameter(
                layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
            )

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,  # w_q
        torch.Tensor,  # w_s
        torch.Tensor | None,  # w_zp
    ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
        )
