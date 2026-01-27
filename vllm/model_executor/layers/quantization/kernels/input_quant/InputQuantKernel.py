# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
)
from vllm.platforms import current_platform

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()

__all__ = [
    "_FP8_DTYPE",
    "_FP8_MIN",
    "_FP8_MAX",
    "InputQuantKernel",
    "InputQuantConfig",
]


@dataclass
class InputQuantConfig:
    static: bool
    group_shape: GroupShape
    column_major_scales: bool = False
    use_ue8m0: bool = False
    num_token_padding: int | None = None
    tma_aligned_scales: bool = False


_ConfigT = TypeVar("_ConfigT", bound=InputQuantConfig)


class InputQuantKernel(ABC, Generic[_ConfigT]):
    def __init__(self, config: _ConfigT):
        self.config = config
        self.group_shape = config.group_shape
        self.is_static_quant = config.static
        self.is_group_quant = config.group_shape.is_per_group()
        self.is_column_major_scales = config.column_major_scales
        self.use_ue8m0 = config.use_ue8m0
        self.num_token_padding = config.num_token_padding
        if self.is_group_quant:
            self.group_size = self.group_shape.col
        else:
            if not self.is_static_quant:
                assert self.group_shape in (
                    GroupShape.PER_TOKEN,
                    GroupShape.PER_TENSOR,
                ), (
                    "Only per-token or per-tensor scales are supported for dynamic "
                    "non-group quantization."
                )

    @classmethod
    @abstractmethod
    def is_supported(
        cls,
    ) -> tuple[bool, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def ordered_fallback_kernels(cls) -> list[type["InputQuantKernel[_ConfigT]"]]:
        raise NotImplementedError

    @abstractmethod
    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def apply(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_shape = self.config.group_shape

        if group_shape.is_per_group():
            return self.apply_group_quant(x, scale, scale_ub, kwargs=kwargs)

        # for some kernels per-tensor and per-token quantization
        # share the same implementation
        # since they differ only in scale dimensionality, not computation logic.
        if group_shape.is_per_tensor() or group_shape.is_per_token():
            return self.apply_per_token_per_tensor_quant(
                x, scale, scale_ub, kwargs=kwargs
            )

        # TODO: Per-channel quantization not yet supported.
        # Currently no kernel implements this quantization granularity.

        raise ValueError(
            f"Currently input quant kernel, {self}, does not support {group_shape}"
        )
