# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from typing_extensions import Self

from vllm import envs
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform


@dataclass
class KernelConfig:
    pass


@dataclass
class IntKernelConfig(KernelConfig):
    # TODO: Change to QuantKey like FpKernelConfig
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool


@dataclass
class FpKernelConfig(KernelConfig):
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey
    out_dtype: torch.dtype | None


_FpParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_scale_ub,
]
_IntParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_zp
    torch.Tensor | None,  # azp_adj
]

_ParamsT = TypeVar("_ParamsT", _IntParamsT, _FpParamsT)
_ConfigT = TypeVar("_ConfigT", bound=KernelConfig)


class Kernel(Generic[_ConfigT, _ParamsT], ABC):
    @classmethod
    def try_select(
        cls, c: _ConfigT, compute_capability: int | None
    ) -> tuple[type[Self] | None, list[str]]:
        """
        Try to select a compatible kernel variant.
        """
        kernel_name = cls.get_name()
        if kernel_name in envs.VLLM_DISABLED_KERNELS:
            return None, [f" {kernel_name} is disabled by environment variable"]

        is_supported, reason = cls.is_supported(compute_capability)
        if not is_supported:
            return None, [f"{kernel_name}: {reason}"]

        can_implement, reason = cls.can_implement(c)
        if not can_implement:
            return None, [f"{kernel_name}: {reason}"]

        return cls, []

    @classmethod
    def get_name(cls) -> str:
        """
        Return the kernel name in format: linear.provider.precision.ClassName
        """
        module_path = cls.__module__
        prefix = "vllm.model_executor.kernels."
        if module_path.startswith(prefix):
            module_path = module_path[len(prefix) :]
        return f"{module_path}.{cls.__name__}"

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, c: _ConfigT, layer_param_names: Sequence[str]) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

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

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer) -> _ParamsT:
        raise NotImplementedError


class FpKernel(Kernel[FpKernelConfig, _FpParamsT], ABC):
    def __init__(self, c: FpKernelConfig, layer_param_names: Sequence[str]) -> None:
        act_scale_descriptor = c.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
        )
        self.fp8_dtype = current_platform.fp8_dtype()
        super().__init__(c, layer_param_names)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def _get_layer_params(self, layer) -> _FpParamsT:
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (
            getattr(layer, w),
            getattr(layer, w_s),
            getattr(layer, x_s, None),
            getattr(layer, x_s_ub, None),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fp8_dtype = self.fp8_dtype
        maybe_out_dtype = self.config.out_dtype
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        #   ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, layer.input_scale is None and x_s computed from x.
        #   If static, layer.input_scale is scalar and x_s is input_scale.
        # View input as 2D matrix for fp8 methods
        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], w.shape[1]]
        out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype

        # If input not quantized
        # TODO(luka) remove this path if not used anymore
        x_2d_q = x_2d
        if x.dtype != fp8_dtype:
            x_2d_q, x_s = self.quant_fp8(
                x_2d,
                x_s,
                x_s_ub,
            )
        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )

    @abstractmethod
    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        return None


class IntKernel(Kernel[IntKernelConfig, _IntParamsT], ABC):
    def _get_layer_params(self, layer) -> _IntParamsT:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s, None),
            getattr(layer, i_zp, None),
            getattr(layer, azp_adj, None),
        )
