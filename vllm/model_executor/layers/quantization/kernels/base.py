# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
from typing_extensions import Self


@dataclass
class MMLinearLayerConfig: ...


@dataclass
class Params:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None

    # Attribute names on the layer
    WEIGHT: ClassVar[str] = "weight"
    WEIGHT_SCALE: ClassVar[str] = "weight_scale"
    INPUT_SCALE: ClassVar[str] = "input_scale"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
        )


@dataclass
class FP8Params(Params):
    """FP8 layer parameters with typed fields"""

    input_scale_ub: torch.Tensor | None

    INPUT_SCALE_UB: ClassVar[str] = "input_scale_ub"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "FP8Params":
        """Extract parameters from layer"""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


@dataclass
class Int8Params(Params):
    """Int8 layer parameters with typed fields"""

    input_zero_point: torch.Tensor | None
    azp_adj: torch.Tensor | None

    INPUT_ZERO_POINT: ClassVar[str] = "input_zero_point"
    AZP_ADJ: ClassVar[str] = "azp_adj"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "Int8Params":
        """Extract parameters from layer"""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_zero_point=getattr(layer, cls.INPUT_ZERO_POINT, None),
            azp_adj=getattr(layer, cls.AZP_ADJ, None),
        )


_ParamsT = TypeVar("_ParamsT", bound=Params)
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)


class MMLinearKernel(ABC, Generic[_ConfigT, _ParamsT]):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(self, config: _ConfigT) -> None:
        self.config = config

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> _ParamsT:
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        return None

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


_BaseKernelT = TypeVar("_BaseKernelT", bound=MMLinearKernel)
_FallbackKernelT = TypeVar("_FallbackKernelT", bound=MMLinearKernel)


class DynamicMMLinearKernel(
    MMLinearKernel, Generic[_ConfigT, _ParamsT, _BaseKernelT, _FallbackKernelT]
):
    base_type: type[_BaseKernelT]
    fallback_type: type[_FallbackKernelT]

    def __init__(self, config: _ConfigT):
        self.base = self.base_type(config)
        self.fallback = self.fallback_type(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        is_base_supported, reason_1 = cls.base_type.is_supported(compute_capability)
        is_fallback_supported, reason_2 = cls.fallback_type.is_supported(
            compute_capability
        )
        if is_fallback_supported and is_fallback_supported:
            return True, None

        # Both unsupported: include both reasons
        if not is_base_supported and not is_fallback_supported:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )

        # Exactly one unsupported: report that one
        if not is_base_supported:
            return False, f"base is not supported due to {reason_1}"

        # Here: base is supported but fallback is not
        return False, f"fallback is not supported due to {reason_2}"

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> _ParamsT:
        get_fallback_params = kwargs.get("get_fallback_params", False)
        if get_fallback_params:
            return self.fallback._get_layer_params(layer)
        return self.base._get_layer_params(layer)

    @classmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        can_implmenet_base, reason_1 = cls.base_type.can_implement(config)
        can_implmenet_fallback, reason_2 = cls.fallback_type.can_implement(config)
        if can_implmenet_base and can_implmenet_fallback:
            return True, None

        if not can_implmenet_base and not can_implmenet_fallback:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )

        if not can_implmenet_base:
            return False, f"base is not supported due to {reason_1}"

        return False, f"fallback is not supported due to {reason_2}"

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def predicate(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PyTorch's torch.compile cannot handle input-dependent control flow in standard
        # Python conditionals. torch.cond() explicitly registers both code paths in the
        # computation graph,
        # allowing torch.compile to capture both branches.
        # without torch.cond, the predicate condition
        # won't be able to be captured by torch
        # compile

        return torch.cond(
            self.predicate(layer, x, bias),
            self.base.apply_weights,
            self.fallback.apply_weights,
            (layer, x, bias),
        )
