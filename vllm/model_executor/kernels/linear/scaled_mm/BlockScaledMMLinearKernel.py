# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from typing_extensions import Self

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.utils import replace_parameter

from ..base import (
    FP8Params,
    MMLinearKernel,
)
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig


@dataclass
class FP8BlockParams(FP8Params):
    weight_scale_inv: torch.Tensor | None
    weight_scale: torch.Tensor | None

    WEIGHT_SCALE_INV: ClassVar[str] = "weight_scale_inv"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale_inv=getattr(layer, cls.WEIGHT_SCALE_INV, None),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE, None),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


class Fp8BlockScaledMMLinearKernel(
    MMLinearKernel[FP8ScaledMMLinearLayerConfig, FP8BlockParams], ABC
):
    # Set to False in subclasses that accept BF16 input directly (e.g. FlashInfer)
    # and therefore do not need the input quantization step in apply_weights.
    apply_input_quant: ClassVar[bool] = True

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.weight_group_shape = config.weight_quant_key.scale.group_shape
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
        )
        self.use_triton = False

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        act_quant_key = config.activation_quant_key
        if act_quant_key.scale.static:
            return (
                False,
                "Only dynamic per token group activation quantization is supported.",
            )

        return True, None

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8BlockParams:
        return FP8BlockParams.from_layer(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        params = self._get_layer_params(layer)
        # Fp8LinearMethod registered weight scale
        # buffer as weight_scale_inv unlike compressed tensors.
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )
        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
        replace_parameter(layer, scale_attr_name, new_weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        maybe_out_dtype = self.config.out_dtype
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]
        out_dtype = input_2d.dtype if maybe_out_dtype is None else maybe_out_dtype

        if self.apply_input_quant:
            q_input, input_scale = self.quant_fp8(
                input_2d, input_scale, scale_up, use_triton=self.use_triton
            )
        else:
            q_input = input_2d
            # Provide a concrete placeholder so apply_block_scaled_mm args are
            # always Tensors. Subclasses with apply_input_quant=False must not
            # use As in apply_block_scaled_mm.
            input_scale = (
                input_scale if input_scale is not None else input_2d.new_ones(1)
            )

        output = self.apply_block_scaled_mm(
            A=q_input,
            B=weight,
            As=input_scale,
            Bs=weight_scale,
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=out_dtype).view(*output_shape)

    @abstractmethod
    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8BlockScaledDynamicMMLinearKernel(Fp8BlockScaledMMLinearKernel, ABC):
    """Dynamic FP8 block-scaled kernel that dispatches via torch.cond at runtime.

    Extends Fp8BlockScaledMMLinearKernel to inherit apply_weights and overrides
    apply_block_scaled_mm to dispatch between two sub-kernels using torch.cond,
    enabling torch.compile compatibility.

    Subclasses must define:
        base_type:     The primary kernel class (used when predicate is True).
        fallback_type: The fallback kernel class (used when predicate is False).

    By default both branches receive FP8 input (apply_input_quant=True inherited).
    Override apply_input_quant=False when the base kernel requires BF16 input
    (e.g. FlashInfer), and override apply_block_scaled_mm to handle quantization
    inside the fallback branch closure.
    """

    base_type: ClassVar[type[Fp8BlockScaledMMLinearKernel]]
    fallback_type: ClassVar[type[Fp8BlockScaledMMLinearKernel]]

    def __init__(self, config: "FP8ScaledMMLinearLayerConfig") -> None:
        super().__init__(config)
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
        if is_base_supported and is_fallback_supported:
            return True, None
        if not is_base_supported and not is_fallback_supported:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )
        if not is_base_supported:
            return False, f"base is not supported due to {reason_1}"
        return False, f"fallback is not supported due to {reason_2}"

    @classmethod
    def can_implement(
        cls, config: "FP8ScaledMMLinearLayerConfig"
    ) -> tuple[bool, str | None]:
        can_implement_base, reason_1 = cls.base_type.can_implement(config)
        can_implement_fallback, reason_2 = cls.fallback_type.can_implement(config)
        if can_implement_base and can_implement_fallback:
            return True, None
        if not can_implement_base and not can_implement_fallback:
            return (
                False,
                f"base cannot implement due to {reason_1}; "
                f"fallback cannot implement due to {reason_2}",
            )
        if not can_implement_base:
            return False, f"base cannot implement due to {reason_1}"
        return False, f"fallback cannot implement due to {reason_2}"

    @abstractmethod
    def predicate(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar boolean Tensor selecting the branch for torch.cond.

        Returns True to dispatch to base, False to dispatch to fallback.
        Must return a scalar boolean Tensor (not a Python bool) for
        torch.compile compatibility.
        """
        raise NotImplementedError

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # torch.cond registers both branches in the computation graph so
        # torch.compile can capture dynamic dispatch without breaking tracing.
        # All operands must be concrete Tensors — non-tensor state is accessed
        # via self.config or captured by the branch method closures.
        return torch.cond(
            self.predicate(A, B, As, Bs),
            self.base.apply_block_scaled_mm,
            self.fallback.apply_block_scaled_mm,
            [A, B, As, Bs],
        )
