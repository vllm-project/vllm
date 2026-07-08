# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, cast

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinFP8ScaledMMLinearKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform

__all__ = ["QuarkW8A8Fp8"]

logger = init_logger(__name__)


class QuarkW8A8Fp8(QuarkScheme):
    def __init__(
        self, weight_config: dict[str, Any], input_config: dict[str, Any] | None
    ):
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))
        self.is_static_input_scheme: bool = False
        self.input_qscheme: str | None = None
        if input_config is not None:
            self.is_static_input_scheme = not cast(bool, input_config.get("is_dynamic"))
            self.input_qscheme = cast(str, input_config.get("qscheme"))

        self.is_per_block = self.weight_qscheme == "per_block"
        self.weight_config = weight_config
        block_size = weight_config.get("block_size")
        if self.is_per_block:
            if not block_size:
                raise ValueError(
                    "Quark W8A8 FP8 per-block weight quantization requires "
                    "`block_size` in the weight quantization config."
                )
            self.weight_block_size = list(block_size)
        else:
            self.weight_block_size = list(block_size or [128, 128])
        per_token_activation = (
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"
        )
        per_channel_weight = self.weight_qscheme == "per_channel"

        if self.is_per_block:
            self.activation_quant_key = kFp8Dynamic128Sym
            self.weight_quant_key = kFp8Static128BlockSym
        else:
            self.activation_quant_key = (
                kFp8DynamicTokenSym if per_token_activation else kFp8StaticTensorSym
            )
            # A per-output-channel weight scale is one fp32 value per weight row
            # (length N). Tag it as ``GroupShape.PER_CHANNEL`` to match the
            # canonical compressed-tensors CHANNEL strategy, so kernel selection
            # (e.g. AITER's pre-shuffled FP8 GEMM) treats it uniformly.
            self.weight_quant_key = (
                kFp8StaticChannelSym if per_channel_weight else kFp8StaticTensorSym
            )
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.fp8_linear = None

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        if self.fp8_linear is None:
            raise RuntimeError("FP8 linear kernel is not initialized")

        if self.is_per_block:
            if isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel):
                self.fp8_linear.process_weights_after_loading(layer)
                return
            # Non-Marlin per-block FP8 kernels use dynamic 128-group
            # activation scaling, so there is no checkpoint-provided static
            # input scale to pass through.
            layer.input_scale = None
            self.fp8_linear.process_weights_after_loading(layer)
            return

        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.weight_qscheme == "per_tensor":
            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)
                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                max_w_scale = layer.weight_scale
                weight = layer.weight

            max_w_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=max_w_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.weight_qscheme == "per_channel":
            weight = layer.weight

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)
                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data
            if self.activation_quant_key.scale.group_shape == GroupShape.PER_TOKEN:
                weight_scale = weight_scale.view(-1, 1)
            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization scheme {self.weight_qscheme}")

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

        self.fp8_linear.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if self.is_per_block:
            self._create_per_block_weights(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
            return

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.weight_qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.weight_qscheme == "per_tensor"
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def _create_per_block_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        input_size = kwargs.get("input_size", input_size_per_partition)
        output_size = kwargs.get("output_size", sum(output_partition_sizes))
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = self.weight_block_size

        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )

        weight = create_fp8_weight_parameter(
            output_size_per_partition,
            input_size_per_partition,
            weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale_dtype = (
            torch.float8_e8m0fnu
            if self.weight_config.get("scale_type") == "float8_e8m0fnu"
            else None
        )
        weight_scale = create_fp8_scale_parameter(
            BlockQuantScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            self.weight_block_size,
            weight_loader,
            scale_dtype=scale_dtype,
        )
        # DeepSeek V4 weight mappers route checkpoint ".scale" tensors here.
        layer.register_parameter("weight_scale_inv", weight_scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        logger.info_once(
            "Selected %s for QuarkW8A8Fp8 per-block",
            type(self.fp8_linear).__name__,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fp8_linear is None:
            raise RuntimeError("FP8 linear kernel is not initialized")
        return self.fp8_linear.apply_weights(layer, x, bias)
