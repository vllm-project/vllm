# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockE8M0Sym,
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

__all__ = ["QuarkW8A8Fp8", "QuarkW8A8Fp8PerBlock"]

logger = init_logger(__name__)


class QuarkW8A8Fp8(QuarkScheme):
    supported_activation_quant_keys: list[QuantKey | None] = [
        kFp8DynamicTensorSym,
        kFp8DynamicTokenSym,
        kFp8StaticTensorSym,
    ]
    supported_weight_quant_keys: list[QuantKey] = [
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    ]

    def __init__(
        self,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(weight_quant_key, activation_quant_key)
        self.weight_qscheme = (
            "per_channel" if weight_quant_key == kFp8StaticChannelSym else "per_tensor"
        )
        self.is_static_input_scheme = activation_quant_key == kFp8StaticTensorSym
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
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

            assert self.activation_quant_key is not None
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

        assert self.activation_quant_key is not None
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)


class QuarkW8A8Fp8PerBlock(QuarkScheme):
    supported_activation_quant_keys = [kFp8Dynamic128Sym]
    supported_weight_quant_keys = [
        kFp8Static128BlockSym,
        kFp8Static128BlockE8M0Sym,
    ]

    def __init__(
        self,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ):
        super().__init__(weight_quant_key, activation_quant_key)
        self.weight_scale_dtype = weight_quant_key.scale.dtype
        if self.weight_quant_key == kFp8Static128BlockE8M0Sym:
            # TODO: oracle needs to support kFp8Static128BlockE8M0Sym.
            self.weight_quant_key = kFp8Static128BlockSym
        self.weight_block_size = list(weight_quant_key.scale.group_shape)
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

    @classmethod
    def get_min_capability(cls) -> int:
        return QuarkW8A8Fp8.get_min_capability()

    def process_weights_after_loading(self, layer) -> None:
        # Quark exports the dequant multiplier as ``weight_scale`` (the same
        # numerical convention as DeepSeek's ``weight_scale_inv``). Kernels
        # multiply ``weight * scale``; do not invert here.
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
            self.weight_scale_dtype
            if self.weight_scale_dtype != torch.float32
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
        # Match Quark-exported checkpoints (``.weight_scale``).
        # ``Fp8LinearMethod.create_weights`` registers weight_scale_inv, which has the
        # same semantic meaning as Quark's `weight_scale`.
        layer.register_parameter("weight_scale", weight_scale)

        assert self.activation_quant_key is not None
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        logger.info_once(
            "Selected %s for QuarkW8A8Fp8PerBlock",
            type(self.fp8_linear).__name__,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)
