# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.hw_agnostic.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.hw_agnostic.layers.linear import LinearMethodBase
from vllm.model_executor.hw_agnostic.quantization.fp8_config import Fp8Config
from vllm.model_executor.hw_agnostic.quantization.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.hw_agnostic.quantization.quant_keys import (
    GroupShape,
    create_fp8_quant_key,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


class Fp8LinearMethod(LinearMethodBase):
    """Block-scaled FP8 linear method for the hw-agnostic path."""

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            self.activation_quant_key = create_fp8_quant_key(
                static=self.act_q_static,
                group_shape=GroupShape(1, self.weight_block_size[0]),
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
        else:
            self.weight_quant_key = kFp8StaticTensorSym
            # Per-tensor for static activation, per-token for dynamic — the
            # Triton scaled-MM kernel handles both layouts.
            self.activation_quant_key = (
                kFp8StaticTensorSym if self.act_q_static else kFp8DynamicTokenSym
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
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
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        if not self.block_quant:
            scale = create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
                scale_dtype=(torch.float8_e8m0fnu if self.is_scale_e8m0 else None),
            )
            # ``weight_scale_inv`` parameter name preserved for
            # checkpoint compatibility with block-quantized FP8 models.
            layer.register_parameter("weight_scale_inv", scale)

        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale = None
        if self.block_quant:
            assert not self.act_q_static
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.fp8_linear.apply_weights(layer, x, bias)

            # Per-tensor/channel batch-invariant: dequant to BF16, then
            # GEMM. The hw-agnostic kernel selector has no Cutlass option,
            # so always take the dequant branch.
            weight_fp8 = layer.weight.to(torch.bfloat16)
            weight_scale = layer.weight_scale.to(torch.bfloat16)
            if weight_scale.numel() == 1:
                weight_bf16 = weight_fp8 * weight_scale
            elif (
                weight_scale.dim() == 1 and weight_scale.shape[0] == weight_fp8.shape[0]
            ):
                weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
            else:
                weight_bf16 = weight_fp8 * weight_scale
            return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        return self.fp8_linear.apply_weights(layer, x, bias)
