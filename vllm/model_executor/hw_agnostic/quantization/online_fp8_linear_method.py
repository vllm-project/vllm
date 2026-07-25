# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hw-agnostic online FP8 linear method (BF16/FP16 -> FP8 at load time)."""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.hw_agnostic.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.hw_agnostic.layers.linear import LinearMethodBase
from vllm.model_executor.hw_agnostic.quantization.quant_keys import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter


class Fp8PerTensorOnlineLinearMethod(LinearMethodBase):
    """Online tensorwise FP8 linear quantization.

    Loads fp16/bf16 weights onto meta device and quantizes them per-tensor
    during ``process_weights_after_loading``.
    """

    uses_meta_device: bool = True

    def __init__(self):
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.block_quant = False
        self.weight_quant_key = kFp8StaticTensorSym
        # Per-token dynamic activations; non-block layouts run through the
        # dequant fallback in ``apply``.
        self.activation_quant_key = kFp8DynamicTokenSym

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

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.input_scale = None
        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        replace_parameter(layer, "weight", qweight.t().data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        self.fp8_linear.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT:
            # Batch-invariant fallback: dequant to BF16, then GEMM.
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
