# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


@register_weight_loader_v2_supported_method
class _ModelOptNvfp4LinearSource(QuantizeMethodBase):
    """Load serialized NVFP4 linears without requiring an NVFP4 GEMM backend."""

    def __init__(self, source_method):
        self.source_method = source_method

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        from vllm.model_executor.layers.quantization.modelopt import ACT, WEIGHT, Shapes

        source = self.source_method
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        shapes = Shapes(output_partition_sizes, input_size_per_partition, params_dtype)
        source.wkey.create_weights(layer, WEIGHT, source.ctx, shapes, weight_loader)
        if source.akey is not None:
            source.akey.create_weights(layer, ACT, source.ctx, shapes, weight_loader)
        source.fmt.extra_weights(layer, shapes, source.ctx, weight_loader)

    def dequantize_weight(self, layer):
        from vllm.model_executor.layers.quantization.utils import (
            nvfp4_emulation_utils,
        )

        input_size = layer.weight.shape[1] * 2
        weight = torch.empty(
            (layer.weight.shape[0], input_size),
            dtype=self.source_method.out_dtype,
            device=layer.weight.device,
        )
        chunk_rows = max(1, 16 * 1024 * 1024 // input_size)
        offset = 0
        for partition, width in enumerate(layer.logical_widths):
            for start in range(offset, offset + width, chunk_rows):
                rows = slice(start, min(start + chunk_rows, offset + width))
                weight[rows] = nvfp4_emulation_utils.dequantize_to_dtype(
                    layer.weight[rows].contiguous(),
                    layer.weight_scale[rows].contiguous(),
                    layer.weight_scale_2[partition],
                    weight.dtype,
                    block_size=self.source_method.ctx.group_size,
                    swizzle=False,
                )
            offset += width
        return weight

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("Requantization sources do not execute GEMMs.")


class OnlineRequantizationMixin:
    """Shared lifecycle for converting serialized weights before quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requantization_source: QuantizeMethodBase | None = None
        self.requantization_source_parameters: dict[str, torch.nn.Parameter] = {}

    def set_requantization_source(self, source_method: QuantizeMethodBase) -> None:
        """Configure serialized-weight conversion before online quantization."""
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptLinearMethod,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kNvfp4Static,
        )

        if (
            isinstance(source_method, ModelOptLinearMethod)
            and source_method.spec.weight == kNvfp4Static
        ):
            source_method = _ModelOptNvfp4LinearSource(source_method)
        self.requantization_source = source_method
        self.uses_meta_device = False

    def create_requantization_source_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ) -> bool:
        """Delegate weight creation to the source and track its parameters."""
        if self.requantization_source is None:
            return False

        existing_parameter_names = set(layer._parameters)
        self.requantization_source.create_weights(
            layer, *weight_args, **extra_weight_attrs
        )
        self.requantization_source_parameters = {
            name: parameter
            for name, parameter in layer._parameters.items()
            if name not in existing_parameter_names and parameter is not None
        }
        return True

    def release_requantization_source_weights(self, layer: torch.nn.Module) -> None:
        """Release serialized parameters after successful requantization."""
        for name, source_parameter in self.requantization_source_parameters.items():
            if layer._parameters.get(name) is source_parameter:
                delattr(layer, name)
        self.requantization_source_parameters.clear()
