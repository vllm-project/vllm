# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import Module

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.online.requantization import (
    OnlineRequantizationMixin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter


def _is_tp_sharded(layer: Module, *, reduces_output_dim: bool = True) -> bool:
    """Whether the weight is sharded along a dim its ``amax`` reduces over.

    Row parallel shards the input dim, column parallel the output dim.
    Per-output-channel scales only reduce the input dim, so they are already
    unsharded-equivalent on column-parallel layers.
    """
    if layer.tp_size == 1:
        return False
    is_row_parallel = layer.input_size != layer.input_size_per_partition
    is_column_parallel = layer.output_size != layer.output_size_per_partition
    return is_row_parallel or (reduces_output_dim and is_column_parallel)


class OnlineLinearBase(OnlineRequantizationMixin, LinearMethodBase):
    """Shared base for online linear methods. Loads fp16/bf16 checkpoint
    weights onto meta device and materializes them just-in-time."""

    uses_meta_device: bool = True
    activation_quant_key: QuantKey | None

    def __init__(self):
        super().__init__()
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

    def get_weight_for_quantization(self, layer: torch.nn.Module) -> torch.Tensor:
        if self.requantization_source is None:
            return layer.weight
        weight = self.requantization_source.dequantize_weight(layer)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("Linear requantization requires a single weight tensor.")
        return weight

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
        if self.create_requantization_source_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            return

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
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)
