# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Set

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearLayerConfig, choose_scaled_mm_linear_kernel)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

logger = init_logger(__name__)


class CompressedTensorsW8A8Int8(CompressedTensorsScheme):
    _kernel_backends_being_used: Set[str] = set()

    def __init__(self, strategy: str, is_static_input_scheme: bool,
                 input_symmetric: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        layer.logical_widths = output_partition_sizes

        scaled_mm_linear_kernel_config = ScaledMMLinearLayerConfig(
            is_channelwise=(self.strategy == QuantizationStrategy.CHANNEL),
            is_static_input_scheme=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric)

        kernel_type = choose_scaled_mm_linear_kernel(
            scaled_mm_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsW8A8Int8",
                        kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(data=torch.empty(
                1, dtype=torch.float32),
                                            weight_loader=weight_loader)
            layer.register_parameter("input_scale", input_scale)

            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8),
                    weight_loader=weight_loader)
                layer.register_parameter("input_zero_point", input_zero_point)

        self.kernel = kernel_type(c=scaled_mm_linear_kernel_config,
                                  w_q_param_name="weight",
                                  w_s_param_name="weight_scale",
                                  i_s_param_name="input_scale",
                                  i_zp_param_name="input_zero_point",
                                  azp_adj_param_name="azp_adj")

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
