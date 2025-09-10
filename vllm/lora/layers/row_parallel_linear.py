# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import  Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_reduce)

# yapf: disable
from vllm.model_executor.layers.linear import RowParallelLinear

from .base_linear import BaseLinearLayerWithLoRA

from .utils import _not_fully_sharded_can_replace

class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

        self.tp_size = get_tensor_model_parallel_world_size()
        # reset input_size
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size

        self.tp_rank = get_tensor_model_parallel_rank()
        # There is only one LoRA layer.
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:

        shard_size = self.input_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def slice_bias(self, bias: torch.Tensor) -> torch.Tensor:
        return bias

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias

        if not self.base_layer.return_bias:
            return output

        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear


