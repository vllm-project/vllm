# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.platforms import current_platform

from .base_linear import BaseLinearLayerWithLoRA
from .utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace


class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__(base_layer)

        # reset input_size
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size
        # There is only one LoRA layer.
        self.n_slices = 1

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        shard_size = self.input_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_a = lora_a[:, start_idx:end_idx]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
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
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
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
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is RowParallelLinear


# The following layer is based on the tensor parallelism strategy given in
# Y. Sheng et al., S-LoRA: Serving Thousands of Concurrent LoRA Adapters. 2023,
# https://arxiv.org/abs/2311.03285.


class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    """
    Differs from RowParallelLinearWithLoRA by slicing the
    LoRA B's also.

    Based on S-LoRA, slicing happens along the output dim.
    This yields a combined partial sum from the row parallel base
    layer and column partitioned output from the LoRA.
    """

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_b_stacked[0].shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[start_idx:end_idx, :]
        return lora_b

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape
        buffer = torch.zeros(
            (self.n_slices, x.shape[0], self.lora_a_stacked[0].shape[2]),
            dtype=torch.float32,
            device=x.device,
        )

        shrunk_buffer: torch.Tensor | None = self.punica_wrapper.add_shrink(
            buffer, x, self.lora_a_stacked, 1.0
        )
        if not current_platform.can_update_inplace():
            buffer = shrunk_buffer
        if self.tp_size > 1:
            buffer = tensor_model_parallel_all_reduce(buffer)

        # following S-LoRA, allows the fusing of all_gather and all_reduce
        # by adding the column partitioned lora output to a slice of output
        # tensor, which is a partial sum due to row parallel. All that
        # remains is a standard all_reduce. User should be aware though that
        # the output is not the same as a normal row_parallel, it should be
        # reduced before being used
        # NOTE offset are based on the rank.
        shard_size = self.lora_b_stacked[0].shape[2]
        offset_start = self.tp_rank * shard_size
        lora_output: torch.Tensor | None = self.punica_wrapper.add_expand(
            output,
            buffer,
            self.lora_b_stacked,
            self.output_slices,
            offset_start=offset_start,
            add_input=True,
        )

        if not current_platform.can_update_inplace():
            output = lora_output

        output = output.view(*out_orig_shape)
        return output

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )
