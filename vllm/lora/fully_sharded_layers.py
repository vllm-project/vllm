# pylint: disable=unused-argument
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              RowParallelLinearWithLoRA)
from vllm.lora.punica import bgmv, dispatch_bgmv_low_level

if TYPE_CHECKING:
    pass


def _fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        return (can_replace(*args, **kwargs)
                and kwargs['lora_config'].fully_sharded_loras)

    return dec


# these layers are based on the tensor parallelism strategy given in
# Y. Sheng et al., S-LoRA: Serving Thousands of Concurrent LoRA Adapters. 2023,
# https://arxiv.org/abs/2311.03285.


class ColumnParallelLinearWithShardedLoRA(ColumnParallelLinearWithLoRA):
    """
    Differs from ColumnParallelLinearWithLoRA by slicing LoRA A also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.lora_a_stacked.shape[2]
        start_idx = tp_rank * shard_size
        lora_a = lora_a[:, start_idx:start_idx + shard_size]
        return lora_a

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffer = torch.zeros((x.shape[0], self.lora_a_stacked.shape[2]),
                             dtype=torch.float32,
                             device=x.device)

        bgmv(buffer, x, self.lora_a_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        buffer = tensor_model_parallel_all_gather(buffer)
        bgmv(output, buffer, self.lora_b_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        # now have column partitioned output

        output = output.view(*out_orig_shape)
        return output

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


def _mcp_apply(x, bias, layer):
    """
    MergedColumnParallelLinearWithShardedLoRA and 
    QKVParallelLinearWithShardedLora share the same 
    LoRa weight application method.
    
    The main difference is the step by shard_size for lora_b which can
    vary for QKVParallelLinearWithShardedLora but is constant for 
    MergedColumnParallelLinearWithShardedLoRA.
    """
    # expecting 2 for column parallel and 3 for qkv
    n = len(layer.lora_a_stacked)
    output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape
    buffers = torch.zeros((n, x.shape[0], layer.lora_a_stacked[0].shape[2]),
                          dtype=torch.float32,
                          device=x.device)
    for idx in range(n):
        bgmv(buffers[idx], x, layer.lora_a_stacked[idx],
             layer.indices[:layer.indices_len[0]], 0, 1.0)

    buffers = tensor_model_parallel_all_gather(buffers)
    left_offset = 0
    for idx in range(n):
        shard_size = layer.lora_b_stacked[idx].shape[2]
        dispatch_bgmv_low_level(output, buffers[idx],
                                layer.lora_b_stacked[idx],
                                layer.indices[:layer.indices_len[0]], 0, 1.0,
                                left_offset, shard_size)
        left_offset += shard_size

    output = output.view(*out_orig_shape)
    # now have column partitioned and packed output
    return output


class MergedColumnParallelLinearWithShardedLoRA(
        MergedColumnParallelLinearWithLoRA):
    """
    Differs from MergedColumnParallelLinearWithLoRA by slicing the 
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        if lora_a[0] is None or lora_a[1] is None:
            return lora_a
        output_shard_size = self.lora_a_stacked[0].shape[2]
        output_start_idx = self.tp_rank * output_shard_size
        lora_a = [
            lora_a[0][:,
                      output_start_idx:output_start_idx + output_shard_size],
            lora_a[1][:, output_start_idx:output_start_idx + output_shard_size]
        ]
        return lora_a

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class MergedQKVParallelLinearWithShardedLora(MergedQKVParallelLinearWithLora):
    """
    Differs from QKVParallelLinearWithLora by slicing the 
    LoRA A's also.

    Based on S-LoRA, slicing happens along the rank dim.
    """

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        if lora_a[0] is None or lora_a[1] is None or lora_a[2] is None:
            return lora_a
        shard_size = [self.lora_a_stacked[i].shape[2] for i in range(3)]
        start_idx = [self.tp_rank * shard_size[i] for i in range(3)]
        lora_a = [
            lora_a[0][:, start_idx[0]:start_idx[0] + shard_size[0]],
            lora_a[1][:, start_idx[1]:start_idx[1] + shard_size[1]],
            lora_a[2][:, start_idx[2]:start_idx[2] + shard_size[2]]
        ]
        return lora_a

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        return _mcp_apply(x, bias, self)

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )


class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    """
    Differs from RowParallelLinearWithLoRA by slicing the 
    LoRA B's also.

    Based on S-LoRA, slicing happens along the output dim.
    This yields a combined partial sum from the row parallel base 
    layer and column partitioned output from the LoRA.
    """

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_b_stacked.shape[2]
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[:, start_idx:end_idx]
        return lora_b

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffer = torch.zeros((x.shape[0], self.lora_a_stacked.shape[2]),
                             dtype=torch.float32,
                             device=x.device)
        bgmv(buffer, x, self.lora_a_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        buffer = tensor_model_parallel_all_reduce(buffer)

        # following S-LoRA, allows the fusing of all_gather and all_reduce
        # by adding the column partitioned lora output to a slice of output
        # tensor, which is a partial sum due to row parallel. All that
        # remains is a standard all_reduce. User should be aware though that
        # the output is not the same as a normal row_parallel, it should be
        # reduced before being used
        shard_size = self.lora_b_stacked.shape[2]
        start_idx = self.tp_rank * shard_size
        dispatch_bgmv_low_level(output, buffer, self.lora_b_stacked,
                                self.indices[:self.indices_len[0]], 0, 1.0,
                                start_idx, shard_size)

        output = output.view(*out_orig_shape)
        return output

    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # specifying kwargs so they can be easily accessed in decorator
        return super().can_replace_layer(
            source_layer=source_layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
            decorate=False,
        )
