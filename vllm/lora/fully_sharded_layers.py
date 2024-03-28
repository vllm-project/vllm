# pylint: disable=unused-argument
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.lora.punica import bgmv, dispatch_bgmv_low_level
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear,
                                               QKVParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.utils import (
    split_tensor_along_last_dim, divide)

from vllm.lora.layers import BaseLayerWithLoRA

if TYPE_CHECKING:
    pass


class ColumnParallelLinearWithShardedLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_rank = get_tensor_model_parallel_rank()

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        # As in S-LoRA, column parallel for lora_a and lora_b

        tp_size = get_tensor_model_parallel_world_size()
        lora_a_output_size_per_partition = divide(lora_config.max_lora_rank,
                                                  tp_size)
        self.lora_a_stacked = torch.zeros(
            max_loras,
            1,
            lora_a_output_size_per_partition,
            self.base_layer.weight.shape[1],
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            max_loras,
            1,
            self.base_layer.weight.shape[0],
            lora_config.max_lora_rank,
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )

        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.output_dim = self.lora_b_stacked.shape[1]

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        self.shard_size = self.lora_a_stacked.shape[2]
        self.start_idx = self.tp_rank * self.shard_size
        lora_a = lora_a[:, self.start_idx:self.start_idx + self.shard_size]

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)

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

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: Tensor whose last dimension is `input_size`.

        Returns:
            - output
            - bias
        """
        bias = (self.base_layer.bias
                if not self.base_layer.skip_bias_add else None)

        # Matrix multiply.
        output_parallel = self.apply_weights(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @property
    def linear_weights(self):
        return self.base_layer.linear_weights


class MergedColumnParallelLinearWithShardedLoRA(
        ColumnParallelLinearWithShardedLoRA):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.n_slices = 2
        if not (len(self.base_layer.output_sizes) == self.n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # As in S-LoRA, column parallel for lora_a and lora_b
        lora_a_output_size_per_partition = divide(lora_config.max_lora_rank,
                                                  self.tp_size)

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(self.n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.base_layer.weight.shape[0] // self.n_slices,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(self.n_slices))

        self.indices: Optional[torch.Tensor] = None
        self.output_dim = self.lora_b_stacked[0].shape[2]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_b_stacked[1][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: List[torch.Tensor],
        lora_b: List[torch.Tensor],
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        self.output_shard_size = self.lora_a_stacked[0].shape[2]
        self.output_start_idx = self.tp_rank * self.output_shard_size
        if self.tp_size > 1:
            lora_a = [
                lora_a[i][:, self.output_start_idx:self.output_start_idx +
                          self.output_shard_size] for i in range(self.n_slices)
            ]

            shard_size = self.output_dim
            start_idx = self.tp_rank * shard_size
            end_idx = (self.tp_rank + 1) * shard_size
            lora_b = lora_b[0][:,
                               start_idx:end_idx], lora_b[1][:,
                                                             start_idx:end_idx]

        if lora_a[0] is not None:
            self.lora_a_stacked[0][
                index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                    lora_a[0].T, non_blocking=True)
            self.lora_b_stacked[0][
                index, 0, :lora_b[0].shape[1], :lora_b[0].shape[0]].copy_(
                    lora_b[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][
                index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                    lora_a[1].T, non_blocking=True)
            self.lora_b_stacked[1][
                index, 0, :lora_b[1].shape[1], :lora_b[1].shape[0]].copy_(
                    lora_b[1].T, non_blocking=True)

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)

        x = x.view(-1, x.shape[-1])
        output, out_orig_shape = output.view(-1,
                                             output.shape[-1]), output.shape
        buffers = torch.zeros(
            (self.n_slices, x.shape[0], self.lora_a_stacked[0].shape[2]),
            dtype=torch.float32,
            device=x.device)
        for slice_idx in range(self.n_slices):
            bgmv(buffers[slice_idx], x, self.lora_a_stacked[slice_idx],
                 self.indices[:self.indices_len[0]], 0, 1.0)

        buffers = tensor_model_parallel_all_gather(buffers)
        shard_size_b = self.lora_b_stacked[0].shape[2]
        left_offset = 0
        for slice_idx in range(self.n_slices):
            dispatch_bgmv_low_level(output, buffers[slice_idx],
                                    self.lora_b_stacked[slice_idx],
                                    self.indices[:self.indices_len[0]], 0, 1.0,
                                    left_offset, shard_size_b)
            left_offset += shard_size_b

        output = output.view(*out_orig_shape)
        # now have column partitioned and packed output
        return output


class QKVParallelLinearWithShardedLora(ColumnParallelLinearWithShardedLoRA):
    """ColumnParallelLinear layer that is composed of 3 sublayers (slices)
    packed together in qkv proj fashion
    (q_proj + k_proj + v_proj -> qkv_proj).

    This means we have 3 LoRAs, each applied to one slice of the layer.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        # As in S-LoRA, column parallel for lora_a and lora_b
        lora_a_output_size_per_partition = divide(lora_config.max_lora_rank,
                                                  self.tp_size)

        # q, k, v
        self.lora_a_stacked = (
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
        )
        self.lora_b_stacked = (
            torch.zeros(
                max_loras,
                1,
                self.q_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
        )

        self.output_slices = (self.q_proj_shard_size, self.kv_proj_shard_size,
                              self.kv_proj_shard_size)
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[1][index] = 0
        self.lora_a_stacked[2][index] = 0
        self.lora_b_stacked[2][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            if lora_b[0] is not None:
                lora_b_q = lora_b[0][:, self.q_proj_shard_size *
                                     self.q_shard_id:self.q_proj_shard_size *
                                     (self.q_shard_id + 1)]
                self.lora_b_stacked[0][
                    index, 0, :lora_b_q.shape[1], :lora_b_q.shape[0]].copy_(
                        lora_b_q.T, non_blocking=True)
            if lora_b[1] is not None:
                lora_b_k = lora_b[1][:, self.kv_proj_shard_size *
                                     self.kv_shard_id:self.kv_proj_shard_size *
                                     (self.kv_shard_id + 1)]
                self.lora_b_stacked[1][
                    index, 0, :lora_b_k.shape[1], :lora_b_k.shape[0]].copy_(
                        lora_b_k.T, non_blocking=True)
            if lora_b[2] is not None:
                lora_b_v = lora_b[2][:, self.kv_proj_shard_size *
                                     self.kv_shard_id:self.kv_proj_shard_size *
                                     (self.kv_shard_id + 1)]
                self.lora_b_stacked[2][
                    index, 0, :lora_b_v.shape[1], :lora_b_v.shape[0]].copy_(
                        lora_b_v.T, non_blocking=True)
        else:
            if lora_b[0] is not None:
                self.lora_b_stacked[0][
                    index, 0, :lora_b[0].shape[1], :lora_b[0].shape[0]].copy_(
                        lora_b[0].T, non_blocking=True)
            if lora_b[1] is not None:
                self.lora_b_stacked[1][
                    index, 0, :lora_b[1].shape[1], :lora_b[1].shape[0]].copy_(
                        lora_b[1].T, non_blocking=True)
            if lora_b[2] is not None:
                self.lora_b_stacked[2][
                    index, 0, :lora_b[2].shape[1], :lora_b[2].shape[0]].copy_(
                        lora_b[2].T, non_blocking=True)

        for lora_ in lora_a:
            if lora_ is None:
                return

        self.shard_size = [self.lora_a_stacked[i].shape[2] for i in range(3)]
        self.start_idx = [self.tp_rank * self.shard_size[i] for i in range(3)]
        lora_a = [
            lora_a[i][:,
                      self.start_idx[i]:self.start_idx[i] + self.shard_size[i]]
            for i in range(3)
        ]

        self.lora_a_stacked[0][
            index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                lora_a[0].T, non_blocking=True)
        self.lora_a_stacked[1][
            index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                lora_a[1].T, non_blocking=True)
        self.lora_a_stacked[2][
            index, 0, :lora_a[2].shape[1], :lora_a[2].shape[0]].copy_(
                lora_a[2].T, non_blocking=True)

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)

        buffers = torch.zeros((3, x.shape[0], self.lora_a_stacked[0].shape[2]),
                              dtype=torch.float32,
                              device=x.device)
        for proj_idx in range(3):
            bgmv(buffers[proj_idx], x, self.lora_a_stacked[proj_idx],
                 self.indices[:self.indices_len[0]], 0, 1.0)

        buffers = tensor_model_parallel_all_gather(buffers)
        left_offset = 0
        for proj_idx in range(3):
            shard_size = self.lora_b_stacked[proj_idx].shape[2]
            dispatch_bgmv_low_level(output, buffers[proj_idx],
                                    self.lora_b_stacked[proj_idx],
                                    self.indices[:self.indices_len[0]], 0, 1.0,
                                    left_offset, shard_size)
            left_offset += shard_size

        # now have column partitioned and packed output
        return output


class RowParallelLinearWithShardedLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_rank = get_tensor_model_parallel_rank()

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.base_layer.weight.shape[1],
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )

        # As in S-LoRA, column parallel for lora_b.
        # Needs an all_reduce beforehand
        tp_size = get_tensor_model_parallel_world_size()
        lora_b_output_size_per_partition = divide(
            self.base_layer.weight.shape[0], tp_size)

        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_b_output_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )

        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        if self.base_layer.tp_size > 1:
            # lora_a row tp
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.base_layer.weight.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_a = lora_a[start_idx:end_idx, :]

            # lora_b col tp
            shard_size = self.lora_b_stacked.shape[2]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_b = lora_b[:, start_idx:end_idx]

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x)

        buffer = torch.zeros((x.shape[0], self.lora_a_stacked.shape[2]),
                             dtype=torch.float32,
                             device=x.device)
        bgmv(buffer, x, self.lora_a_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        buffer = tensor_model_parallel_all_reduce(buffer)

        # following S-LoRA, allows the fusing of all_gather and all_reduce
        # by adding the column partitioned lora output to a slice of output
        # tensor. All that remains is a standard all_reduce. User should
        # be aware though that the output is not the same as a normal
        # row_parallel, it should be reduced before being used col parallel
        shard_size = self.lora_b_stacked.shape[2]
        start_idx = self.tp_rank * shard_size
        dispatch_bgmv_low_level(output, buffer, self.lora_b_stacked,
                                self.indices[:self.indices_len[0]], 0, 1.0,
                                start_idx, shard_size)

        return output

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply_weights(input_parallel)
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
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight
