# pylint: disable=unused-argument
import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.utils import (
    split_tensor_along_last_dim)

if TYPE_CHECKING:
    pass


def _apply_lora(
    x: torch.Tensor,
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
):
    """Applies lora to each input.

    This method applies all loras to each input. It uses the
    indices vector to determine which lora yields the
    correct output. An index of -1 means no lora should be
    applied. This method adds the final lora results to the
    output.

    Input shapes:
        x:               (batch_size, hidden_dim)
        lora_a_stacked:  (num_loras, lora_rank, hidden_dim)
        lora_b_stacked:  (num_loras, output_dim, lora_rank)
        indices:         (batch_size)
        output:          (batch_size, output_dim)
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    add_lora(output, x, lora_a_stacked, lora_b_stacked, indices, 0, 1.0)
    return output.view_as(org_output)


def _apply_lora_packed_nslice(
    x: torch.Tensor,
    lora_a_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lora_b_stacked: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    output: torch.Tensor,
    output_slices: Tuple[int, ...],
):
    """Applies lora to each input.

    This method applies all loras to each input. It uses the
    indices vector to determine which lora yields the
    correct output. An index of -1 means no lora should be
    applied. This method adds the final lora results to the
    output.

    This method is used for layers that are composed of multiple sublayers
    (slices) packed together.

    Input shapes:
        x:                 (batch_size, hidden_dim)
        lora_a_stacked:    3 element tuple of (num_loras, lora_rank, hidden_dim)
        lora_b_stacked:    3 element tuple of (num_loras, output_dim, lora_rank)
        indices:           (batch_size)
        output:            (batch_size, q_slice_size + 2*kv_slice_size)
        output_slices:     n-1 element tuple of (slice_size...),
                           where n is number of slices
    """
    org_output = output
    x = x.view(-1, x.shape[-1])
    output = output.view(-1, output.shape[-1])
    indices = indices.view(-1)
    offset_left = 0
    for slice_idx in range(len(output_slices)):
        add_lora_slice(output, x, lora_a_stacked[slice_idx],
                       lora_b_stacked[slice_idx], indices, 0, 1.0, offset_left,
                       output_slices[slice_idx])
        offset_left += output_slices[slice_idx]
    return output.view_as(org_output)


@dataclass
class LoRAMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)


class BaseLayerWithLoRA(nn.Module):

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        """Sets the mapping indices."""
        ...

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        lora_vocab_start_idx = self.base_layer.org_vocab_size
        weights_idx = None
        if self.base_layer.vocab_end_index > lora_vocab_start_idx:
            # We can start adding lora weights
            weights_idx = max(
                lora_vocab_start_idx - self.base_layer.vocab_start_index, 0)
            self.embeddings_slice = (self.base_layer.vocab_start_index -
                                     self.base_layer.org_vocab_size +
                                     weights_idx,
                                     self.base_layer.vocab_end_index -
                                     self.base_layer.org_vocab_size)
            self.embeddings_weights = self.base_layer.weight.data[weights_idx:]
            self.embeddings_weights.fill_(0)
        else:
            self.embeddings_slice = None
            self.embeddings_weights = None

        self.embeddings_tensors = torch.zeros(
            (
                max_loras,
                lora_config.lora_extra_vocab_size,
                self.base_layer.embedding_dim,
            ),
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.org_vocab_size +
                lora_config.lora_extra_vocab_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.embedding_dim,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked_2d = self.lora_a_stacked.view(
            self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1],
            self.lora_a_stacked.shape[2],
        )
        self.indices: Optional[torch.Tensor] = None
        self.indices_len: Optional[List[int]] = None
        self.embeddings_indices = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index, :lora_a.shape[0], :lora_a.shape[1]].copy_(
            lora_a, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[
                index, :embeddings_tensor.shape[0], :embeddings_tensor.
                shape[1]].copy_(embeddings_tensor, non_blocking=True)
            if self.embeddings_slice is not None:
                # TODO(yard1): Optimize this copy, we don't need to copy
                # everything, just the modified part
                embeddings = self.embeddings_tensors.view(
                    self.embeddings_tensors.shape[0] *
                    self.embeddings_tensors.shape[1],
                    self.embeddings_tensors.shape[2]
                )[self.embeddings_slice[0]:self.embeddings_slice[1]]
                self.embeddings_weights[:embeddings.shape[0]].copy_(embeddings)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = x > self.base_layer.org_vocab_size - 1
        indices = self.embeddings_indices[1][:self.indices_len[3]].view_as(x)
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = self.embeddings_indices[0][:self.indices_len[3]].view_as(x)
        full_output = self.base_layer.forward(
            x.add_(indices * added_tokens_mask))

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] *
                full_lora_a_embeddings.shape[1], -1)
        bgmv(full_output, full_lora_a_embeddings, self.lora_b_stacked,
             self.indices[:self.indices_len[0]], 0, 1.0)
        return full_output.view_as(full_output_org)

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is VocabParallelEmbedding


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_a_stacked = torch.zeros(
            max_loras,
            1,
            lora_config.max_lora_rank,
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
        self.output_dim = self.lora_b_stacked.shape[2]

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
        if self.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_dim
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

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        _apply_lora(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
        )
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

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is ColumnParallelLinear or (
            type(source_layer) is MergedColumnParallelLinear
            and len(packed_modules_list) == 1)


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
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
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.base_layer.weight.shape[0] // 2,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ) for _ in range(n_slices))

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
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.output_dim
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
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
        _apply_lora_packed_nslice(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
            (self.output_dim, self.output_dim),
        )
        return output

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is MergedColumnParallelLinear and len(
            packed_modules_list) == 2


class QKVParallelLinearWithLora(ColumnParallelLinearWithLoRA):
    """
    ColumnParallelLinear layer that is specifically designed for  
    qkv_proj. Certain models, such as chtglm3 and baichuan-7b,  
    only contains a single LoRA within their qkv_proj layer. 

    During inference with Tensor Parallel, the weights of lora_b 
    must be accurately partitioned according to the respective ranks.
    
    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.q_proj_total_size = (self.base_layer.total_num_heads *
                                  self.base_layer.head_size)
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.kv_proj_total_size = (self.base_layer.total_num_kv_heads *
                                   self.base_layer.head_size)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        if self.tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            self.q_shard_id = tp_rank
            self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
            lora_b_q = lora_b[:, self.q_proj_shard_size *
                              self.q_shard_id:self.q_proj_shard_size *
                              (self.q_shard_id + 1)]
            k_offset = self.q_proj_total_size
            lora_b_k = lora_b[:, k_offset + self.kv_proj_shard_size *
                              self.kv_shard_id:k_offset +
                              self.kv_proj_shard_size * (self.kv_shard_id + 1)]
            v_offset = k_offset + self.kv_proj_total_size
            lora_b_v = lora_b[:, v_offset + self.kv_proj_shard_size *
                              self.kv_shard_id:v_offset +
                              self.kv_proj_shard_size * (self.kv_shard_id + 1)]
            lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=1)

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is QKVParallelLinear and len(
            packed_modules_list) == 1


class MergedQKVParallelLinearWithLora(ColumnParallelLinearWithLoRA):
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
        tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas

        # q, k, v
        self.lora_a_stacked = (
            torch.zeros(
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.base_layer.weight.shape[1],
                dtype=lora_config.lora_dtype,
                device=self.base_layer.weight.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_config.max_lora_rank,
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

        if lora_a[0] is not None:
            self.lora_a_stacked[0][
                index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                    lora_a[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][
                index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                    lora_a[1].T, non_blocking=True)
        if lora_a[2] is not None:
            self.lora_a_stacked[2][
                index, 0, :lora_a[2].shape[1], :lora_a[2].shape[0]].copy_(
                    lora_a[2].T, non_blocking=True)

    def apply_weights(self, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.linear_method.apply_weights(
            self.base_layer.linear_weights, x, bias)
        _apply_lora_packed_nslice(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
            self.output_slices,
        )
        return output

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is QKVParallelLinear and len(
            packed_modules_list) == 3


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer

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
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.weight.shape[0],
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
            tensor_model_parallel_rank = get_tensor_model_parallel_rank()
            shard_size = self.base_layer.weight.shape[1]
            start_idx = tensor_model_parallel_rank * shard_size
            end_idx = (tensor_model_parallel_rank + 1) * shard_size
            lora_a = lora_a[start_idx:end_idx, :]

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
        _apply_lora(
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[0]],
            output,
        )
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

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is RowParallelLinear


class LogitsProcessorWithLoRA(BaseLayerWithLoRA):

    def __init__(
        self,
        base_layer: LogitsProcessor,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

    @property
    def logits_as_input(self):
        return self.base_layer.logits_as_input

    @property
    def vocab_size(self):
        return self.base_layer.vocab_size

    @property
    def scale(self):
        return self.base_layer.scale

    @property
    def org_vocab_size(self):
        return self.base_layer.org_vocab_size

    @property
    def include_gpu_probs_tensor(self):
        return self.base_layer.include_gpu_probs_tensor

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        # Keep this in sync with csrc/punica/bgmv/bgmv_config.h
        if 32000 < self.base_layer.vocab_size > 33024:
            raise ValueError("When using LoRA, vocab size must be "
                             "32000 >= vocab_size <= 33024")
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                # Pad for kernel compatibility
                math.ceil(self.base_layer.vocab_size /
                          lora_config.lora_vocab_padding_size) *
                lora_config.lora_vocab_padding_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.embeddings_tensors = torch.full(
            (max_loras, lora_config.lora_extra_vocab_size, self.hidden_size),
            fill_value=float("-inf"),
            dtype=self.dtype,
            device=self.device,
        )
        self.indices = None
        self.indices_padded = None
        self.indices_len = None

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = float("-inf")

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[
                index, :embeddings_tensor.shape[0], :embeddings_tensor.
                shape[1], ] = embeddings_tensor

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = sampler_indices
        self.indices_padded = sampler_indices_padded
        self.indices_len = indices_len

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_gather(logits)
        if logits is None:
            return None

        lora_logits = torch.empty(
            self.embeddings_tensors.shape[0] + 1,
            self.embeddings_tensors.shape[1],
            hidden_states.shape[0],
            dtype=self.embeddings_tensors.dtype,
            device=self.embeddings_tensors.device,
        )
        torch.matmul(self.embeddings_tensors,
                     hidden_states.T,
                     out=lora_logits[:-1])
        lora_logits[-1] = float("-inf")
        lora_logits = lora_logits.mT
        lora_logits = (lora_logits.reshape(
            lora_logits.shape[0] * lora_logits.shape[1],
            lora_logits.shape[2],
        ).index_select(0,
                       self.indices_padded[:self.indices_len[2]]).nan_to_num_(
                           nan=float("-inf"),
                           posinf=float("inf"),
                           neginf=float("-inf")))
        logits[:,
               self.base_layer.org_vocab_size:self.base_layer.org_vocab_size +
               lora_logits.shape[1]] = lora_logits

        _apply_lora(
            hidden_states,
            self.lora_a_stacked,
            self.lora_b_stacked,
            self.indices[:self.indices_len[1]],
            logits,
        )

        # Remove paddings in vocab (if any).
        logits = logits[:, :self.base_layer.vocab_size]

        return logits

    def forward(self, *args, **kwargs):
        return type(self.base_layer).forward(self, *args, **kwargs)

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        # Special handling for the LogitsProcessor.
        return False


_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    cls
    for cls in globals().values() if inspect.isclass(cls)
    and issubclass(cls, BaseLayerWithLoRA) and cls is not BaseLayerWithLoRA
}


def from_layer(layer: nn.Module,
               max_loras: int,
               lora_config: LoRAConfig,
               packed_modules_list: List,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        if lora_cls.can_replace_layer(layer, lora_config, packed_modules_list,
                                      model_config):
            ret = lora_cls(layer)
            ret.create_lora_weights(max_loras, lora_config, model_config)
            return ret
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device)
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret
