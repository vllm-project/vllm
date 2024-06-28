# pylint: disable=unused-argument
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_gather)
from vllm.distributed.utils import divide
from vllm.lora.punica import add_lora, add_lora_slice, bgmv
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

if TYPE_CHECKING:
    pass


def _get_lora_device(base_layer: nn.Module) -> torch.device:
    # code borrowed from https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/vllm/lora/layers.py#L34
    """Returns the device for where to place the LoRA tensors."""
    # unquantizedLinear
    if hasattr(base_layer, "weight"):
        return base_layer.weight.device
    # GPTQ/AWQ/SqueezeLLM
    elif hasattr(base_layer, "qweight"):
        return base_layer.qweight.device
    # marlin
    elif hasattr(base_layer, "B"):
        return base_layer.B.device
    else:
        raise ValueError(f"Unsupported base layer: {base_layer}")


def _not_fully_sharded_can_replace(can_replace):
    """
    decorator which adds the condition of not using fully sharded loras
    intended to wrap can_replace_layer()
    """

    def dec(*args, **kwargs):
        decorate = kwargs.pop('decorate') if 'decorate' in kwargs else True
        condition = (not kwargs['lora_config'].fully_sharded_loras
                     if decorate else True)
        return can_replace(*args, **kwargs) and condition

    return dec


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

    def slice_lora_a(
        self, lora_a: Union[torch.Tensor, List[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, None]]]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    def slice_lora_b(
        self, lora_b: Union[torch.Tensor, List[Union[torch.Tensor, None]]]
    ) -> Union[torch.Tensor, List[Union[torch.Tensor, None]]]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

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
        long_lora_indices: torch.Tensor,
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
        self.embeddings_slice: Optional[Tuple[int, int]]
        self.embeddings_weights: Optional[torch.Tensor]

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        if self.base_layer.num_added_embeddings_per_partition > 0:
            # We can start adding lora weights
            self.embeddings_weights = self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:self.
                base_layer.num_org_embeddings_per_partition +
                self.base_layer.num_added_embeddings_per_partition]
            self.embeddings_slice = (
                self.base_layer.shard_indices.added_vocab_start_index -
                self.base_layer.org_vocab_size,
                self.base_layer.shard_indices.added_vocab_end_index -
                self.base_layer.org_vocab_size)
            self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:].fill_(0)
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
        # Lazily initialized.
        self.indices: torch.Tensor
        self.indices_len: List[int]
        self.embeddings_indices: torch.Tensor

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
                assert self.embeddings_weights is not None
                self.embeddings_weights[:embeddings.shape[0]].copy_(embeddings)

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        long_lora_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.embeddings_indices = embeddings_indices
        self.indices_len = indices_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = x > self.base_layer.org_vocab_size - 1
        embedding_len = self.indices_len[3]
        indices = self.embeddings_indices[1][:embedding_len].view_as(x)
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = self.embeddings_indices[0][:embedding_len].view_as(x)
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
    """
    LoRA on top of ColumnParallelLinear layer.
    
    LoRA B is sliced for tensor parallelism.
    """

    def __init__(self, base_layer: ColumnParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size = self.base_layer.input_size
        self.output_size = self.base_layer.output_size_per_partition
        self.device = _get_lora_device(self.base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_config = lora_config
        self.tp_size = get_tensor_model_parallel_world_size()
        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))
        self.lora_a_stacked = torch.zeros(
            max_loras,
            1,
            lora_a_output_size_per_partition,
            self.input_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.lora_b_stacked = torch.zeros(
            max_loras,
            1,
            self.output_size,
            lora_config.max_lora_rank,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.output_dim = self.lora_b_stacked.shape[2]

        # lazily initialized.
        self.indices: torch.Tensor
        self.indices_len: List[int]

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = self.output_dim
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        lora_b = lora_b[:, start_idx:end_idx]
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

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
        long_lora_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
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
        output_parallel = self.apply(input_, bias)
        if self.base_layer.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = (self.base_layer.bias
                       if self.base_layer.skip_bias_add else None)
        return output, output_bias

    @classmethod
    @_not_fully_sharded_can_replace
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
        self.lora_config = lora_config
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(n_slices))
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.output_size // 2,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(n_slices))

        self.output_dim = self.lora_b_stacked[0].shape[2]
        # Lazily initialized.
        self.indices: torch.Tensor

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_b_stacked[1][index] = 0

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_a

    def slice_lora_b(
        self, lora_b: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        if lora_b[0] is None or lora_b[1] is None:
            return lora_b
        shard_size = self.output_dim
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = [
            lora_b[0][:, start_idx:end_idx], lora_b[1][:, start_idx:end_idx]
        ]
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

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

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
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
    @_not_fully_sharded_can_replace
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

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tensor_model_parallel_rank()
        self.q_shard_id = tp_rank
        self.kv_shard_id = tp_rank // self.base_layer.num_kv_head_replicas
        lora_b_q = lora_b[:, self.q_proj_shard_size *
                          self.q_shard_id:self.q_proj_shard_size *
                          (self.q_shard_id + 1)]
        k_offset = self.q_proj_total_size
        lora_b_k = lora_b[:, k_offset +
                          self.kv_proj_shard_size * self.kv_shard_id:k_offset +
                          self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        v_offset = k_offset + self.kv_proj_total_size
        lora_b_v = lora_b[:, v_offset +
                          self.kv_proj_shard_size * self.kv_shard_id:v_offset +
                          self.kv_proj_shard_size * (self.kv_shard_id + 1)]
        lora_b = torch.cat([lora_b_q, lora_b_k, lora_b_v], dim=1)
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    @classmethod
    @_not_fully_sharded_can_replace
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
        self.lora_config = lora_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))
        # q, k, v
        self.lora_a_stacked = (
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )
        self.lora_b_stacked = (
            torch.zeros(
                max_loras,
                1,
                self.q_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

        self.output_slices = (self.q_proj_shard_size, self.kv_proj_shard_size,
                              self.kv_proj_shard_size)
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        # lazily initialized.
        self.indices_len: List[int]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_b_stacked[1][index] = 0
        self.lora_a_stacked[2][index] = 0
        self.lora_b_stacked[2][index] = 0

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_a

    def slice_lora_b(
        self, lora_b: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        lora_b_q, lora_b_k, lora_b_v = None, None, None
        if lora_b[0] is not None:
            lora_b_q = lora_b[0][:, self.q_proj_shard_size *
                                 self.q_shard_id:self.q_proj_shard_size *
                                 (self.q_shard_id + 1)]
        if lora_b[1] is not None:
            lora_b_k = lora_b[1][:, self.kv_proj_shard_size *
                                 self.kv_shard_id:self.kv_proj_shard_size *
                                 (self.kv_shard_id + 1)]
        if lora_b[2] is not None:
            lora_b_v = lora_b[2][:, self.kv_proj_shard_size *
                                 self.kv_shard_id:self.kv_proj_shard_size *
                                 (self.kv_shard_id + 1)]
        lora_b = [lora_b_q, lora_b_k, lora_b_v]
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        if lora_b[0] is not None:
            lora_b_q = lora_b[0]
            self.lora_b_stacked[0][
                index, 0, :lora_b_q.shape[1], :lora_b_q.shape[0]].copy_(
                    lora_b_q.T, non_blocking=True)
        if lora_b[1] is not None:
            lora_b_k = lora_b[1]
            self.lora_b_stacked[1][
                index, 0, :lora_b_k.shape[1], :lora_b_k.shape[0]].copy_(
                    lora_b_k.T, non_blocking=True)
        if lora_b[2] is not None:
            lora_b_v = lora_b[2]
            self.lora_b_stacked[2][
                index, 0, :lora_b_v.shape[1], :lora_b_v.shape[0]].copy_(
                    lora_b_v.T, non_blocking=True)

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

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
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
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is QKVParallelLinear and len(
            packed_modules_list) == 3


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size
        self.device = _get_lora_device(self.base_layer)

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:
        self.lora_config = lora_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.input_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        tp_size = get_tensor_model_parallel_world_size()
        lora_b_output_size_per_partition = (
            self.output_size if not lora_config.fully_sharded_loras else
            divide(self.output_size, tp_size))

        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_b_output_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        # Lazily initialized
        self.indices: torch.Tensor
        self.indices_len: List[int]

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = self.input_size
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.base_layer.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

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
        long_lora_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.indices = base_indices
        self.indices_len = indices_len

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)
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
        return output, output_bias

    @property
    def weight(self):
        return self.base_layer.weight if hasattr(
            self.base_layer, "weight") else self.base_layer.qweight

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        return type(source_layer) is RowParallelLinear


class LogitsProcessorWithLoRA(BaseLayerWithLoRA):
    """
    LoRA wrapper for LogitsProcessor, with extra logic to handle the
    application of the LoRA adapter and added LoRA vocabulary.

    Args:
        base_layer: LogitsProcessor layer
        hidden_size: hidden size of the model
        dtype: data type of the model
        device: device of the model
        sharded_to_full_mapping: index mapping from sharded vocab to full vocab
            received from base_layer.get_sharded_to_full_mapping(). If None,
            no reindexing will be done.
    """

    def __init__(self, base_layer: LogitsProcessor, hidden_size: int,
                 dtype: torch.dtype, device: torch.device,
                 sharded_to_full_mapping: Optional[List[int]]) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sharded_to_full_mapping = sharded_to_full_mapping

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
    def soft_cap(self):
        return self.base_layer.soft_cap

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
        if 32000 < self.base_layer.vocab_size > 128512:
            raise ValueError("When using LoRA, vocab size must be "
                             "32000 >= vocab_size <= 128512")
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
        if self.sharded_to_full_mapping is not None:
            self.sharded_to_full_mapping_gpu = torch.tensor(
                self.sharded_to_full_mapping,
                device=self.device,
                dtype=torch.long)
        else:
            self.sharded_to_full_mapping_gpu = None
        # Lazily initialized.
        self.indices: torch.Tensor
        self.indices_len: List[int]
        self.indices_padded: torch.Tensor

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
        long_lora_indices: torch.Tensor,
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

        if self.sharded_to_full_mapping_gpu is not None:
            # Reindex full logits tensor to ensure 1:1 mapping between
            # index and token_id
            # Example for:
            #   org_vocab_size = 4
            #   added_vocab_size = 2
            #   pad_to_size = 8
            #   tp_size = 2

            # indices:  [0, 1, 2,  3, 4, 5, 6,  7]
            # token_id: [0, 1, 4, -1, 2, 3, 5, -1]

            # Therefore, the mapping is expected to be:
            # [0, 1, 4, 6, 2, 3, 5, 7] so that when we reindex,
            # we get:
            # indices:  [0, 1, 2, 3, 4, 5,  6,  7]
            # token_id: [0, 1, 2, 3, 4, 5, -1, -1]
            logits = logits[:, self.sharded_to_full_mapping_gpu]

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


class LinearScalingRotaryEmbeddingWithLora(BaseLayerWithLoRA):
    """Implements RoPE-scaled embeddings with linear scaling for
    multiple LoRA adapters with a specialized kernel.

    Replace LinearScalingRotaryEmbedding with MultiLinearScalingRotaryEmbedding
    which can handle multi lora adapters in a specialied kernel.
    """

    def __init__(self, base_layer: RotaryEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        # Lazily initialized
        self.long_lora_indices: torch.Tensor
        self.indices_len: List[int]

    @property
    def scaling_factors(self):
        return self.base_layer.scaling_factors

    @property
    def rotary_dim(self):
        return self.base_layer.rotary_dim

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        scaling_factors = list(
            lora_config.long_lora_scaling_factors
        ) if lora_config.long_lora_scaling_factors else []
        base_scaling_factor = (self.base_layer.scaling_factor if isinstance(
            self.base_layer, LinearScalingRotaryEmbedding) else 1.0)
        scaling_factors = sorted(
            list(set([base_scaling_factor] + scaling_factors)))
        self.base_layer = LinearScalingRotaryEmbedding(
            self.base_layer.head_size,
            self.base_layer.rotary_dim,
            self.base_layer.max_position_embeddings,
            self.base_layer.base,
            self.base_layer.is_neox_style,
            scaling_factors,
            self.base_layer.dtype,
        )

    def reset_lora(self, index: int):
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        ...

    def set_mapping(
        self,
        base_indices: torch.Tensor,
        sampler_indices: torch.Tensor,
        sampler_indices_padded: torch.Tensor,
        embeddings_indices: torch.Tensor,
        long_lora_indices: torch.Tensor,
        indices_len: List[int],
    ):
        self.long_lora_indices = long_lora_indices
        self.indices_len = indices_len

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_layer(
            positions,
            query,
            key,
            offsets=self.long_lora_indices[:self.indices_len[4]])

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self.base_layer.scaling_factor_to_offset

    @classmethod
    def can_replace_layer(cls, source_layer: nn.Module,
                          lora_config: LoRAConfig, packed_modules_list: List,
                          model_config: Optional[PretrainedConfig]) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        return type(source_layer) is LinearScalingRotaryEmbedding or type(
            source_layer) is RotaryEmbedding

    def extra_repr(self) -> str:
        return self.base_layer.extra_repr()
