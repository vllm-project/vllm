# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module implements BD-LoRA (Wang et al., 2025,
https://arxiv.org/pdf/2510.23346v1). BD-LoRA aims to eliminate communication
overheads in multi-LoRA scenarios. In BD-LoRA, some adapters are saved in a
block-diagonal manner. Following the standard megatron-sharding, the
column-parallel linear layers (e.g. QKV, up, gate projections) have a vanilla
LoRA A adapter and a block-diagonal LoRA B adapter. Similarly, row-parallel
linear layers (e.g. down, out projections) have a block-diagonal LoRA A and
vanilla LoRA B adapter.

In order to save memory, block-diagonal matrices are saved by stacking their
blocks first dimension.

For example, assume the following setup:
  - Up projection linear layer with shape (out_features, in_features) = (2048, 1024)
  - LoRA modules of rank 64
  - Sharding with TP degree 2

Then:
  - In Vanilla LoRA, we would have adapters LoRA A (64, 1024) and LoRA B (2048, 64)
  - In BD-LoRA, the LoRA B matrix will be block-diagonal (each block having
    shape (out_features / TP, rank / TP) = (1024, 32))
  - The blocks are stacked along the first dimension, so the actual BD-LoRA
    adapter for LoRA B will be saved with shape (2048, 32)
  - To serve, we slice LoRA B along the out-features dimension, and LoRA A
    along the rank dimension

To achieve this, we add an option ('block_diagonal_sharded_loras') to the
LoRA-Config to specify that BD-LoRA will be used, and we provide subclasses
in this module that perform the sharding logic.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, logging

from vllm.config import LoRAConfig
from vllm.lora.layers.column_parallel_linear import (
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
)
from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.model_executor.layers.linear import QKVParallelLinear

logger = logging.get_logger(__name__)


class BdLoraMixin:
    """This mixin ensures that each BD-LoRA class has the correct replace-layer
    function.

    When writing a BD-LoRA class, put this Mixin as the first parent class, so that this
    function overrides the can_replace_layer class from other classes.
    """

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            super().can_replace_layer(  # type: ignore
                source_layer=source_layer,
                lora_config=lora_config,
                packed_modules_list=packed_modules_list,
                model_config=model_config,
                decorate=False,
            )
            and lora_config.block_diagonal_sharded_loras
        )


class MergedColumnParallelLinearWithBlockDiagonalShardedLoRA(
    BdLoraMixin, MergedColumnParallelLinearWithLoRA
):
    """BD-LoRA version of MergedColumnParallelLinearWithLoRA.

    Uses a block-diagonal LoRA B adapter.
    """

    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]:
        sliced_lora_a = [None] * self.n_slices
        shard_size = self.lora_a_stacked[0].shape[2]
        for i, shard_id in enumerate(self.output_ids):
            if (lora_a_i := lora_a[i]) is not None:
                sliced_lora_a[i] = lora_a_i[
                    shard_size * shard_id : shard_size * (shard_id + 1), :
                ]  # type: ignore
        return sliced_lora_a


class QKVParallelLinearWithBlockDiagonalShardedLoRA(
    BdLoraMixin, QKVParallelLinearWithLoRA
):
    # So far, I have not found a neural net that uses this class (instead of
    # the merged versions) so I have not come around to test it.
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MergedQKVParallelLinearWithBlockDiagonalShardedLoRA(
    MergedColumnParallelLinearWithBlockDiagonalShardedLoRA
):
    """BD-LoRA version of MergedQKVParallelLinearWithLoRA.

    Uses a block-diagonal LoRA B adapter.
    In essence, it is a specialized form of
    MergedColumnParallelLinearWithBlockDiagonalShardedLoRA and follows the same
    sharding logic, but we need this class to mark QKV Layers which have been
    replaced (so we only override the can_replace_layer method).
    """

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            type(source_layer) is QKVParallelLinear
            and len(packed_modules_list) == 3
            and lora_config.block_diagonal_sharded_loras
        )


class RowParallelLinearWithBlockDiagonalShardedLoRA(
    BdLoraMixin, RowParallelLinearWithLoRA
):
    """BD-LoRA version of RowParallelLinearWithLoRA.

    Uses a block-diagonal LoRA A adapter.
    """

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        shard_size = self.lora_config.max_lora_rank // self.tp_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        # Lora B has shape [out_size, rank]
        shard_size = self.lora_config.max_lora_rank // self.tp_size
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = lora_b[:, start_idx:end_idx]
        return lora_b
