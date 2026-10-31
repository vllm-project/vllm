# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LoRA implementations for specialized, model-specific layers.

Add LoRA support for custom layers that require specialized handling here.
"""

from typing import TYPE_CHECKING, cast

import torch
from transformers import PreTrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

from .base_linear import BaseLinearLayerWithLoRA
from .column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.common.grouped_wo_a import DeepseekV4GroupedWoA


class DeepseekV4GroupedWoAWithLoRA(ColumnParallelLinearWithLoRA):
    """Adds each head group's adapter delta to the grouped ``wo_a`` projection.

    All groups share LoRA-A; group ``g`` uses LoRA-B rows
    ``g * o_lora_rank : (g + 1) * o_lora_rank``, stored as its own buffer so
    the kernels get a contiguous LoRA-B at a fixed address.
    """

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PreTrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        n_groups = cast("DeepseekV4GroupedWoA", self.base_layer).n_local_groups
        lora_b = self.lora_b_stacked[0]
        self.lora_b_stacked = torch.zeros(
            n_groups,
            *lora_b.shape[:2],
            lora_b.shape[2] // n_groups,
            lora_b.shape[3],
            dtype=lora_b.dtype,
            device=lora_b.device,
        ).unbind(0)

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        for lora_b in self.lora_b_stacked:
            lora_b[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        group_size = self.lora_b_stacked[0].shape[2]
        for group, rows in enumerate(lora_b.split(group_size)):
            group_b = self.lora_b_stacked[group][index, 0]
            group_b[: rows.shape[0], : rows.shape[1]].copy_(rows, non_blocking=True)

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        if not self.lora_config.fully_sharded_loras:
            return lora_a
        # Fully sharded LoRA keeps a slice of LoRA-A's rank dim on each rank.
        shard_size = self.lora_a_stacked[0].shape[2]
        return lora_a[self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size]

    def grouped_forward(
        self,
        o: torch.Tensor | QuantizedActivation,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
        einsum_recipe: tuple[int, int, int],
        tma_aligned_scales: bool,
    ) -> torch.Tensor:
        assert not isinstance(o, QuantizedActivation), "RopeQuant is off under LoRA"
        base = cast("DeepseekV4GroupedWoA", self.base_layer)
        z = base.grouped_forward(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=einsum_recipe,
            tma_aligned_scales=tma_aligned_scales,
        )
        x = base.grouped_input(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        shrunk = torch.empty(
            n_groups,
            x.shape[0],
            self.lora_a_stacked[0].shape[2],
            dtype=torch.float32,
            device=x.device,
        )
        for group in range(n_groups):
            self.punica_wrapper.add_shrink(
                shrunk[group : group + 1],
                x[:, group].contiguous(),
                self.lora_a_stacked,
                1.0,
            )
        if self.lora_config.fully_sharded_loras and self.tp_size > 1:
            # Each rank holds a slice of LoRA-A's rank dim.
            shrunk = tensor_model_parallel_all_gather(shrunk)
        for group in range(n_groups):
            group_output = z[:, group].contiguous()
            self.punica_wrapper.add_expand(
                group_output,
                shrunk[group : group + 1],
                (self.lora_b_stacked[group],),
                (o_lora_rank,),
            )
            z[:, group] = group_output
        return z

    @classmethod
    def can_replace_layer(
        cls, source_layer, lora_config, packed_modules_list, model_config=None
    ) -> bool:
        return type(source_layer).__name__ == "DeepseekV4GroupedWoA"


class DeepseekV4CompressorKVScoreWithLoRA(MergedColumnParallelLinearWithLoRA):
    """Adds the adapter delta to the compressor's FP32 ``wkv``/``wgate`` output."""

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer(x)
        # Punica expand writes in the LoRA dtype, so build the delta separately.
        delta = torch.zeros_like(output, dtype=self.lora_b_stacked[0].dtype)
        return output.add_(self._apply_lora_to_output(x, delta))

    @classmethod
    def can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        decorate: bool = True,
    ) -> bool:
        return (
            type(source_layer).__name__ == "DeepseekV4CompressorKVScoreLinear"
            and len(packed_modules_list) == 2
        )


class Step5FusedQKVIndexerWithLoRA(MergedColumnParallelLinearWithLoRA):
    apply = BaseLinearLayerWithLoRA.apply

    def __init__(self, base_layer):
        ColumnParallelLinearWithLoRA.__init__(self, base_layer)
        q_size = base_layer.num_heads * base_layer.head_size
        kv_size = base_layer.num_kv_heads * base_layer.head_size
        self.output_slices = (
            q_size,
            kv_size,
            kv_size,
            base_layer.local_index_q_size,
            base_layer.local_index_k_size,
            base_layer.local_index_z_size,
            base_layer.local_gate_size,
        )
        self.n_slices = len(self.output_slices)

    def slice_lora_b(self, lora_b):
        base = self.base_layer
        kv_rank = self.tp_rank // base.num_kv_head_replicas
        offsets = [
            self.tp_rank * self.output_slices[0],
            kv_rank * self.output_slices[1],
            kv_rank * self.output_slices[2],
            *(
                base._indexer_source_offset(shard)[0]
                for shard in base.INDEXER_SHARD_IDS
            ),
        ]
        return [
            None if weight is None else weight[start : start + size]
            for weight, start, size in zip(
                lora_b, offsets, self.output_slices, strict=True
            )
        ]

    def forward(self, hidden_states):
        output = self.apply(hidden_states)
        size = self.base_layer.qkv_size
        return output[..., :size], output[..., size:]

    @classmethod
    def can_replace_layer(
        cls,
        source_layer,
        lora_config,
        packed_modules_list,
        model_config=None,
        decorate: bool = True,
    ) -> bool:
        return (
            type(source_layer).__name__ == "Step5FusedQKVIndexerLinear"
            and len(packed_modules_list) == 7
            and not lora_config.fully_sharded_loras
        )


class Step5SparseIndexerWWithLoRA(ColumnParallelLinearWithLoRA):
    @classmethod
    def can_replace_layer(
        cls, source_layer, lora_config, packed_modules_list, model_config=None
    ):
        return (
            type(source_layer).__name__ == "Step5SparseIndexerIndexTPLinear"
            and not lora_config.fully_sharded_loras
        )
