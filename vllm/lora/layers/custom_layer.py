# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LoRA implementations for specialized, model-specific layers.

Add LoRA support for custom layers that require specialized handling here.
"""

from vllm.distributed.utils import divide
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
)

from .base_linear import BaseLinearLayerWithLoRA
from .column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
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


class MiniMaxM3QKVIndexerWithLoRA(MergedQKVParallelLinearWithLoRA):
    def __init__(self, base_layer):
        super().__init__(base_layer)
        self.output_slices = tuple(
            divide(size, self.tp_size) for size in base_layer.output_sizes
        )
        self.output_ids = (*self.output_ids, self.kv_shard_id, 0)

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
            type(source_layer)
            is maybe_get_oot_by_class(MinimaxM3QKVParallelLinearWithIndexer)
            and len(packed_modules_list) == 5
            and not lora_config.fully_sharded_loras
        )


class MiniMaxM3DenseQKVWithLoRA(MergedQKVParallelLinearWithLoRA):
    def set_lora(self, index, lora_a, lora_b):
        assert all(b is None for b in lora_b[3:])
        super().set_lora(index, lora_a[:3], lora_b[:3])

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
            type(source_layer) is maybe_get_oot_by_class(QKVParallelLinear)
            and len(packed_modules_list) == 5
            and not lora_config.fully_sharded_loras
        )
