# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata of the whole-batch sequence-parallel DSA layers."""

from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.models.common.ops.sequence_parallel import (
    zigzag_chunk_size,
    zigzag_shards,
)
from vllm.models.deepseek_v32.sequence_parallel import zigzag_local_batch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

AttentionGroups = list[list[AttentionGroup]]


class DeepseekV32SequenceParallelModelState(DefaultModelState):
    """Builds the sequence-parallel layers' attention metadata over this TP
    rank's zigzag tokens (`zigzag_local_batch`). MTP draft layers keep the
    step's metadata."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.world_size = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        layers = vllm_config.compilation_config.static_forward_context
        self.sp_layer_names = {
            name
            for name, layer in layers.items()
            if getattr(layer, "sequence_parallel", False)
        }
        self.max_local_tokens = 2 * zigzag_chunk_size(
            self.max_num_tokens, self.world_size
        )
        # A request split by both of a rank's chunks counts twice, and FULL
        # graphs give each token a request of its own.
        self.max_local_reqs = max(self.max_num_reqs + 1, self.max_local_tokens)
        # Persistent, so FULL graphs replay over the same buffers.
        self.query_start_loc = torch.zeros(
            self.max_local_reqs + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(
            self.max_local_reqs, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(
            self.max_local_tokens, dtype=torch.int64, device=device
        )
        self.block_tables: dict[int, torch.Tensor] = {}
        self.slot_mappings: torch.Tensor | None = None
        # The groups last split, and their sequence-parallel and other groups.
        self._split: tuple[AttentionGroups, AttentionGroups, AttentionGroups] | None = (
            None
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: AttentionGroups,
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        sp_groups, other_groups = self._split_attn_groups(attn_groups, kv_cache_config)
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            other_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        return attn_metadata | self._prepare_local_attn(
            input_batch,
            # A FULL graph replays one local layout per size.
            cudagraph_mode == CUDAGraphMode.FULL or for_capture,
            block_tables,
            slot_mappings,
            sp_groups,
            kv_cache_config,
            for_capture,
        )

    def _split_attn_groups(
        self,
        attn_groups: AttentionGroups,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[AttentionGroups, AttentionGroups]:
        """The groups of the sequence-parallel layers, and the other groups."""
        if self._split is None or self._split[0] is not attn_groups:
            sp_groups: AttentionGroups = []
            other_groups: AttentionGroups = []
            for groups, kv_cache_group in zip(
                attn_groups, kv_cache_config.kv_cache_groups
            ):
                own = []
                others = []
                for group in groups:
                    names = [n for n in group.layer_names if n in self.sp_layer_names]
                    if len(names) < len(group.layer_names):
                        others.append(group)
                    if not names:
                        continue
                    if len(names) < len(group.layer_names):
                        # The group also holds MTP layers: these layers get
                        # builders of their own.
                        group = replace(group, layer_names=names, metadata_builders=[])
                        group.create_metadata_builders(
                            self.vllm_config,
                            self.device,
                            group.kernel_block_size,
                            kv_cache_group=kv_cache_group,
                        )
                    table = group.kernel_block_table
                    if table is not None and table.shape[1] < self.max_local_reqs:
                        group.kernel_block_table = table.new_zeros(
                            table.shape[0], self.max_local_reqs, table.shape[2]
                        )
                    own.append(group)
                sp_groups.append(own)
                other_groups.append(others)
            self._split = (attn_groups, sp_groups, other_groups)
        return self._split[1], self._split[2]

    def _prepare_local_attn(
        self,
        input_batch: InputBatch,
        per_token: bool,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        sp_groups: AttentionGroups,
        kv_cache_config: KVCacheConfig,
        for_capture: bool,
    ) -> dict[str, Any]:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        local = zigzag_local_batch(
            input_batch.query_start_loc_np[: num_reqs + 1],
            num_tokens,
            num_tokens_padded,
            self.world_size,
            self.rank,
            per_token,
        )
        shards = zigzag_shards(num_tokens_padded, self.device)
        num_local_slots = shards.local_rows.shape[0]
        num_local_reqs = len(local.req_indices)
        if per_token:
            # Padded like a FULL graph's batch: zero-length requests and the
            # padding tokens after the last one.
            num_reqs_padded = num_local_slots
            num_actual_tokens = num_local_slots
        else:
            # A rank without tokens keeps one zero-length request.
            num_reqs_padded = max(num_local_reqs, 1)
            num_actual_tokens = local.num_tokens
        num_pad_reqs = num_reqs_padded - num_local_reqs
        req_indices_np = np.pad(local.req_indices, (0, num_pad_reqs))
        query_start_loc_np = np.pad(local.query_start_loc, (0, num_pad_reqs), "edge")

        req_indices = async_tensor_h2d(
            req_indices_np, device=self.device, dtype=torch.int64
        )
        tails = async_tensor_h2d(local.tails, device=self.device, dtype=torch.int32)
        async_tensor_h2d(
            query_start_loc_np, out=self.query_start_loc[: num_reqs_padded + 1]
        )
        seq_lens = self.seq_lens[:num_reqs_padded]
        torch.sub(
            input_batch.seq_lens[req_indices[:num_local_reqs]],
            tails,
            out=seq_lens[:num_local_reqs],
        )
        seq_lens[num_local_reqs:].zero_()
        torch.index_select(
            input_batch.positions,
            0,
            shards.local_rows,
            out=self.positions[:num_local_slots],
        )

        seq_lens_cpu_upper_bound = torch.zeros(num_reqs_padded, dtype=torch.int32)
        seq_lens_cpu_upper_bound[:num_local_reqs] = (
            input_batch.seq_lens_cpu_upper_bound[local.req_indices]
            - torch.from_numpy(local.tails)
        )
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(seq_lens_cpu_upper_bound.max())
        max_query_len = int(np.diff(query_start_loc_np).max(initial=0))

        if self.slot_mappings is None:
            self.slot_mappings = slot_mappings.new_full(
                (slot_mappings.shape[0], self.max_local_tokens), PAD_SLOT_ID
            )
        rank_rows = shards.rank_major_rows[
            self.rank * num_local_slots : (self.rank + 1) * num_local_slots
        ]
        local_block_tables = list(block_tables)
        for kv_cache_group_id, groups in enumerate(sp_groups):
            if not groups:
                continue
            block_table = block_tables[kv_cache_group_id]
            local_block_table = self.block_tables.get(kv_cache_group_id)
            if local_block_table is None:
                local_block_table = block_table.new_zeros(
                    self.max_local_reqs, block_table.shape[1]
                )
                self.block_tables[kv_cache_group_id] = local_block_table
            local_block_tables[kv_cache_group_id] = torch.index_select(
                block_table, 0, req_indices, out=local_block_table[:num_reqs_padded]
            )
            local_slot_mapping = self.slot_mappings[kv_cache_group_id, :num_local_slots]
            torch.index_select(
                slot_mappings[kv_cache_group_id],
                0,
                shards.local_rows,
                out=local_slot_mapping,
            )
            # Rows past the step's tokens only pad the zigzag chunks.
            local_slot_mapping.masked_fill_(rank_rows >= num_tokens, PAD_SLOT_ID)

        return build_attn_metadata(
            attn_groups=sp_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_actual_tokens,
            query_start_loc_gpu=self.query_start_loc[: num_reqs_padded + 1],
            query_start_loc_cpu=torch.from_numpy(query_start_loc_np),
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            block_tables=local_block_tables,
            slot_mappings=self.slot_mappings[:, :num_local_slots],
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            positions=self.positions[:num_local_slots],
            is_prefilling=torch.from_numpy(
                input_batch.is_prefilling_np[req_indices_np]
            ),
            for_cudagraph_capture=for_capture,
            req_idx=input_batch.idx_mapping_np[req_indices_np],
        )
