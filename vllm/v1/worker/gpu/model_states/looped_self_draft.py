# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 attention metadata for the shared looped draft/verify prefix.

Shared prefix layers and private draft layers process just the last token of
each verify query. Target-only layers retain the ordinary multi-token metadata.
Separate builders prevent the prefix build
from overwriting buffers referenced by the continuation's CUDA graphs.
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup


class LoopedSelfDraftModelState(DefaultModelState):
    supports_prompt_embeds = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.draft_activation_cache = model.draft_activation_cache
        self.prefix_groups: list[list[AttentionGroup]] | None = None
        self.prefix_slot_buffer: torch.Tensor | None = None
        self.prefix_query_start = torch.zeros(
            self.max_num_reqs + 1, device=device, dtype=torch.int32
        )
        self.prefix_names = (
            set(model.get_draft_attn_layer_names())
            if self.draft_activation_cache
            else set()
        )

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        # Profiling builders are sized for a temporary, much smaller KV cache.
        self.prefix_groups = None

    def prepare_execution(
        self, scheduler_output: SchedulerOutput, dummy_run: bool = False
    ) -> bool:
        cache = self.draft_activation_cache
        if cache is None:
            return False
        cache.allowed = not dummy_run and cache.can_reuse(scheduler_output)
        cache.verify = False
        # Only reuse batches replay the captured graph.
        return not cache.allowed

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        cache = self.draft_activation_cache
        if cache is None:
            return metadata
        width = cache.num_speculative_tokens + 1
        uniform = bool(np.all(input_batch.num_scheduled_tokens == width))
        cache.verify = uniform and (for_capture or cache.allowed)
        if not cache.verify:
            return metadata

        if self.prefix_groups is None:
            self.prefix_groups, _, _ = init_attn_backend(
                kv_cache_config,
                self.vllm_config,
                self.device,
                active_layer_names=self.prefix_names,
            )
            self.prefix_slot_buffer = torch.empty(
                (len(kv_cache_config.kv_cache_groups), self.max_num_reqs),
                device=self.device,
                dtype=slot_mappings.dtype,
            )
        assert self.prefix_slot_buffer is not None
        num_reqs = input_batch.num_reqs
        padded_reqs = input_batch.num_tokens_after_padding // width
        # FULL graphs include padding; eager batches have no padding.
        query_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: padded_reqs + 1] // width
        )
        # Keep preparation asynchronous: copying a pageable CPU tensor here
        # would synchronize the serving stream on every verification step.
        torch.div(
            input_batch.query_start_loc[: padded_reqs + 1],
            width,
            rounding_mode="floor",
            out=self.prefix_query_start[: padded_reqs + 1],
        )
        num_tokens = input_batch.num_tokens_after_padding
        self.prefix_slot_buffer[:, :padded_reqs].copy_(
            slot_mappings[
                :,
                cache.num_speculative_tokens : num_tokens : width,
            ]
        )
        cache.verify_indices[:num_reqs].copy_(input_batch.idx_mapping)
        cache.verify_indices[num_reqs:].zero_()
        prefix_metadata = build_attn_metadata(
            attn_groups=self.prefix_groups,
            num_reqs=padded_reqs,
            num_tokens=padded_reqs,
            query_start_loc_gpu=self.prefix_query_start[: padded_reqs + 1],
            query_start_loc_cpu=query_cpu,
            max_query_len=1,
            seq_lens=input_batch.seq_lens[:padded_reqs],
            max_seq_len=self.max_model_len
            if for_capture
            else int(input_batch.seq_lens_cpu_upper_bound.max()),
            block_tables=block_tables,
            slot_mappings=self.prefix_slot_buffer[:, :padded_reqs],
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:padded_reqs],
            positions=input_batch.positions[cache.num_speculative_tokens :: width],
            is_prefilling=torch.zeros(padded_reqs, dtype=torch.bool),
            for_cudagraph_capture=for_capture,
        )
        metadata.update(prefix_metadata)
        cache.prefix_slots = {
            name: self.prefix_slot_buffer[group_idx, :padded_reqs]
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
            if name in self.prefix_names
        }
        return metadata
