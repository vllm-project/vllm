# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import copy
from dataclasses import replace as dataclass_replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.utils import AttentionGroup


class StandaloneDraftModelSpeculator(AutoRegressiveSpeculator):
    reuse_target_attn_metadata = False
    pass_hidden_states_to_model = False
    prefill_sample_position_offset = 0
    prefill_seq_len_offset = 1

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        spec = self.speculative_config
        parallel = vllm_config.parallel_config
        unsupported = vllm_config._get_v2_model_runner_unsupported_features()
        if unsupported:
            raise ValueError(f"V2 standalone drafting does not support {unsupported}")
        spec.verify_equal_vocab_size_if_draft_model()
        self.is_dummy_run = torch.zeros((), dtype=torch.bool, device=device)

        self.draft_vllm_config = replace(
            vllm_config,
            model_config=self.draft_model_config,
            quant_config=None,
            parallel_config=replace(spec.draft_parallel_config, rank=parallel.rank),
            compilation_config=copy(vllm_config.compilation_config),
            scheduler_config=replace(
                vllm_config.scheduler_config,
                max_num_batched_tokens=self.max_num_tokens,
                max_model_len=self.draft_model_config.max_model_len,
                is_encoder_decoder=self.draft_model_config.is_encoder_decoder,
            ),
            attention_config=replace(
                vllm_config.attention_config, backend=spec.attention_backend
            ),
            cache_config=replace(
                vllm_config.cache_config,
                cache_dtype=spec.kv_cache_dtype or vllm_config.cache_config.cache_dtype,
            ),
        )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return self.draft_vllm_config

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            return get_model(
                vllm_config=self.draft_vllm_config,
                prefix="draft_model",
                load_config=self.speculative_config.draft_load_config,
            )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        # Share the scheduler's block tables, but use an expanded, persistent
        # slot buffer for both graph capture and replay.
        self.block_tables = copy(block_tables)
        self.block_tables.max_num_batched_tokens = self.max_num_tokens
        self.block_tables.slot_mappings = torch.full(
            (block_tables.num_kv_cache_groups, self.max_num_tokens),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=self.device,
        )
        self.model_state = DefaultModelState(
            self.draft_vllm_config, self.model, None, self.device
        )

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        dummy_run: bool = False,
    ) -> InputBatch:
        self.is_dummy_run.fill_(dummy_run)
        prepare_draft_model_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
            self.max_model_len,
        )
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens + num_reqs
        query_start_loc_np = input_batch.query_start_loc_np[: num_reqs + 1] + np.arange(
            num_reqs + 1, dtype=np.int32
        )
        return dataclass_replace(
            input_batch,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_scheduled_tokens=input_batch.num_scheduled_tokens + 1,
            input_ids=self.input_buffers.input_ids[:num_tokens],
            positions=self.input_buffers.positions[:num_tokens],
            is_padding=self.input_buffers.is_padding[:num_tokens],
            query_start_loc=self.input_buffers.query_start_loc[: num_reqs + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            seq_lens_cpu_upper_bound=(input_batch.seq_lens_cpu_upper_bound + 1).clamp(
                max=self.max_model_len
            ),
        )

    def prepare_attn(
        self, input_batch: InputBatch, batch_desc: BatchExecutionDescriptor
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
        self.block_tables.gather_block_tables(
            self.idx_mapping[: input_batch.num_reqs], self.max_num_reqs
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            self.idx_mapping[: input_batch.num_reqs],
            self.input_buffers.query_start_loc[: input_batch.num_reqs + 1],
            self.input_buffers.positions,
            batch_desc.num_tokens,
        )
        slot_mappings.masked_fill_(
            self.input_buffers.is_padding[None, : batch_desc.num_tokens]
            | self.is_dummy_run,
            PAD_SLOT_ID,
        )
        attn_metadata = self._build_draft_attn_metadata(
            num_reqs=input_batch.num_reqs,
            num_reqs_padded=batch_desc.num_reqs or input_batch.num_reqs,
            num_tokens_padded=batch_desc.num_tokens,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=0,
            query_start_loc_np=input_batch.query_start_loc_np,
        )
        return attn_metadata, build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )

    def compute_decode_slot_mappings(
        self, num_reqs: int, num_tokens: int
    ) -> torch.Tensor:
        slot_mappings = super().compute_decode_slot_mappings(num_reqs, num_tokens)
        # Positions clamp at the context limit, but these draws are discarded.
        # Keep their KV writes from replacing the last valid cached token.
        slot_mappings.masked_fill_(
            (self.sample_src_positions[None, :num_tokens] >= self.max_model_len)
            | self.is_dummy_run,
            PAD_SLOT_ID,
        )
        return slot_mappings


@triton.jit
def _prepare_draft_model_prefill_inputs_kernel(
    last_token_indices_ptr,
    current_step_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_is_padding_ptr,
    draft_query_start_loc_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    target_query_start_loc_ptr,
    target_seq_lens_ptr,
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    max_num_reqs: tl.constexpr,
    max_num_tokens: tl.constexpr,
    max_model_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        end = tl.load(target_query_start_loc_ptr + num_reqs) + num_reqs
        for i in range(end, max_num_tokens, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            tl.store(draft_input_ids_ptr + idx, 0, idx < max_num_tokens)
            tl.store(draft_positions_ptr + idx, 0, idx < max_num_tokens)
            tl.store(draft_is_padding_ptr + idx, True, idx < max_num_tokens)
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            tl.store(draft_query_start_loc_ptr + idx, end, idx <= max_num_reqs)
            tl.store(draft_seq_lens_ptr + idx, 0, idx < max_num_reqs)
            tl.store(last_token_indices_ptr + idx, 0, idx < max_num_reqs)
        tl.store(current_step_ptr, 0)
        return

    start = tl.load(target_query_start_loc_ptr + req_idx)
    end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    query_len = end - start
    rejected = tl.load(num_rejected_ptr + req_idx)
    accepted = query_len - rejected
    seq_len = tl.load(target_seq_lens_ptr + req_idx) - rejected
    append = seq_len < max_model_len
    valid_len = accepted + append.to(tl.int32)
    # Right alignment preserves causal positions while keeping CPU-visible
    # query lengths independent of the device's rejection counts.
    pad = query_len + 1 - valid_len
    draft_start = start + req_idx
    state_idx = tl.load(idx_mapping_ptr + req_idx)
    sampled = tl.load(num_sampled_ptr + req_idx)
    if sampled > 0:
        next_token = tl.load(last_sampled_ptr + state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + state_idx).to(tl.int32)

    for i in range(0, query_len + 1, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        src_idx = idx - pad
        is_target = (src_idx >= 0) & (src_idx < accepted)
        token = tl.load(target_input_ids_ptr + start + src_idx, is_target, other=0)
        position = tl.load(target_positions_ptr + start + src_idx, is_target, other=0)
        is_bonus = append & (src_idx == accepted)
        token = tl.where(is_bonus, next_token, token)
        position = tl.where(is_bonus, seq_len, position)
        is_padding = ~(is_target | is_bonus)
        tl.store(draft_input_ids_ptr + draft_start + idx, token, idx <= query_len)
        tl.store(draft_positions_ptr + draft_start + idx, position, idx <= query_len)
        tl.store(draft_is_padding_ptr + draft_start + idx, is_padding, idx <= query_len)
    tl.store(draft_query_start_loc_ptr + req_idx, draft_start)
    tl.store(draft_seq_lens_ptr + req_idx, tl.minimum(seq_len + 1, max_model_len))
    tl.store(last_token_indices_ptr + req_idx, end + req_idx)


def prepare_draft_model_prefill_inputs(
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    max_num_reqs: int,
    max_model_len: int,
) -> None:
    _prepare_draft_model_prefill_inputs_kernel[(input_batch.num_reqs + 1,)](
        last_token_indices,
        current_draft_step,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.is_padding,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        input_batch.idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
        input_buffers.max_num_tokens,
        max_model_len,
        BLOCK_SIZE=1024,
    )
