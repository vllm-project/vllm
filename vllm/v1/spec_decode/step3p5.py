# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.distributed import get_dcp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.block_table import BlockTable
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@triton.jit
def _prepare_step_mtp_input_buffers_kernel(
    last_token_indices_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    cached_history_lens_ptr,
    history_slot_mapping_ptr,
    next_token_ids_ptr,
    prefill_token_overrides_ptr,
    prefill_token_overrides_stride0,
    num_rejected_ptr,
    query_start_loc_ptr,
    history_window_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    history_slot = tl.load(history_slot_mapping_ptr + req_idx)
    if history_slot < 0:
        query_start = tl.load(query_start_loc_ptr + req_idx)
        tl.store(draft_seq_lens_ptr + req_idx, 0)
        tl.store(last_token_indices_ptr + req_idx, max(query_start - 1, 0))
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    if query_len <= 0:
        # Keep the later gather in bounds for zero-query rows. The sampled
        # value is discarded by the runner.
        tl.store(draft_seq_lens_ptr + req_idx, 0)
        tl.store(last_token_indices_ptr + req_idx, max(query_start - 1, 0))
        return

    history_len = tl.load(cached_history_lens_ptr + history_slot)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill = min(max(0, num_rejected - 1), history_len)
    num_input_tokens = query_len - num_rejected

    seq_len = tl.load(draft_seq_lens_ptr + req_idx)
    tl.store(draft_seq_lens_ptr + req_idx, seq_len - num_reprefill)

    next_token = tl.load(next_token_ids_ptr + req_idx)
    first_override = tl.load(
        prefill_token_overrides_ptr + req_idx * prefill_token_overrides_stride0
    )
    next_token = tl.where(first_override >= 0, first_override, next_token)

    for i in range(1, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(
            draft_input_ids_ptr + query_start + num_reprefill - 1 + block,
            input_ids,
            mask=mask,
        )

    last_token_index = query_start + num_reprefill + num_input_tokens - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    for i in range(0, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(
            draft_positions_ptr + query_start + num_reprefill + block,
            target_pos,
            mask=mask,
        )

    first_position = tl.load(target_positions_ptr + query_start)
    for i in range(num_reprefill):
        cache_read_slot = history_window_size - num_reprefill + i
        cached_token_id = tl.load(
            cached_draft_input_ids_ptr
            + history_slot * cached_draft_input_ids_stride0
            + cache_read_slot,
        )
        tl.store(draft_input_ids_ptr + query_start + i, cached_token_id)
        tl.store(
            draft_positions_ptr + query_start + i,
            first_position - num_reprefill + i,
        )


@triton.jit
def _prepare_step_mtp_hidden_states_kernel(
    draft_hidden_states_ptr,
    draft_hidden_states_stride0,
    target_hidden_states_ptr,
    target_hidden_states_stride0,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    draft_input_embeds_ptr,
    draft_input_embeds_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    cached_history_lens_ptr,
    history_slot_mapping_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    history_window_size,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    dim_block = dim_block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    history_slot = tl.load(history_slot_mapping_ptr + req_idx)
    if history_slot < 0:
        return
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    if query_len <= 0:
        return

    history_len = tl.load(cached_history_lens_ptr + history_slot)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill = min(max(0, num_rejected - 1), history_len)
    num_input_hidden_states = query_len - num_rejected

    query_block = query_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    query_mask = (query_block < num_input_hidden_states)[:, None] & dim_mask[None, :]
    hidden_state = tl.load(
        target_hidden_states_ptr
        + (query_start + query_block)[:, None] * target_hidden_states_stride0
        + dim_block[None, :],
        mask=query_mask,
    )
    tl.store(
        draft_hidden_states_ptr
        + (query_start + num_reprefill + query_block)[:, None]
        * draft_hidden_states_stride0
        + dim_block[None, :],
        hidden_state,
        mask=query_mask,
    )

    if query_block_idx == 0:
        for i in range(num_reprefill):
            cache_read_slot = history_window_size - num_reprefill + i
            cached_hidden_state = tl.load(
                cached_target_hidden_states_ptr
                + history_slot * cached_target_hidden_states_stride0
                + cache_read_slot * cached_target_hidden_states_stride1
                + dim_block,
                mask=dim_mask,
            )
            tl.store(
                draft_hidden_states_ptr
                + (query_start + i) * draft_hidden_states_stride0
                + dim_block,
                cached_hidden_state,
                mask=dim_mask,
            )
            if USE_INPUT_EMBEDS:
                cached_embed = tl.load(
                    cached_draft_input_embeds_ptr
                    + history_slot * cached_draft_input_embeds_stride0
                    + cache_read_slot * cached_draft_input_embeds_stride1
                    + dim_block,
                    mask=dim_mask,
                )
                tl.store(
                    draft_input_embeds_ptr
                    + (query_start + i) * draft_input_embeds_stride0
                    + dim_block,
                    cached_embed,
                    mask=dim_mask,
                )


@triton.jit
def _cache_step_mtp_inputs_kernel(
    draft_input_ids_ptr,
    draft_input_embeds_ptr,
    draft_input_embeds_stride0,
    draft_hidden_states_ptr,
    draft_hidden_states_stride0,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    cached_history_lens_ptr,
    history_slot_mapping_ptr,
    last_token_indices_ptr,
    query_start_loc_ptr,
    history_window_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < hidden_size

    history_slot = tl.load(history_slot_mapping_ptr + req_idx)
    if history_slot < 0:
        return
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    if query_end <= query_start:
        return
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    valid_query_len = last_token_index - query_start + 1
    current_history_len = min(history_window_size, valid_query_len)
    old_history_len = tl.load(cached_history_lens_ptr + history_slot)
    old_history_to_keep = min(
        old_history_len,
        history_window_size - current_history_len,
    )
    history_len = old_history_to_keep + current_history_len
    old_cache_read_start = history_window_size - old_history_to_keep
    old_cache_write_start = history_window_size - history_len
    current_query_start = last_token_index - current_history_len + 1
    current_cache_write_start = history_window_size - current_history_len

    # A short query appends to the retained suffix instead of replacing it.
    # History is right-aligned so rejection repair can always read its suffix
    # from ``history_window_size - num_reprefill``.
    for i in range(old_history_to_keep):
        cache_read_slot = old_cache_read_start + i
        cache_write_slot = old_cache_write_start + i
        if block_idx == 0:
            input_id = tl.load(
                cached_draft_input_ids_ptr
                + history_slot * cached_draft_input_ids_stride0
                + cache_read_slot,
            )
            tl.store(
                cached_draft_input_ids_ptr
                + history_slot * cached_draft_input_ids_stride0
                + cache_write_slot,
                input_id,
            )
        if USE_INPUT_EMBEDS:
            input_embed = tl.load(
                cached_draft_input_embeds_ptr
                + history_slot * cached_draft_input_embeds_stride0
                + cache_read_slot * cached_draft_input_embeds_stride1
                + block,
                mask=mask,
            )
            tl.store(
                cached_draft_input_embeds_ptr
                + history_slot * cached_draft_input_embeds_stride0
                + cache_write_slot * cached_draft_input_embeds_stride1
                + block,
                input_embed,
                mask=mask,
            )
        hidden_state = tl.load(
            cached_target_hidden_states_ptr
            + history_slot * cached_target_hidden_states_stride0
            + cache_read_slot * cached_target_hidden_states_stride1
            + block,
            mask=mask,
        )
        tl.store(
            cached_target_hidden_states_ptr
            + history_slot * cached_target_hidden_states_stride0
            + cache_write_slot * cached_target_hidden_states_stride1
            + block,
            hidden_state,
            mask=mask,
        )

    for i in range(current_history_len):
        query_index = current_query_start + i
        cache_write_slot = current_cache_write_start + i
        if block_idx == 0:
            input_id = tl.load(draft_input_ids_ptr + query_index)
            tl.store(
                cached_draft_input_ids_ptr
                + history_slot * cached_draft_input_ids_stride0
                + cache_write_slot,
                input_id,
            )
        if USE_INPUT_EMBEDS:
            input_embed = tl.load(
                draft_input_embeds_ptr
                + query_index * draft_input_embeds_stride0
                + block,
                mask=mask,
            )
            tl.store(
                cached_draft_input_embeds_ptr
                + history_slot * cached_draft_input_embeds_stride0
                + cache_write_slot * cached_draft_input_embeds_stride1
                + block,
                input_embed,
                mask=mask,
            )
        hidden_state = tl.load(
            draft_hidden_states_ptr + query_index * draft_hidden_states_stride0 + block,
            mask=mask,
        )
        tl.store(
            cached_target_hidden_states_ptr
            + history_slot * cached_target_hidden_states_stride0
            + cache_write_slot * cached_target_hidden_states_stride1
            + block,
            hidden_state,
            mask=mask,
        )


@triton.jit
def _publish_step_mtp_history_lens_kernel(
    cached_history_lens_ptr,
    history_slot_mapping_ptr,
    last_token_indices_ptr,
    query_start_loc_ptr,
    history_window_size,
):
    req_idx = tl.program_id(0)
    history_slot = tl.load(history_slot_mapping_ptr + req_idx)
    if history_slot < 0:
        return
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    if query_end <= query_start:
        return

    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    valid_query_len = last_token_index - query_start + 1
    current_history_len = min(history_window_size, valid_query_len)
    old_history_len = tl.load(cached_history_lens_ptr + history_slot)
    old_history_to_keep = min(
        old_history_len,
        history_window_size - current_history_len,
    )
    tl.store(
        cached_history_lens_ptr + history_slot,
        old_history_to_keep + current_history_len,
    )


@triton.jit
def _shift_step_mtp_inputs_kernel(
    input_ids_ptr,
    input_embeds_ptr,
    input_embeds_stride0,
    next_input_ids_ptr,
    next_input_embeds_ptr,
    next_input_embeds_stride0,
    history_slot_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    req_idx = tl.program_id(0)
    hidden_block_idx = tl.program_id(1)
    dim_block = hidden_block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    history_slot = tl.load(history_slot_mapping_ptr + req_idx)
    if history_slot < 0:
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    if query_end <= query_start:
        return
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    if hidden_block_idx == 0:
        for i in range(1, query_len, BLOCK_SIZE_Q):
            query_block = i + tl.arange(0, BLOCK_SIZE_Q)
            query_mask = query_block < query_len
            input_ids = tl.load(
                input_ids_ptr + query_start + query_block, mask=query_mask
            )
            tl.store(
                input_ids_ptr + query_start + query_block - 1,
                input_ids,
                mask=query_mask,
            )
        next_input_id = tl.load(next_input_ids_ptr + req_idx)
        tl.store(input_ids_ptr + last_token_index, next_input_id)

    for i in range(1, query_len, BLOCK_SIZE_Q):
        query_block = i + tl.arange(0, BLOCK_SIZE_Q)
        query_mask = query_block < query_len
        mask = query_mask[:, None] & dim_mask[None, :]
        input_embed = tl.load(
            input_embeds_ptr
            + (query_start + query_block)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            mask=mask,
        )
        tl.store(
            input_embeds_ptr
            + (query_start + query_block - 1)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            input_embed,
            mask=mask,
        )
    next_input_embed = tl.load(
        next_input_embeds_ptr + req_idx * next_input_embeds_stride0 + dim_block,
        mask=dim_mask,
    )
    tl.store(
        input_embeds_ptr + last_token_index * input_embeds_stride0 + dim_block,
        next_input_embed,
        mask=dim_mask,
    )


@triton.jit
def _pad_step_mtp_trailing_slots_kernel(
    slot_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start = tl.load(last_token_indices_ptr + req_idx) + 1
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    for i in range(start, end, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        tl.store(slot_mapping_ptr + offsets, PAD_ID, mask=offsets < end)


class Step3p5MTPProposer(EagleProposer):
    """Step3.5 MTP proposer with per-layer draft-step selection."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Step MTP requires padded drafter batches so rejected-token "
                "history can be repaired. Please unset "
                "disable_padded_drafter_batch in the speculative config."
            )
        if self.speculative_config.num_speculative_tokens_per_batch_size is not None:
            raise NotImplementedError(
                "Step MTP does not support dynamic speculative token counts. "
                "Changing K skips later MTP modules and leaves their KV state "
                "stale. Remove num_speculative_tokens_per_batch_size."
            )
        self._per_group_block_tables: dict[int, BlockTable] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}

        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        max_speculative_tokens = self.speculative_config.num_speculative_tokens
        if max_speculative_tokens < 1:
            raise ValueError("Step MTP requires at least one speculative token.")
        self._history_window_size = max_speculative_tokens - 1
        history_buffer_width = max(1, self._history_window_size)
        if self.inputs_embeds_size != self.hidden_size:
            raise ValueError(
                "Step MTP rejection re-prefill requires input embeddings and "
                "hidden states to have the same width."
            )

        self.cached_draft_input_ids = torch.zeros(
            self.max_batch_size,
            history_buffer_width,
            dtype=torch.int32,
            device=device,
        )
        self.cached_draft_input_embeds: torch.Tensor | None = None
        if self.supports_mm_inputs:
            self.cached_draft_input_embeds = torch.zeros(
                self.max_batch_size,
                history_buffer_width,
                self.hidden_size,
                dtype=self.dtype,
                device=device,
            )
        self.cached_target_hidden_states = torch.zeros(
            self.max_batch_size,
            history_buffer_width,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.cached_history_lens = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self.last_token_indices = torch.zeros(
            self.max_batch_size, dtype=torch.int64, device=device
        )
        self._zero_num_rejected = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._next_input_ids = torch.zeros(
            self.max_batch_size, dtype=torch.int32, device=device
        )
        self._next_input_embeds = torch.zeros(
            self.max_batch_size,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

        self._history_slot_mapping = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
            device=device,
        )
        self._history_slot_mapping.cpu.fill_(-1)
        self._history_slot_mapping.gpu.fill_(-1)
        self._prefill_token_overrides = CpuGpuBuffer(
            self.max_batch_size,
            max_speculative_tokens,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
            device=device,
        )
        self._prefill_token_overrides.cpu.fill_(-1)
        self._prefill_token_overrides.gpu.fill_(-1)
        self._history_slot_by_req_id: dict[str, int] = {}
        self._free_history_slots = list(range(self.max_batch_size - 1, -1, -1))

    def _get_or_create_history_slot(self, req_id: str) -> int:
        slot = self._history_slot_by_req_id.get(req_id)
        if slot is not None:
            return slot
        if not self._free_history_slots:
            raise RuntimeError(
                "No free Step MTP history slots. Request history must be "
                "released on finish, preemption, or reset."
            )
        slot = self._free_history_slots.pop()
        self._history_slot_by_req_id[req_id] = slot
        self.cached_history_lens[slot].zero_()
        return slot

    def release_request_history(self, req_id: str) -> None:
        slot = self._history_slot_by_req_id.pop(req_id, None)
        if slot is None:
            return
        self.cached_history_lens[slot].zero_()
        self._free_history_slots.append(slot)

    def prepare_request_state(
        self,
        req_ids: list[str],
        requests: dict[str, "CachedRequestState"],
        num_scheduled_tokens: dict[str, int],
        discard_request_mask,
        num_rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build current-row mappings without tying history to batch rows."""
        num_reqs = len(req_ids)
        num_rows = num_reqs if num_rows is None else num_rows
        if not num_reqs <= num_rows <= self.max_batch_size:
            raise ValueError(
                "Step MTP request metadata rows must cover all requests and fit "
                f"the proposer buffers: requests={num_reqs}, rows={num_rows}, "
                f"capacity={self.max_batch_size}."
            )

        slot_mapping = self._history_slot_mapping.np
        overrides = self._prefill_token_overrides.np
        slot_mapping[:num_rows].fill(-1)
        overrides[:num_rows].fill(-1)

        validated_lookahead: list[list[int] | None] = [None] * num_reqs
        for req_index, req_id in enumerate(req_ids):
            num_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_scheduled == 0 or not bool(discard_request_mask[req_index]):
                continue

            request = requests[req_id]
            next_position = request.num_computed_tokens + num_scheduled
            lookahead_token_ids = []
            for draft_index in range(overrides.shape[1]):
                position = next_position + draft_index
                if self._lookahead_position_uses_embedding(request, position):
                    raise NotImplementedError(
                        "Step MTP chunked prefill cannot read ahead through "
                        f"multimodal embeddings at request {req_id!r}, "
                        f"position {position}."
                    )
                token_id = request.get_token_id(position)
                if token_id < 0:
                    raise NotImplementedError(
                        "Step MTP chunked prefill requires token IDs for the "
                        "full speculative lookahead."
                    )
                lookahead_token_ids.append(token_id)
            validated_lookahead[req_index] = lookahead_token_ids

        for req_index, req_id in enumerate(req_ids):
            if num_scheduled_tokens.get(req_id, 0) == 0:
                continue
            slot_mapping[req_index] = self._get_or_create_history_slot(req_id)
            validated_token_ids = validated_lookahead[req_index]
            if validated_token_ids is not None:
                overrides[req_index, : len(validated_token_ids)] = validated_token_ids

        self._history_slot_mapping.copy_to_gpu(num_rows)
        self._prefill_token_overrides.copy_to_gpu(num_rows)
        return (
            self._history_slot_mapping.gpu[:num_rows],
            self._prefill_token_overrides.gpu[:num_rows],
        )

    @staticmethod
    def _lookahead_position_uses_embedding(
        request: "CachedRequestState",
        position: int,
    ) -> bool:
        prompt_is_token_ids = getattr(request, "prompt_is_token_ids", None)
        num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
        if num_prompt_tokens is None:
            prompt_token_ids = getattr(request, "prompt_token_ids", None)
            prompt_embeds = getattr(request, "prompt_embeds", None)
            if prompt_token_ids is not None:
                num_prompt_tokens = len(prompt_token_ids)
            elif prompt_embeds is not None:
                num_prompt_tokens = len(prompt_embeds)

        if num_prompt_tokens is not None and 0 <= position < num_prompt_tokens:
            if prompt_is_token_ids is not None:
                return (
                    position >= len(prompt_is_token_ids)
                    or not prompt_is_token_ids[position]
                )
            if getattr(request, "prompt_token_ids", None) is None:
                return True

        for feature in getattr(request, "mm_features", ()):
            mm_position = feature.mm_position
            relative_position = position - mm_position.offset
            if not 0 <= relative_position < mm_position.length:
                continue
            if mm_position.is_embed is None:
                return True
            if bool(mm_position.is_embed[relative_position]):
                return True
        return False

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, "CachedRequestState"],
        gpu_input_batch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for req_index, req_id in enumerate(gpu_input_batch.req_ids):
            num_tokens = int(gpu_input_batch.num_tokens_no_spec[req_index])
            if num_tokens <= 0:
                continue
            terminal_position = num_tokens - 1
            request = requests[req_id]
            if self._lookahead_position_uses_embedding(request, terminal_position):
                raise NotImplementedError(
                    "Step MTP requires terminal token IDs for rejection repair "
                    f"but request {req_id!r} uses a prompt embedding at position "
                    f"{terminal_position}."
                )
        return super().prepare_next_token_ids_padded(
            sampled_token_ids,
            requests,
            gpu_input_batch,
            discard_request_mask,
        )

    def validate_current_query_mm_repair(
        self,
        req_ids: list[str],
        requests: dict[str, "CachedRequestState"],
        num_scheduled_tokens: dict[str, int],
        shift_computed_tokens: int = 1,
    ) -> None:
        """Reject MM rows whose embedding mask would need packed remapping."""
        for req_id in req_ids:
            num_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_scheduled <= 0:
                continue
            request = requests[req_id]
            start = request.num_computed_tokens + shift_computed_tokens
            for position in range(start, start + num_scheduled):
                if self._lookahead_position_uses_embedding(request, position):
                    raise NotImplementedError(
                        "Step MTP rejection repair cannot repack a current "
                        "multimodal embedding without remapping its packed "
                        f"mask: request {req_id!r}, position {position}."
                    )

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: "BlockTable",
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        # Bind each draft group to the runner-owned, address-stable slot
        # buffer before CUDA graph capture. The dummy path fills this buffer
        # with PAD slots; real Step MTP forwards update it in place.
        self._per_group_slot_mappings[gid] = block_table.slot_mapping.gpu

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-layer slot_mapping with one buffer per KV cache group."""
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            view = (
                self._slot_mapping_buffer[:num_tokens]
                if source is None
                else source[:num_tokens]
            )
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    def _prepare_packed_inputs(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        history_slot_mapping: torch.Tensor,
        prefill_token_overrides: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        if self.uses_mrope or target_positions.ndim != 1:
            raise NotImplementedError(
                "Step MTP rejection re-prefill currently requires 1D positions."
            )

        num_reqs = common_attn_metadata.batch_size()
        num_tokens = target_token_ids.shape[0]
        num_rejected = (
            self._zero_num_rejected[:num_reqs]
            if num_rejected_tokens_gpu is None
            else num_rejected_tokens_gpu
        )
        positions = (
            target_positions[0] if target_positions.ndim > 1 else target_positions
        )

        _prepare_step_mtp_input_buffers_kernel[(num_reqs,)](
            self.last_token_indices,
            self.input_ids,
            self.positions,
            common_attn_metadata.seq_lens,
            target_token_ids,
            positions,
            self.cached_draft_input_ids,
            self.cached_draft_input_ids.stride(0),
            self.cached_history_lens,
            history_slot_mapping,
            next_token_ids,
            prefill_token_overrides,
            prefill_token_overrides.stride(0),
            num_rejected,
            common_attn_metadata.query_start_loc,
            self._history_window_size,
            BLOCK_SIZE=1024,
        )

        common_attn_metadata.positions = self.positions[:num_tokens]
        if common_attn_metadata.dcp_local_seq_lens is not None:
            prepare_dcp_local_seq_lens(
                common_attn_metadata.dcp_local_seq_lens,
                common_attn_metadata.seq_lens,
                num_reqs,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
        return num_tokens, num_rejected

    def _cached_draft_input_embeds_strides(self) -> tuple[int, int]:
        if self.cached_draft_input_embeds is None:
            return 0, 0
        return (
            self.cached_draft_input_embeds.stride(0),
            self.cached_draft_input_embeds.stride(1),
        )

    def _prepare_packed_hidden_states(
        self,
        target_hidden_states: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor,
        history_slot_mapping: torch.Tensor,
    ) -> None:
        num_reqs = common_attn_metadata.batch_size()
        hidden_block_size = 256
        query_block_size = 16
        grid = (
            num_reqs,
            triton.cdiv(common_attn_metadata.max_query_len, query_block_size),
            triton.cdiv(self.hidden_size, hidden_block_size),
        )
        _prepare_step_mtp_hidden_states_kernel[grid](
            self.hidden_states,
            self.hidden_states.stride(0),
            target_hidden_states,
            target_hidden_states.stride(0),
            self.cached_target_hidden_states,
            self.cached_target_hidden_states.stride(0),
            self.cached_target_hidden_states.stride(1),
            self.inputs_embeds,
            self.inputs_embeds.stride(0),
            self.cached_draft_input_embeds,
            *self._cached_draft_input_embeds_strides(),
            self.cached_history_lens,
            history_slot_mapping,
            num_rejected_tokens_gpu,
            common_attn_metadata.query_start_loc,
            self._history_window_size,
            self.hidden_size,
            BLOCK_SIZE_Q=query_block_size,
            BLOCK_SIZE_H=hidden_block_size,
            USE_INPUT_EMBEDS=self.supports_mm_inputs,
        )

    def _cache_packed_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        history_slot_mapping: torch.Tensor,
    ) -> None:
        num_reqs = common_attn_metadata.batch_size()
        hidden_block_size = 256
        _cache_step_mtp_inputs_kernel[
            (num_reqs, triton.cdiv(self.hidden_size, hidden_block_size))
        ](
            self.input_ids,
            self.inputs_embeds,
            self.inputs_embeds.stride(0),
            self.hidden_states,
            self.hidden_states.stride(0),
            self.cached_draft_input_ids,
            self.cached_draft_input_ids.stride(0),
            self.cached_draft_input_embeds,
            *self._cached_draft_input_embeds_strides(),
            self.cached_target_hidden_states,
            self.cached_target_hidden_states.stride(0),
            self.cached_target_hidden_states.stride(1),
            self.cached_history_lens,
            history_slot_mapping,
            self.last_token_indices,
            common_attn_metadata.query_start_loc,
            self._history_window_size,
            self.hidden_size,
            BLOCK_SIZE=hidden_block_size,
            USE_INPUT_EMBEDS=self.supports_mm_inputs,
        )
        _publish_step_mtp_history_lens_kernel[(num_reqs,)](
            self.cached_history_lens,
            history_slot_mapping,
            self.last_token_indices,
            common_attn_metadata.query_start_loc,
            self._history_window_size,
        )

    def _rebuild_packed_slot_mappings(
        self,
        num_tokens: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        num_reqs = common_attn_metadata.batch_size()
        draft_gids = {
            attn_group.kv_cache_group_id for attn_group in self.draft_attn_groups
        }
        for gid in draft_gids:
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                raise RuntimeError(
                    f"Missing block table for Step MTP KV cache group {gid}."
                )
            block_table.compute_slot_mapping(
                num_reqs,
                common_attn_metadata.query_start_loc,
                self.positions[:num_tokens],
            )
            slot_mapping = block_table.slot_mapping.gpu
            _pad_step_mtp_trailing_slots_kernel[(num_reqs,)](
                slot_mapping,
                common_attn_metadata.query_start_loc,
                self.last_token_indices,
                PADDING_SLOT_ID,
                BLOCK_SIZE=256,
            )
            self._per_group_slot_mappings[gid] = slot_mapping

        common_attn_metadata.slot_mapping = self._per_group_slot_mappings[
            self.kv_cache_gid
        ][:num_tokens]

    def _shift_packed_inputs(
        self,
        draft_token_ids: torch.Tensor,
        prefill_token_overrides: torch.Tensor,
        override_index: int,
        common_attn_metadata: CommonAttentionMetadata,
        history_slot_mapping: torch.Tensor,
    ) -> None:
        num_rows = common_attn_metadata.batch_size()
        num_reqs = draft_token_ids.shape[0]
        overrides = prefill_token_overrides[:num_reqs, override_index]
        torch.where(
            overrides >= 0,
            overrides,
            draft_token_ids.to(torch.int32),
            out=self._next_input_ids[:num_reqs],
        )
        self._next_input_embeds[:num_reqs].copy_(
            self.model.embed_input_ids(self._next_input_ids[:num_reqs])
        )

        hidden_block_size = 256
        _shift_step_mtp_inputs_kernel[
            (num_rows, triton.cdiv(self.hidden_size, hidden_block_size))
        ](
            self.input_ids,
            self.inputs_embeds,
            self.inputs_embeds.stride(0),
            self._next_input_ids,
            self._next_input_embeds,
            self._next_input_embeds.stride(0),
            history_slot_mapping,
            common_attn_metadata.query_start_loc,
            self.last_token_indices,
            self.hidden_size,
            BLOCK_SIZE_Q=16,
            BLOCK_SIZE_H=hidden_block_size,
        )

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        # The proposer always works in unpadded shape.
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[
                    gid
                ].get_device_tensor(num_reqs)
                if gid in self._per_group_slot_mappings:
                    sm = self._per_group_slot_mappings[gid]
                    if sm.shape[0] >= num_actual_tokens:
                        sm = sm[:num_actual_tokens]
                    cm.slot_mapping = sm
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.build_metadata_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        """Step3.5 MTP uses the lm_head stored in each MTP layer."""
        # The base MTP path shares target lm_head into shared_head.head.
        # Step3.5 checkpoints carry per-MTP-layer shared_head weights.
        return

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            model_config=self.draft_model_config,
            quant_config=get_draft_quant_config(base),
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Step3.5 MTP draft layers may span multiple KV cache groups."""
        return

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer_name
                            and target_layer_name in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer_name
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[tuple[tuple[str, str], int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), gid)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = self.draft_attn_groups[0].kv_cache_spec.block_size
        else:
            self.kv_cache_gid = 0
        self.block_size = (
            int(kernel_block_sizes[self.kv_cache_gid])
            if kernel_block_sizes is not None
            and self.kv_cache_gid < len(kernel_block_sizes)
            else int(
                kv_cache_config.kv_cache_groups[
                    self.kv_cache_gid
                ].kv_cache_spec.block_size
            )
        )

    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            if self.use_local_argmax_reduction:
                return (
                    self.model.get_top_tokens(
                        hidden_states,
                        spec_step_idx=spec_step_idx,
                    ),
                    None,
                )
            logits = self.model.compute_logits(
                hidden_states, spec_step_idx=spec_step_idx
            )
            return logits.argmax(dim=-1), None

        logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
        return self._sample_from_logits(logits, sampling_metadata)

    def _prepare_text_model_inputs(
        self,
        num_input_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_ids = self.input_ids[:num_input_tokens]
        self.inputs_embeds[:num_input_tokens].copy_(
            self.model.embed_input_ids(input_ids)
        )
        return input_ids, self.inputs_embeds[:num_input_tokens]

    def propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
        history_slot_mapping: torch.Tensor | None = None,
        prefill_token_overrides: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch_size = next_token_ids.shape[0]
        if history_slot_mapping is None or prefill_token_overrides is None:
            raise ValueError(
                "Step MTP requires request-scoped history slots and prompt "
                "lookahead metadata from the model runner."
            )

        num_tokens, num_rejected_tokens_gpu = self._prepare_packed_inputs(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            next_token_ids=next_token_ids,
            common_attn_metadata=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            history_slot_mapping=history_slot_mapping,
            prefill_token_overrides=prefill_token_overrides,
        )
        self._rebuild_packed_slot_mappings(num_tokens, common_attn_metadata)

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        self._prepare_packed_hidden_states(
            target_hidden_states,
            common_attn_metadata,
            num_rejected_tokens_gpu,
            history_slot_mapping,
        )
        self._cache_packed_inputs(common_attn_metadata, history_slot_mapping)

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
        model_kwargs["spec_step_idx"] = 0

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_indices = self.last_token_indices[:batch_size]
        sample_hidden_states = last_hidden_states[sample_indices]

        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                dtype=torch.int64,
                device=last_hidden_states.device,
            )

        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                sample_hidden_states, sampling_metadata, spec_step_idx=0
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
            sample_hidden_states, sampling_metadata, spec_step_idx=0
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        draft_token_ids_list = [draft_token_ids]
        for spec_step_idx in range(1, self.num_speculative_tokens):
            self.hidden_states[:num_tokens].copy_(hidden_states[:num_tokens])
            self._shift_packed_inputs(
                draft_token_ids_list[-1],
                prefill_token_overrides,
                spec_step_idx,
                common_attn_metadata,
                history_slot_mapping,
            )
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index=spec_step_idx
            )

            if self.supports_mm_inputs:
                input_ids = None
            else:
                input_ids = self.input_ids[:num_input_tokens]

            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(num_input_tokens),
                "inputs_embeds": self.inputs_embeds[:num_input_tokens],
                "spec_step_idx": spec_step_idx,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(num_input_tokens),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                last_hidden_states[sample_indices],
                sampling_metadata,
                spec_step_idx=spec_step_idx,
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids
