# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    index_quantized_activation,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import get_uniform_decode_token_count

logger = init_logger(__name__)


class MultiModuleMTPSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        self.inputs_embeds: torch.Tensor | None = None

        # Input id overrides for the last num_speculative_steps - 1 draft steps.
        # Used by chunked-prefilling requests to swap in the future prefill token
        # for the sampled draft token. Non-chunked-prefilling requests fill the -1
        # value, indicating that the sampled draft token should be used.
        self.draft_input_id_overrides = torch.full(
            (self.max_num_reqs, self.num_speculative_steps - 1),
            -1,
            dtype=torch.int64,
            device=device,
        )

        # Cached input ids, embeddings, and target hidden states from the last
        # decode step. Used to re-prefill tokens to update stale KV cache slots
        # in later MTP modules.
        self.cached_draft_input_ids = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            dtype=torch.int64,
            device=self.device,
        )
        self.cached_draft_input_embeds: torch.Tensor | None = None
        self.cached_target_hidden_states = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        self.cudagraph_manager: SpeculatorCudaGraphManager | None = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not self.supports_mm_inputs:
            return

        self.inputs_embeds = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.cached_draft_input_embeds = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # TODO(TheEpicDolphin): Support piecewise cudagraph for multi-module MTP.
        if cudagraph_mode.has_piecewise_cudagraphs():
            cudagraph_mode = (
                CUDAGraphMode.FULL_DECODE_ONLY
                if cudagraph_mode.has_full_cudagraphs()
                else CUDAGraphMode.NONE
            )
        self.cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

    def capture(self) -> None:
        logger.info("Capturing model for multi-module MTP speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        assert self.cudagraph_manager is not None
        if self.cudagraph_manager.use_breakable_cg:
            self.cudagraph_manager.init_breakable_cg_runner(self.model)
        self.cudagraph_manager.capture(
            self._generate_drafts,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing multi-module MTP CUDA graphs",
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(max_seq_len, self.max_model_len)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        num_tokens = input_batch.num_tokens
        max_query_len = input_batch.num_scheduled_tokens.max()
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            input_batch.query_start_loc[: num_reqs + 1]
        )

        self._prepare_inputs(
            last_hidden_states,
            input_batch,
            num_tokens,
            max_query_len,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            mm_inputs,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay.
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs, num_tokens, max_query_len, input_batch.has_prefill
        )
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            input_batch.num_tokens_after_padding,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        # Rebuild the slot mappings and attention metadata.
        skip_attn = dummy_run and skip_attn_for_dummy_run
        if not skip_attn:
            # Build the slot mappings and attention metadata.
            slot_mappings_tensor = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc,
                self.input_buffers.positions,
                batch_desc.num_tokens,
            )
            # Apply padding values to slots not corresponding to real draft
            # tokens to prevent stale value writes.
            pad_trailing_draft_slots(
                slot_mappings_tensor,
                self.input_buffers.query_start_loc,
                self.last_token_indices[:num_reqs],
                num_reqs,
            )
            slot_mappings = build_slot_mappings_by_layer(
                slot_mappings_tensor, self.kv_cache_config
            )
            draft_attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
                num_tokens_padded=batch_desc.num_tokens,
                query_start_loc_np=input_batch.query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=0,
            )
            assert draft_attn_metadata is not None
            attn_metadata = draft_attn_metadata

        self._prepare_eplb_forward(num_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_drafts(
                num_reqs,
                batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        spec_module_idx: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=(
                    self.inputs_embeds[:num_tokens]
                    if self.inputs_embeds is not None
                    else None
                ),
                spec_step_idx=spec_module_idx,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # PIECEWISE cudagraph (compiled PW or breakable), chosen inside
                # run_pw_graph.
                assert self.cudagraph_manager is not None
                ret_hidden_states = self.cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _prepare_inputs(
        self,
        target_hidden_states: torch.Tensor,
        input_batch: InputBatch,
        num_tokens: int,
        max_query_len: int,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        num_reqs = input_batch.num_reqs
        prepare_input_buffers(
            num_reqs,
            input_batch,
            self.cached_draft_input_ids,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.draft_input_id_overrides,
            self.input_buffers,
            self.last_token_indices,
            self.max_num_reqs,
            self.num_speculative_steps,
        )

        # Compute the input embeddings with the MM embeddings merged in.
        # TODO(TheEpicDolphin): When the batch has no MM content (is_mm_embed
        # all False), skip this embed/copy and run the model with
        # inputs_embeds=None. Requirements:
        # 1. Eager steps can switch freely, but FULL cudagraphs bake the
        # embeds/no-embeds path at capture, so extending to decode steps
        # needs a uses_input_embeds property in the graph descriptors,
        # resulting in 2x captures.
        # 2. The skip condition must also verify the request's cached
        # re-prefill window (cached_draft_input_embeds) holds no MM
        # embeddings, or the rejection re-prefill gap would be re-embedded
        # from token ids incorrectly.
        if self.inputs_embeds is not None:
            mm_embeds, is_mm_embed = mm_inputs or (None, None)
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_buffers.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

        prepare_input_hidden_states_and_embeddings(
            num_reqs,
            max_query_len,
            self.hidden_states,
            target_hidden_states,
            self.cached_target_hidden_states,
            self.inputs_embeds,
            self.cached_draft_input_embeds,
            input_batch,
            self.input_buffers,
            num_rejected,
            self.num_speculative_steps,
        )

    def _generate_drafts(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        sample_positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        # Cache the trailing token's ids, hidden states (and embeddings for
        # MM models), which are needed if the trailing tokens are re-prefilled
        # during the next decode step.
        cache_inputs(
            self.input_buffers,
            self.inputs_embeds,
            self.hidden_states,
            self.cached_draft_input_ids,
            self.cached_draft_input_embeds,
            self.cached_target_hidden_states,
            last_token_indices,
            idx_mapping,
            num_reqs,
            self.num_speculative_steps,
            use_input_embeds=self.inputs_embeds is not None,
        )

        for step in range(self.num_speculative_steps):
            # Update the current draft step.
            self.current_draft_step.fill_(step)

            # Run the model forward pass.
            last_hidden_states, hidden_states = self._run_model(
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                spec_module_idx=step,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            )

            # Sample draft tokens for the current step.
            sample_hidden_states = index_quantized_activation(
                last_hidden_states, last_token_indices
            )
            draft_tokens = self.sample_draft(
                sample_hidden_states,
                sample_positions,
                idx_mapping,
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
            )

            self.draft_tokens[:num_reqs, step] = draft_tokens
            if step < self.num_speculative_steps - 1:
                self.hidden_states[:num_tokens] = hidden_states
                # Mid-prefill requests append the known future prefill token
                # instead of the sampled draft.
                overrides = self.draft_input_id_overrides[:num_reqs, step]
                next_input_tokens = torch.where(overrides >= 0, overrides, draft_tokens)
                # Shift the draft inputs left by one and append the next
                # input token id/embeddings.
                draft_embeds = (
                    self.model.embed_input_ids(next_input_tokens)
                    if self.inputs_embeds is not None
                    else None
                )
                update_draft_inputs(
                    next_input_tokens,
                    draft_embeds,
                    self.input_buffers,
                    self.inputs_embeds,
                    last_token_indices,
                    idx_mapping,
                    num_reqs,
                )
                sample_positions += 1


@triton.jit
def _prepare_input_buffers_kernel(
    last_token_indices_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    draft_input_id_overrides_ptr,
    draft_input_id_overrides_stride0,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride0,
    num_sampled_ptr,
    num_rejected_ptr,
    target_seq_lens_ptr,
    query_start_loc_ptr,
    max_num_reqs,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(target_seq_lens_ptr + req_idx)

    # Get the number of rejected tokens, and the number of trailing tokens from
    # the last decode step that need to be re-prefilled to update the stale
    # KV cache slots in later MTP modules.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill_tokens = max(0, num_rejected - 1)

    # The rejected tokens are excluded as inputs to the draft model.
    num_input_tokens = query_len - num_rejected
    # Re-prefilled tokens are packed into the query window without widening
    # it, so the effective KV length shrinks by num_reprefill_tokens.
    # Example (r = rejected draft token, x = unused trailing slot, later padded)
    # re-prefilling t1 and t2 from the last decode step:
    #   t0 t1 t2 [ t3 t4 r r r ] => t0 [ t1 t2 t3 t4 . ]
    seq_len -= num_reprefill_tokens

    # Write the updated sequence length.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)

    # Get the next draft input token.
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefill. Seed with the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # For chunked-prefilling requests, copy the future prefill tokens that
    # will later override the input token ids for the last
    # num_speculative_steps - 1 draft steps.
    for i in range(1, num_speculative_steps):
        future_token = tl.load(
            next_prefill_tokens_ptr + i * next_prefill_tokens_stride0 + req_state_idx
        )
        override = tl.where(num_sampled > 0, -1, future_token).to(tl.int64)
        tl.store(
            draft_input_id_overrides_ptr
            + req_idx * draft_input_id_overrides_stride0
            + i
            - 1,
            override,
        )

    # Copy the target's input ids (read at the target offset) shifted left by 1,
    # and right by the number of re-prefills, into the draft buffer.
    for i in range(1, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(
            draft_input_ids_ptr + query_start + num_reprefill_tokens - 1 + block,
            input_ids,
            mask=mask,
        )
    last_token_index = query_start + num_reprefill_tokens + num_input_tokens - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy the target's positions, shifted over by the number of tokens to be
    # re-prefilled.
    for i in range(0, num_input_tokens, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_input_tokens
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(
            draft_positions_ptr + query_start + num_reprefill_tokens + block,
            target_pos,
            mask=mask,
        )

    # Fill the re-prefill gap with the cached token ids from the previous
    # decode step. These tokens sit immediately before the query's first
    # token, so their positions are contiguous and derived here.
    first_position = tl.load(target_positions_ptr + query_start)
    for i in range(num_reprefill_tokens):
        cache_read_slot = num_speculative_steps - 1 - num_reprefill_tokens + i
        cached_token_id = tl.load(
            cached_draft_input_ids_ptr
            + req_state_idx * cached_draft_input_ids_stride0
            + cache_read_slot
        )
        tl.store(draft_input_ids_ptr + query_start + i, cached_token_id)
        tl.store(
            draft_positions_ptr + query_start + i,
            first_position - num_reprefill_tokens + i,
        )

    if req_idx == (num_reqs - 1):
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(query_start_loc_ptr + block, query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_input_buffers(
    num_reqs: int,
    input_batch: InputBatch,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [num_prefill_lookahead, max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1]
    draft_input_id_overrides: torch.Tensor,
    input_buffers: InputBuffers,
    # [max_num_reqs]
    last_token_indices: torch.Tensor,
    max_num_reqs: int,
    num_speculative_steps: int,
) -> None:
    _prepare_input_buffers_kernel[(num_reqs,)](
        last_token_indices,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0) if cached_draft_input_ids is not None else 0,
        draft_input_id_overrides,
        draft_input_id_overrides.stride(0),
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        next_prefill_tokens.stride(0),
        num_sampled,
        num_rejected,
        input_batch.seq_lens,
        input_buffers.query_start_loc,
        max_num_reqs,
        num_speculative_steps,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_input_hidden_states_and_embeddings_kernel(
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    target_hidden_states_ptr,
    target_hidden_states_stride0,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    input_embeds_ptr,
    input_embeds_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    idx_mapping_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    num_speculative_steps,
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

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    # Get the number of rejected tokens, and the number of trailing tokens from
    # the last decode step that need to be re-prefilled to update the stale
    # KV cache slots in later MTP modules.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefill_hidden_states = max(0, num_rejected - 1)

    # The rejected tokens are excluded as inputs to the draft model.
    num_input_hidden_states = query_len - num_rejected

    # Copy the output target hidden states as inputs to the first MTP module,
    # shifted right by the re-prefill gap. Each program copies one
    # (BLOCK_SIZE_Q, BLOCK_SIZE_H) tile of this request's query.
    query_block = query_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    query_mask = (query_block < num_input_hidden_states)[:, None] & dim_mask[None, :]
    hidden_state = tl.load(
        target_hidden_states_ptr
        + (query_start + query_block)[:, None] * target_hidden_states_stride0
        + dim_block[None, :],
        mask=query_mask,
    )
    tl.store(
        draft_input_hidden_states_ptr
        + (query_start + num_reprefill_hidden_states + query_block)[:, None]
        * draft_input_hidden_states_stride0
        + dim_block[None, :],
        hidden_state,
        mask=query_mask,
    )

    if query_block_idx == 0:
        # Fill the re-prefill gap with the cached hidden states and embeddings
        # (for MM models) from the previous decode step, mirroring the token
        # ids and positions inserted by _prepare_input_buffers_kernel. The gap
        # is at most num_speculative_steps - 1 tokens, so a serial loop in the
        # first query block suffices.
        for i in range(num_reprefill_hidden_states):
            cache_read_slot = (
                num_speculative_steps - 1 - num_reprefill_hidden_states + i
            )
            cached_hidden_state = tl.load(
                cached_target_hidden_states_ptr
                + req_state_idx * cached_target_hidden_states_stride0
                + cache_read_slot * cached_target_hidden_states_stride1
                + dim_block,
                mask=dim_mask,
            )
            tl.store(
                draft_input_hidden_states_ptr
                + (query_start + i) * draft_input_hidden_states_stride0
                + dim_block,
                cached_hidden_state,
                mask=dim_mask,
            )
            if USE_INPUT_EMBEDS:
                cached_embed = tl.load(
                    cached_draft_input_embeds_ptr
                    + req_state_idx * cached_draft_input_embeds_stride0
                    + cache_read_slot * cached_draft_input_embeds_stride1
                    + dim_block,
                    mask=dim_mask,
                )
                tl.store(
                    input_embeds_ptr
                    + (query_start + i) * input_embeds_stride0
                    + dim_block,
                    cached_embed,
                    mask=dim_mask,
                )


def prepare_input_hidden_states_and_embeddings(
    num_reqs: int,
    # Upper bound on the draft query length of any request in the batch.
    max_query_len: int,
    # [num_tokens, hidden_size]
    hidden_states: torch.Tensor,
    # [num_tokens, hidden_size]
    target_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor | None,
    # [num_tokens, hidden_size]
    input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    input_batch: InputBatch,
    input_buffers: InputBuffers,
    # [num_reqs]
    num_rejected: torch.Tensor,
    num_speculative_steps: int,
) -> None:
    use_input_embeds = input_embeds is not None
    hidden_size = target_hidden_states.shape[-1]
    query_block_size = 16
    hidden_block_size = 256
    grid = (
        num_reqs,
        triton.cdiv(max_query_len, query_block_size),
        triton.cdiv(hidden_size, hidden_block_size),
    )
    _prepare_input_hidden_states_and_embeddings_kernel[grid](
        hidden_states,
        hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0)
        if cached_target_hidden_states is not None
        else 0,
        cached_target_hidden_states.stride(1)
        if cached_target_hidden_states is not None
        else 0,
        input_embeds,
        input_embeds.stride(0) if input_embeds is not None else 0,
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        input_batch.idx_mapping,
        num_rejected,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE_Q=query_block_size,
        BLOCK_SIZE_H=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


@triton.jit
def _pad_trailing_draft_slots_kernel(
    slot_mappings_ptr,
    slot_mappings_stride0,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    # Slots computed from stale token positions in the range
    # [last_token_index + 1, query_end) can result in writes to blocks.
    # Pad these slot values so that attention kernels ignore them.
    start = tl.load(last_token_indices_ptr + req_idx) + 1
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    base = slot_mappings_ptr + group_idx * slot_mappings_stride0
    for i in range(start, end, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        tl.store(base + offs, PAD_ID, mask=mask)


def pad_trailing_draft_slots(
    # [num_groups, num_tokens_padded]
    slot_mappings: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    num_reqs: int,
) -> None:
    num_groups = slot_mappings.shape[0]
    _pad_trailing_draft_slots_kernel[(num_groups, num_reqs)](
        slot_mappings,
        slot_mappings.stride(0),
        query_start_loc,
        last_token_indices,
        PAD_SLOT_ID,
        BLOCK_SIZE=256,
    )


@triton.jit
def _cache_inputs_kernel(
    draft_input_ids_ptr,
    draft_input_embeds_ptr,
    draft_input_embeds_stride0,
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    idx_mapping_ptr,
    last_token_indices_ptr,
    query_start_loc_ptr,
    num_speculative_steps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < hidden_size

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)

    # Snapshot the last num_speculative_steps - 1 input draft token ids/hidden
    # states and embeddings (for MM models), indexed by request state. These
    # may be needed to re-prefill the tokens during the next decode step.
    cache_window_size = num_speculative_steps - 1
    window_start = last_token_index - cache_window_size + 1
    for i in range(max(window_start, query_start), last_token_index + 1):
        cache_write_slot = i - window_start
        if block_idx == 0:
            input_id = tl.load(draft_input_ids_ptr + i)
            tl.store(
                cached_draft_input_ids_ptr
                + req_state_idx * cached_draft_input_ids_stride0
                + cache_write_slot,
                input_id,
            )
        if USE_INPUT_EMBEDS:
            input_embeds = tl.load(
                draft_input_embeds_ptr + i * draft_input_embeds_stride0 + block,
                mask=mask,
            )
            tl.store(
                cached_draft_input_embeds_ptr
                + req_state_idx * cached_draft_input_embeds_stride0
                + cache_write_slot * cached_draft_input_embeds_stride1
                + block,
                input_embeds,
                mask=mask,
            )
        hidden_state = tl.load(
            draft_input_hidden_states_ptr
            + i * draft_input_hidden_states_stride0
            + block,
            mask=mask,
        )
        tl.store(
            cached_target_hidden_states_ptr
            + req_state_idx * cached_target_hidden_states_stride0
            + cache_write_slot * cached_target_hidden_states_stride1
            + block,
            hidden_state,
            mask=mask,
        )


def cache_inputs(
    input_buffers: InputBuffers,
    # [num_tokens, hidden_size]
    draft_input_embeds: torch.Tensor | None,
    # [num_tokens, hidden_size]
    draft_input_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_speculative_steps: int,
    use_input_embeds: bool,
) -> None:
    hidden_size = draft_input_hidden_states.shape[-1]
    hidden_block_size = 1024
    _cache_inputs_kernel[(num_reqs, triton.cdiv(hidden_size, hidden_block_size))](
        input_buffers.input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0) if draft_input_embeds is not None else 0,
        draft_input_hidden_states,
        draft_input_hidden_states.stride(0),
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0),
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0),
        cached_target_hidden_states.stride(1),
        idx_mapping,
        last_token_indices,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


@triton.jit
def _shift_input_ids_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    draft_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    # Use the post-rejection last token index so the shift and insertion align
    # with the position the draft token was sampled from.
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # Shift input token ids to the left by one position and
    # insert the last sampled draft token.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(input_ids_ptr + query_start + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + last_token_index, draft_token)


@triton.jit
def _shift_input_embeds_kernel(
    input_embeds_ptr,
    input_embeds_stride0,
    draft_embeds_ptr,
    draft_embeds_stride0,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    block_idx = tl.program_id(1)
    dim_block = block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # Shift input token embeddings to the left by one position and
    # insert the last sampled draft token's embeddings.
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
    draft_embed = tl.load(
        draft_embeds_ptr + req_idx * draft_embeds_stride0 + dim_block,
        mask=dim_mask,
    )
    tl.store(
        input_embeds_ptr + last_token_index * input_embeds_stride0 + dim_block,
        draft_embed,
        mask=dim_mask,
    )


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    draft_embeds: torch.Tensor | None,
    input_buffers: InputBuffers,
    input_embeds: torch.Tensor | None,
    last_token_indices: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
) -> None:
    _shift_input_ids_kernel[(num_reqs,)](
        input_buffers.input_ids,
        idx_mapping,
        input_buffers.query_start_loc,
        last_token_indices,
        draft_tokens,
        BLOCK_SIZE=1024,
    )
    if input_embeds is not None:
        assert draft_embeds is not None
        hidden_size = input_embeds.shape[-1]
        hidden_block_size = 256
        _shift_input_embeds_kernel[
            (num_reqs, triton.cdiv(hidden_size, hidden_block_size))
        ](
            input_embeds,
            input_embeds.stride(0),
            draft_embeds,
            draft_embeds.stride(0),
            idx_mapping,
            input_buffers.query_start_loc,
            last_token_indices,
            hidden_size,
            BLOCK_SIZE_Q=16,
            BLOCK_SIZE_H=hidden_block_size,
        )
