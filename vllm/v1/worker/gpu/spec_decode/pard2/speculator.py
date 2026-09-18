# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PARD-2 speculator: a standalone parallel draft LM fed target hidden states.

PARD-2 (https://arxiv.org/abs/2605.08632) drafts all K tokens in one pass. Its
input for position t is ``embed(token[t]) + scale * target_proj(feat[t-1])``
with ``feat[-1] = 0``, so tokens are NOT shifted and the draft sees the whole
sequence starting at token 0.

That unshifted layout is what lets the step split into two forwards, each of
which fits an existing metadata builder:

  1. Context pass -- the tokens the target just confirmed. Same batch shape as
     the target, so its attention metadata and slot mappings are reused
     verbatim, as the autoregressive speculator does for EAGLE. This extends
     the draft's own KV cache.
  2. Draft pass -- K query rows per request (the sampled token followed by K-1
     mask tokens), uniform width, so it reuses DFlash's input-prep kernel and
     ``_build_uniform_attn_metadata``, and can be captured as a FULL graph.

MRv1 instead widens every query by K in one fused pass
(``extend_all_queries_by_N``), which has no MRv2 equivalent because a wider
batch is no longer uniform across requests.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.pard2.utils import load_pard2_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class Pard2Speculator(DraftModelSpeculator):
    _speculator_name = "PARD-2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # Fed to the draft as feat[t-1]; the draft projects it internally.
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        # feat[t-1] for the first row of a request's next span, which lives in
        # the previous step's target output and is otherwise unreachable.
        self.prev_last_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )

        self.supports_mm_inputs = False

        # The sampled token is itself a prediction, so K rows, not 1 + K.
        self.num_query_per_req = self.num_speculative_steps
        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        # -1 keeps padding rows from scattering into request slot 0 during capture.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        # Scratch for the context pass's feat[t-1] gather.
        self.shift_src = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # prepare_dflash_inputs writes these for DFlash's context-KV precompute,
        # which a standalone draft has no analogue for.
        self._unused_context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._unused_context_slots = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_ids: list[int] = []

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_pard2_model(target_model, self.vllm_config)

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
        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "%s draft attention (%s) does not support full CUDA graphs; "
                "running the draft eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        # Only the K-wide draft pass is uniform; the context pass runs eagerly.
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=True,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
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
            output = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=None,
            )
        # PARD-2 returns (logits_hidden, feedback_hidden) in fused mode.
        return output[0] if isinstance(output, tuple) else output

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 1,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _last_accepted_rows(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Row index of each request's last accepted token.

        The batch keeps the target's shape, so a request's span still carries
        its rejected positions as a tail; those rows are forwarded but must not
        be treated as real features.
        """
        return (
            input_batch.query_start_loc[1 : num_reqs + 1] - 1 - num_rejected[:num_reqs]
        ).to(torch.int64)

    def _fill_context_hidden_states(
        self,
        input_batch: InputBatch,
        target_hidden_states: torch.Tensor,
        last_accepted_rows: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        """Right-shift the target's hidden states by one within each request.

        Row t of the draft pairs token[t] with feat[t-1]. A request's first row
        takes feat from the previous step, or zero when the span starts at
        position 0 -- the row PARD-2 is trained to see and that EAGLE's left
        shift drops.
        """
        query_start_loc = input_batch.query_start_loc[:num_reqs]
        idx_mapping = input_batch.idx_mapping[:num_reqs]

        src = self.shift_src[:num_tokens]
        torch.arange(num_tokens, out=src)
        src -= 1
        # Mark each request's first row; -1 would wrap to the previous request.
        src.scatter_(0, query_start_loc.to(torch.int64), -1)
        is_span_start = src < 0
        src.clamp_(min=0)

        torch.index_select(
            target_hidden_states[:num_tokens],
            0,
            src,
            out=self.hidden_states[:num_tokens],
        )

        # A span starting at position 0 is a prefill: feat[-1] = 0.
        starts_at_zero = input_batch.positions[query_start_loc.to(torch.int64)] == 0
        carried = torch.where(
            starts_at_zero.unsqueeze(1),
            torch.zeros_like(self.prev_last_hidden[:num_reqs]),
            self.prev_last_hidden.index_select(0, idx_mapping.to(torch.int64)),
        )
        self.hidden_states[:num_tokens].masked_scatter_(
            is_span_start.unsqueeze(1), carried
        )

        # Carry this span's last accepted feature into the next step.
        self.prev_last_hidden.index_copy_(
            0,
            idx_mapping.to(torch.int64),
            target_hidden_states.index_select(0, last_accepted_rows),
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
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        if aux_hidden_states:
            target_hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            target_hidden_states = last_hidden_states

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        # --- Pass 1: absorb the confirmed tokens into the draft's KV cache. ---
        # Same shape as the target's batch, so its metadata and slot mappings
        # apply unchanged.
        last_accepted_rows = self._last_accepted_rows(
            input_batch, num_rejected, num_reqs
        )

        if not (dummy_run and skip_attn_for_dummy_run):
            self.input_buffers.input_ids[:num_target_tokens].copy_(
                self.target_input_buffers.input_ids[:num_target_tokens]
            )
            self.input_buffers.positions[:num_target_tokens].copy_(
                self.target_input_buffers.positions[:num_target_tokens]
            )
            self._fill_context_hidden_states(
                input_batch,
                target_hidden_states,
                last_accepted_rows,
                num_reqs,
                num_target_tokens,
            )
            self._prepare_eplb_forward(num_target_tokens)
            self._run_model(
                input_batch.num_tokens_after_padding,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )

        # --- Pass 2: K query rows per request, sampled all at once. ---
        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self._unused_context_positions,
                self._unused_context_slots,
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.block_tables.cp_rank,
                self.block_tables.cp_size,
                self.block_tables.cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                # The sampled token occupies row 0 and predicts the first draft
                # token, so every row is a prediction.
                True,
            )

        # Repeat-last-feat: no new real features exist past the context, so all K
        # rows reuse the last accepted one.
        self.hidden_states[:num_query_tokens].view(
            num_reqs, self.num_query_per_req, -1
        ).copy_(target_hidden_states.index_select(0, last_accepted_rows).unsqueeze(1))

        if dummy_run and skip_attn_for_dummy_run:
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, self.num_query_per_req)
            if dp_sync is not None
            else (None, num_query_tokens)
        )
        batch_desc, batch_sync = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=batch_sync,
        )
        num_tokens_padded = batch_desc.num_tokens
        num_tokens_across_dp = (
            batch_sync.num_tokens_across_dp if batch_sync is not None else None
        )

        draft_attn_metadata = self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            num_query_per_req=self.num_query_per_req,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]
