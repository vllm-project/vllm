# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables, _compute_slot_mappings_kernel
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class PEagleSpeculator(DraftModelSpeculator):
    """PEagle (Parallel Eagle) speculator.

    all N draft tokens are produced in one forward pass by prepending N-1 pard
    tokens to each request's Eagle input, instead of running N sequential decode steps.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.supports_mm_inputs = False

        self._init_parallel_drafting_params()

        N = self.num_speculative_steps
        max_extended = self.max_num_tokens + self.max_num_reqs * (N - 1)
        self.peagle_max_extended = max_extended
        self.peagle_input_ids = torch.zeros(
            max_extended, dtype=torch.int32, device=device
        )
        self.peagle_positions = torch.zeros(
            max_extended, dtype=torch.int64, device=device
        )
        self.peagle_hidden_states = torch.zeros(
            (max_extended, self.hidden_size), dtype=self.dtype, device=device
        )
        self.peagle_query_start_loc = torch.zeros(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.peagle_seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.peagle_is_hidden_masked = torch.zeros(
            max_extended, dtype=torch.bool, device=device
        )
        self.peagle_eagle_idx = torch.zeros(
            max_extended, dtype=torch.int32, device=device
        )
        self.peagle_slot_mappings: torch.Tensor | None = None
        self.peagle_is_stale_mask = torch.zeros(
            max_extended, dtype=torch.bool, device=device
        )

    def _init_parallel_drafting_params(self) -> None:
        model_hf_config = self.draft_model_config.hf_config
        if hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For PEagle (parallel drafting), the draft model config must "
                "have `pard_token` or `ptd_token_id` in its config.json."
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self, attn_states) -> None:
        pass  # TODO: add CudaGraph support

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if not hasattr(self.model, "mask_hidden"):
            self.parallel_drafting_hidden_state_tensor = torch.zeros(
                self.hidden_size, dtype=self.dtype, device=self.device
            )
            return
        flat_mask = self.model.mask_hidden.view(-1)
        if hasattr(self.model, "combine_hidden_states"):
            projected = self.model.combine_hidden_states(flat_mask)
        else:
            projected = flat_mask
        self.parallel_drafting_hidden_state_tensor = projected.detach().clone()

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

        return load_eagle_model(target_model, self.vllm_config)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        num_kv_groups = len(kv_cache_config.kv_cache_groups)
        self.peagle_slot_mappings = torch.zeros(
            (num_kv_groups, self.peagle_max_extended),
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def model_returns_tuple(self) -> bool:
        """
        Whether the draft model's forward() returns a tuple.

        True: returns (last_hidden_states, hidden_states) — Eagle, Gemma4 MTP.
        False: returns a single tensor used for both — standard MTP (DeepSeek).
        """
        return True

    def _build_peagle_attn_metadata(
        self,
        num_reqs: int,
        num_extended: int,
        max_query_len_extended: int,
        block_tables: list[torch.Tensor],
        extended_slot_mappings: torch.Tensor,
        peagle_qsl_cpu: torch.Tensor,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_extended,
            query_start_loc_gpu=self.peagle_query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=peagle_qsl_cpu,
            max_query_len=max_query_len_extended,
            seq_lens=self.peagle_seq_lens[:num_reqs],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=extended_slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            inputs_embeds = None
            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
            ret_hidden_states = self.model(**model_inputs)
        if self.model_returns_tuple:
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _prepare_extended_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
        num_rejected: torch.Tensor,
    ) -> int:
        """Gather hidden states into peagle buffer. Returns num_extended.

        The peagle_* input/position/mask/idx buffers are already populated by
        prepare_peagle_inputs (called in propose), so this method only handles
        the hidden-state gather that requires Python-side tensor ops.
        """
        N = self.num_speculative_steps
        # Each request gains N-1 pard slots on top of its Eagle query length.
        num_extended = num_tokens + num_reqs * (N - 1)

        assert self.parallel_drafting_hidden_state_tensor is not None
        mask_h = self.parallel_drafting_hidden_state_tensor
        eagle_idx = self.peagle_eagle_idx[:num_extended].long()
        is_hidden_masked = self.peagle_is_hidden_masked[:num_extended]
        self.peagle_hidden_states[:num_extended] = self.hidden_states[eagle_idx]
        torch.where(
            is_hidden_masked.unsqueeze(1),
            mask_h,
            self.peagle_hidden_states[:num_extended],
            out=self.peagle_hidden_states[:num_extended],
        )
        return num_extended

    def _build_extended_context(
        self,
        num_reqs: int,
        num_extended: int,
        eagle_query_start_loc_np: np.ndarray,
        skip_attn: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor] | None]:
        """Compute slot mappings and attention metadata for the extended batch."""
        if skip_attn:
            return None, None

        N = self.num_speculative_steps
        assert self.peagle_slot_mappings is not None
        num_kv_groups = self.peagle_slot_mappings.shape[0]
        _compute_slot_mappings_kernel[(num_kv_groups, num_reqs + 1)](
            num_extended,
            self.idx_mapping[:num_reqs],
            self.peagle_query_start_loc[: num_reqs + 1],
            self.peagle_positions,
            self.block_tables.block_table_ptrs,
            self.block_tables.block_table_strides,
            self.block_tables.block_sizes_tensor,
            self.peagle_slot_mappings,
            self.peagle_slot_mappings.stride(0),
            self.block_tables.cp_rank,
            CP_SIZE=self.block_tables.cp_size,
            CP_INTERLEAVE=self.block_tables.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,
        )
        # Suppress KV writes for stale positions so they don't corrupt the KV cache.
        stale_mask = self.peagle_is_stale_mask[:num_extended]
        self.peagle_slot_mappings[:, :num_extended].masked_fill_(
            stale_mask.unsqueeze(0), PAD_SLOT_ID
        )
        extended_slot_mappings = self.peagle_slot_mappings[:, :num_extended]
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            extended_slot_mappings, self.kv_cache_config
        )

        # Compute query_start_loc on CPU to avoid GPU sync:
        # peagle_qsl[i] = eagle_qsl[i] + i*(N-1).
        eagle_qsl_np = eagle_query_start_loc_np[: num_reqs + 1]
        offsets = np.arange(num_reqs + 1, dtype=np.int32) * (N - 1)
        peagle_qsl_np = eagle_qsl_np + offsets
        peagle_qsl_cpu = torch.from_numpy(peagle_qsl_np)
        max_query_len_extended = (
            int((peagle_qsl_np[1:] - peagle_qsl_np[:-1]).max()) if num_reqs > 0 else 1
        )
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]
        attn_metadata = self._build_peagle_attn_metadata(
            num_reqs=num_reqs,
            num_extended=num_extended,
            max_query_len_extended=max_query_len_extended,
            block_tables=block_tables,
            extended_slot_mappings=extended_slot_mappings,
            peagle_qsl_cpu=peagle_qsl_cpu,
        )
        return attn_metadata, slot_mappings_by_layer

    def _run_peagle_model(
        self,
        num_extended: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings_by_layer: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the model forward pass on the extended batch."""
        batch_descriptor = BatchDescriptor(num_tokens=num_extended)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_extended,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings_by_layer,
            batch_descriptor=batch_descriptor,
        ):
            ret_hidden_states = self.model(
                input_ids=self.peagle_input_ids[:num_extended],
                positions=self.peagle_positions[:num_extended],
                hidden_states=self.peagle_hidden_states[:num_extended],
            )
        if self.model_returns_tuple:
            last_hidden_states, _ = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
        return last_hidden_states

    def _sample_all_tokens(
        self,
        num_reqs: int,
        last_hidden_states: torch.Tensor,
    ) -> None:
        """Sample all N draft tokens from the extended forward pass output."""
        N = self.num_speculative_steps
        last_token_indices = self.last_token_indices[:num_reqs]
        first_sample_positions = self.peagle_positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        for step in range(N):
            self.current_draft_step.fill_(step)
            sample_hidden_states = last_hidden_states[last_token_indices + step]
            self.draft_tokens[:num_reqs, step] = self.sample_draft(
                sample_hidden_states,
                first_sample_positions + step,
                idx_mapping,
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
            )

    def _parallel_eagle(
        self,
        num_reqs: int,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None,
        eagle_query_start_loc_np: np.ndarray,
        num_rejected: torch.Tensor,
        skip_attn: bool = False,
        is_profile: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Single-pass forward: generates all N draft tokens at once.

        Extends each request's Eagle input by N-1 pard tokens, runs one
        forward pass, and samples all N draft tokens simultaneously.
        """
        # During profile_run, skip the N-1 pard extension to keep num_tokens
        # within the model's compile range (1, max_num_batched_tokens).
        if is_profile:
            self._run_model(
                num_tokens,
                None,
                None,
                num_tokens_across_dp,
                CUDAGraphMode.NONE,
                mm_inputs,
            )
            return

        num_extended = self._prepare_extended_inputs(num_reqs, num_tokens, num_rejected)
        attn_metadata, slot_mappings_by_layer = self._build_extended_context(
            num_reqs,
            num_extended,
            eagle_query_start_loc_np,
            skip_attn,
        )
        last_hidden_states = self._run_peagle_model(
            num_extended,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
        )
        self._sample_all_tokens(num_reqs, last_hidden_states)

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
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        if aux_hidden_states:
            assert self.method == "eagle3"
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_tokens].copy_(hidden_states)

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        prepare_peagle_inputs(
            out_input_ids=self.peagle_input_ids,
            out_positions=self.peagle_positions,
            out_query_start_loc=self.peagle_query_start_loc,
            out_seq_lens=self.peagle_seq_lens,
            out_is_hidden_masked=self.peagle_is_hidden_masked,
            out_is_stale_mask=self.peagle_is_stale_mask,
            out_eagle_idx=self.peagle_eagle_idx,
            last_token_indices=self.last_token_indices,
            current_draft_step=self.current_draft_step,
            input_batch=input_batch,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            pard_token_id=self.parallel_drafting_token_id,
            num_speculative_steps=self.num_speculative_steps,
        )

        self._parallel_eagle(
            num_reqs,
            input_batch.num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            eagle_query_start_loc_np=input_batch.query_start_loc_np,
            num_rejected=num_rejected,
            skip_attn=dummy_run and skip_attn_for_dummy_run,
            is_profile=is_profile,
            mm_inputs=mm_inputs,
        )
        return self.draft_tokens[:num_reqs]


# ---------------------------------------------------------------------------
# PEagle (Parallel Eagle) helpers
# ---------------------------------------------------------------------------
@triton.jit
def _prepare_peagle_inputs_kernel(
    # ── Outputs: PEagle extended buffer ──────────────────────────────────
    out_input_ids_ptr,  # [max_extended]
    out_positions_ptr,  # [max_extended]
    out_query_start_loc_ptr,  # [max_num_reqs + 1]
    out_seq_lens_ptr,  # [max_num_reqs]
    out_is_hidden_masked_ptr,  # [max_extended] bool: is_pard | is_stale
    out_is_stale_mask_ptr,  # [max_extended] bool: stale only
    out_eagle_idx_ptr,  # [max_extended] int32: valid eagle pos → target pos
    last_token_indices_ptr,  # [max_num_reqs]  (output)
    draft_current_step_ptr,  # scalar           (output: reset to 0)
    # ── Inputs: directly from target model / input_batch ─────────────────
    target_input_ids_ptr,
    target_positions_ptr,
    target_query_start_loc_ptr,
    target_seq_lens_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    pard_token_id,
    num_speculative_steps,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    N = num_speculative_steps

    # ── Per-request metadata ──────────────────────────────────────────────
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    target_q_start = tl.load(target_query_start_loc_ptr + req_idx)
    target_q_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    target_q_len = target_q_end - target_q_start
    target_seq_len = tl.load(target_seq_lens_ptr + req_idx)

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    adjusted_q_len = target_q_len - num_rejected

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # ── PEagle extended buffer layout per request ─────────────────────────
    # [valid(adjusted_q_len) | pard(N-1) | stale(num_rejected)]
    #
    # Stale tokens come LAST (highest q_idx). Valid tokens start at q_idx 0,
    # which is correct under FA causal mask when seq_lens = target_seq_len + N - 1:
    #   valid token at q_offset i can attend to [0, target_seq_len - target_q_len + i]
    #   = its actual sequence position. Stale tokens over-attend into rejected KV
    #   positions but their output is always discarded.
    out_start = target_q_start + req_idx * (N - 1)
    last_valid_idx = out_start + adjusted_q_len - 1
    total_tokens = target_q_len + N - 1  # = adjusted_q_len + (N-1) + num_rejected

    # last_pos is the sequence position of the last valid eagle token;
    # pard tokens continue from last_pos+1.
    last_pos = tl.load(target_positions_ptr + target_q_start + adjusted_q_len - 1)

    # ── Single pass over all token slots in the extended buffer ───────────
    # Region classification determines behavior at each slot j:
    #   [0, adjusted_q_len)                  → valid
    #   [adjusted_q_len, adjusted_q_len+N-1) → pard
    #   [adjusted_q_len+N-1, total_tokens)   → stale
    for tok in range(0, total_tokens, BLOCK_SIZE_TOKENS):
        j = tok + tl.arange(0, BLOCK_SIZE_TOKENS)
        in_range = j < total_tokens

        is_valid = j < adjusted_q_len
        is_stale = j >= adjusted_q_len + N - 1
        is_pard = ~is_valid & ~is_stale
        is_last_valid = j == adjusted_q_len - 1

        valid_j = j  # offset within valid region (== j)
        pard_j = j - adjusted_q_len  # offset within pard region

        # input_ids: valid → shifted target ids (next_token at last slot)
        #            pard  → pard_token_id; stale → don't-care
        ids_shifted = tl.load(
            target_input_ids_ptr + target_q_start + 1 + valid_j,
            mask=in_range & is_valid & ~is_last_valid,
            other=0,
        )
        ids = tl.where(
            is_last_valid,
            next_token,
            tl.where(is_valid, ids_shifted, tl.where(is_pard, pard_token_id, 0)),
        )
        tl.store(out_input_ids_ptr + out_start + j, ids, mask=in_range)

        # positions: valid → direct copy; pard → last_pos+1+pard_j; stale → don't-care
        pos_from_target = tl.load(
            target_positions_ptr + target_q_start + valid_j,
            mask=in_range & is_valid,
            other=0,
        )
        pos = tl.where(
            is_valid, pos_from_target, tl.where(is_pard, last_pos + 1 + pard_j, 0)
        )
        tl.store(out_positions_ptr + out_start + j, pos, mask=in_range)

        # eagle_idx: valid → target buffer coordinate; pard/stale → 0 (don't-care)
        eagle_idx = tl.where(is_valid, (target_q_start + valid_j).to(tl.int32), 0)
        tl.store(out_eagle_idx_ptr + out_start + j, eagle_idx, mask=in_range)

        tl.store(out_is_hidden_masked_ptr + out_start + j, ~is_valid, mask=in_range)
        tl.store(out_is_stale_mask_ptr + out_start + j, is_stale, mask=in_range)

    # ── Per-request scalar outputs ────────────────────────────────────────
    tl.store(last_token_indices_ptr + req_idx, last_valid_idx)
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    tl.store(out_seq_lens_ptr + req_idx, target_seq_len + N - 1)

    # ── Last-request housekeeping ─────────────────────────────────────────
    if req_idx == num_reqs - 1:
        out_end = out_start + total_tokens
        tl.store(out_query_start_loc_ptr + num_reqs, out_end)
        tl.store(draft_current_step_ptr, 0)


def prepare_peagle_inputs(
    # Outputs: PEagle extended buffer
    out_input_ids: torch.Tensor,  # [max_extended]
    out_positions: torch.Tensor,  # [max_extended]
    out_query_start_loc: torch.Tensor,  # [max_num_reqs + 1]
    out_seq_lens: torch.Tensor,  # [max_num_reqs]
    out_is_hidden_masked: torch.Tensor,  # [max_extended] is_pard | is_stale
    out_is_stale_mask: torch.Tensor,  # [max_extended] stale only
    out_eagle_idx: torch.Tensor,  # [max_extended]
    last_token_indices: torch.Tensor,  # [max_num_reqs]
    current_draft_step: torch.Tensor,  # scalar
    # Inputs: directly from target model / input_batch
    input_batch: "InputBatch",
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    pard_token_id: int,
    num_speculative_steps: int,
) -> None:
    num_reqs = input_batch.num_reqs
    _prepare_peagle_inputs_kernel[(num_reqs,)](
        out_input_ids,
        out_positions,
        out_query_start_loc,
        out_seq_lens,
        out_is_hidden_masked,
        out_is_stale_mask,
        out_eagle_idx,
        last_token_indices,
        current_draft_step,
        input_batch.input_ids,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        pard_token_id,
        num_speculative_steps,
        BLOCK_SIZE_TOKENS=32,
    )
