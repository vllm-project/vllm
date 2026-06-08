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
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


class PEagleSpeculator(EagleSpeculator):
    """PEagle (Parallel Eagle) speculator.

    Extends EagleSpeculator with single-pass parallel drafting: all N draft
    tokens are produced in one forward pass by prepending N-1 pard tokens to
    each request's Eagle input, instead of running N sequential decode steps.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
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
        self.peagle_is_pard_mask = torch.zeros(
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

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._init_parallel_drafting_hidden_state()

    def _init_parallel_drafting_hidden_state(self) -> None:
        # Extract the mask hidden state from the model for use at pard positions.
        # For Eagle3 with aux hidden states, project through combine_hidden_states.
        # If the model has no mask_hidden (e.g., plain Eagle), fall back to zeros.
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

        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
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

    def _prepare_extended_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
        num_rejected: torch.Tensor,
    ) -> int:
        """Populate peagle_* input buffers and hidden states. Returns num_extended."""
        N = self.num_speculative_steps
        prepare_peagle_prefill_inputs(
            last_token_indices=self.last_token_indices,
            peagle_input_ids=self.peagle_input_ids,
            peagle_positions=self.peagle_positions,
            peagle_query_start_loc=self.peagle_query_start_loc,
            peagle_seq_lens=self.peagle_seq_lens,
            peagle_is_pard_mask=self.peagle_is_pard_mask,
            peagle_is_stale_mask=self.peagle_is_stale_mask,
            peagle_eagle_idx=self.peagle_eagle_idx,
            eagle_input_ids=self.input_buffers.input_ids,
            eagle_positions=self.input_buffers.positions,
            eagle_query_start_loc=self.input_buffers.query_start_loc,
            eagle_seq_lens=self.input_buffers.seq_lens,
            num_rejected=num_rejected,
            pard_token_id=self.parallel_drafting_token_id,
            num_speculative_steps=N,
            max_num_reqs=self.max_num_reqs,
            num_reqs=num_reqs,
        )
        # Each request gains N-1 pard slots on top of its Eagle query length.
        num_extended = num_tokens + num_reqs * (N - 1)

        # Eagle positions get target hidden states; pard and stale get mask_hidden.
        assert self.parallel_drafting_hidden_state_tensor is not None
        mask_h = self.parallel_drafting_hidden_state_tensor
        eagle_idx = self.peagle_eagle_idx[:num_extended].long()
        is_pard = self.peagle_is_pard_mask[:num_extended]
        is_stale = self.peagle_is_stale_mask[:num_extended]
        self.peagle_hidden_states[:num_extended] = self.hidden_states[eagle_idx]
        torch.where(
            (is_pard | is_stale).unsqueeze(1),
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

        steps = torch.arange(N, device=self.device)
        all_sample_indices = (
            last_token_indices.unsqueeze(1) + steps.unsqueeze(0)
        ).reshape(-1)  # [B*N]
        all_logits = self.model.compute_logits(
            last_hidden_states[all_sample_indices]
        )  # [B*N, vocab]

        if self.draft_logits is None:
            self.draft_tokens[:num_reqs] = all_logits.argmax(dim=-1).view(num_reqs, N)
        else:
            all_logits_view = all_logits.view(num_reqs, N, -1)
            for step in range(N):
                self.current_draft_step.fill_(step)
                self.draft_tokens[:num_reqs, step] = gumbel_sample(
                    all_logits_view[:, step, :].contiguous(),
                    idx_mapping,
                    self.temperature,
                    self.seeds,
                    first_sample_positions + step + 1,
                    apply_temperature=True,
                    output_processed_logits=self.draft_logits,
                    output_processed_logits_col=self.current_draft_step,
                    use_fp64=self.use_fp64_gumbel,
                )

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


# ---------------------------------------------------------------------------
# PEagle (Parallel Eagle) helpers
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_peagle_prefill_inputs_kernel(
    # Outputs: extended PEagle buffer
    out_input_ids_ptr,  # [max_extended]
    out_positions_ptr,  # [max_extended]
    out_query_start_loc_ptr,  # [max_num_reqs + 1]
    out_seq_lens_ptr,  # [max_num_reqs]
    out_is_pard_mask_ptr,  # [max_extended] bool
    out_is_stale_mask_ptr,  # [max_extended] bool
    out_eagle_idx_ptr,  # [max_extended] int32: extended pos → eagle pos
    last_token_indices_ptr,  # [max_num_reqs] (mutated: set to last valid eagle pos)
    # Inputs: from prepare_prefill_inputs
    eagle_input_ids_ptr,  # [max_num_tokens]
    eagle_positions_ptr,  # [max_num_tokens]
    eagle_query_start_loc_ptr,  # [max_num_reqs + 1]
    eagle_seq_lens_ptr,  # [max_num_reqs]
    num_rejected_ptr,  # [max_num_reqs]
    pard_token_id,
    num_speculative_steps,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    N = num_speculative_steps

    eagle_q_start = tl.load(eagle_query_start_loc_ptr + req_idx)
    eagle_q_end = tl.load(eagle_query_start_loc_ptr + req_idx + 1)
    eagle_q_len = eagle_q_end - eagle_q_start

    # Use adjusted query length to exclude stale (rejected) tokens from the
    # previous speculative round.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    adjusted_q_len = eagle_q_len - num_rejected

    # Extended buffer layout per request:
    #   [stale(num_rejected) | eagle(adjusted_q_len) | pard(N-1)]
    #
    # Placing stale tokens FIRST ensures valid eagle tokens have higher
    # q_idx values under FlashAttention's relative causal mask, so they
    # can attend to all required KV positions.
    out_start = eagle_q_start + req_idx * (N - 1)
    valid_eagle_start = out_start + num_rejected

    # Copy valid eagle input IDs.
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        ids = tl.load(eagle_input_ids_ptr + eagle_q_start + block, mask=mask, other=0)
        tl.store(out_input_ids_ptr + valid_eagle_start + block, ids, mask=mask)

    # Fill pard token IDs after valid eagle tokens.
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_input_ids_ptr + valid_eagle_start + adjusted_q_len + block,
            pard_token_id,
            mask=mask,
        )

    # Copy valid eagle positions.
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        pos = tl.load(eagle_positions_ptr + eagle_q_start + block, mask=mask, other=0)
        tl.store(out_positions_ptr + valid_eagle_start + block, pos, mask=mask)

    # Fill pard positions: last valid eagle pos + 1, +2, ...
    last_pos = tl.load(eagle_positions_ptr + eagle_q_start + adjusted_q_len - 1)
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_positions_ptr + valid_eagle_start + adjusted_q_len + block,
            last_pos + 1 + block,
            mask=mask,
        )

    # is_stale_mask: True for stale slots [out_start, valid_eagle_start).
    # is_pard_mask:  False for eagle tokens, True for pard tokens.
    for i in range(0, num_rejected, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < num_rejected
        tl.store(out_is_stale_mask_ptr + out_start + block, 1, mask=mask)
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        tl.store(out_is_pard_mask_ptr + valid_eagle_start + block, 0, mask=mask)
        tl.store(out_is_stale_mask_ptr + valid_eagle_start + block, 0, mask=mask)
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_is_pard_mask_ptr + valid_eagle_start + adjusted_q_len + block,
            1,
            mask=mask,
        )
        tl.store(
            out_is_stale_mask_ptr + valid_eagle_start + adjusted_q_len + block,
            0,
            mask=mask,
        )

    # Eagle index mapping: valid eagle positions → original eagle buffer index.
    # Pard and stale positions map to 0 (placeholder; hidden state overridden).
    for i in range(0, adjusted_q_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < adjusted_q_len
        tl.store(
            out_eagle_idx_ptr + valid_eagle_start + block,
            eagle_q_start + block,
            mask=mask,
        )
    for i in range(0, N - 1, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < (N - 1)
        tl.store(
            out_eagle_idx_ptr + valid_eagle_start + adjusted_q_len + block, 0, mask=mask
        )

    # Update last_token_indices to the last valid eagle token (sampling start).
    tl.store(last_token_indices_ptr + req_idx, valid_eagle_start + adjusted_q_len - 1)

    # Extended query metadata.
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    eagle_seq_len = tl.load(eagle_seq_lens_ptr + req_idx)
    tl.store(out_seq_lens_ptr + req_idx, eagle_seq_len - num_rejected + N - 1)

    if req_idx == num_reqs - 1:
        out_end = out_start + eagle_q_len + N - 1
        tl.store(out_query_start_loc_ptr + num_reqs, out_end)
        # Pad remaining entries for CUDA graphs.
        for i in range(num_reqs + 1, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(out_query_start_loc_ptr + block, out_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)


def prepare_peagle_prefill_inputs(
    last_token_indices: torch.Tensor,
    peagle_input_ids: torch.Tensor,
    peagle_positions: torch.Tensor,
    peagle_query_start_loc: torch.Tensor,
    peagle_seq_lens: torch.Tensor,
    peagle_is_pard_mask: torch.Tensor,
    peagle_is_stale_mask: torch.Tensor,
    peagle_eagle_idx: torch.Tensor,
    eagle_input_ids: torch.Tensor,
    eagle_positions: torch.Tensor,
    eagle_query_start_loc: torch.Tensor,
    eagle_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    pard_token_id: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    num_reqs: int,
) -> None:
    _prepare_peagle_prefill_inputs_kernel[(num_reqs,)](
        peagle_input_ids,
        peagle_positions,
        peagle_query_start_loc,
        peagle_seq_lens,
        peagle_is_pard_mask,
        peagle_is_stale_mask,
        peagle_eagle_idx,
        last_token_indices,
        eagle_input_ids,
        eagle_positions,
        eagle_query_start_loc,
        eagle_seq_lens,
        num_rejected,
        pard_token_id,
        num_speculative_steps,
        max_num_reqs,
        BLOCK_SIZE=32,
    )
