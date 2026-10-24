# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped self-drafting with shared weights, KV, and draft activations."""

from functools import partial
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces import SupportsLoopedSelfDraft
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.looped_self_draft.activation_cache import (
    LoopedSelfDraftActivationCache,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class LoopedSelfDraftSpeculator(TargetDependentARSpeculator):
    requires_hidden_states = False

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Keep proposal logits in FP32, including models with scaled output heads.
        return torch.float32, 0.0

    def compute_draft_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int
    ) -> torch.Tensor:
        return super().compute_draft_logits(hidden_states, spec_step_idx).float()

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        if not isinstance(target_model, SupportsLoopedSelfDraft):
            raise ValueError(
                "looped_self_draft requires a model implementing "
                "SupportsLoopedSelfDraft"
            )
        self.target_model = target_model
        return target_model.get_looped_self_draft_model()

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self.draft_attn_layer_names = set(target_model.get_draft_attn_layer_names())
        if self.speculative_config.reuse_draft_activations:
            target_model.draft_activation_cache = LoopedSelfDraftActivationCache(self)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        mode = (
            CUDAGraphMode.FULL_DECODE_ONLY
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            else CUDAGraphMode.NONE
        )
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, mode, decode_query_len=1
        )

    def capture(self) -> None:
        assert self.decode_cudagraph_manager is not None
        self.current_draft_step.zero_()
        self.idx_mapping.fill_(-1)
        self.decode_cudagraph_manager.capture(
            partial(self._generate_fused_drafts, start_step=0)
            if self.supports_decode_metadata_update
            else self._generate_draft,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing looped self-draft CUDA graphs",
        )
        if (cache := self.target_model.draft_activation_cache) is not None:
            cache.invalidate()

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )
        self._copy_request_inputs(
            num_reqs, input_batch.idx_mapping, temperature, seeds, dummy_run
        )
        cache = self.target_model.draft_activation_cache
        if cache is not None:
            cache.begin_proposal(input_batch, dummy_run)
        idx_mapping = self.idx_mapping[:num_reqs]

        _prepare_draft_inputs_kernel[(num_reqs + 1,)](
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.sample_src_positions,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            idx_mapping,
            input_batch.seq_lens,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_model_len,
            self.max_num_reqs,
            BLOCK_SIZE=1024,
        )

        draft_dp_sync_state, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync_state, num_reqs)
            if dp_sync_state is not None
            else (None, num_reqs)
        )
        batch_desc, draft_dp_sync_state = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync_state=draft_dp_sync_state,
        )
        num_tokens_across_dp = (
            draft_dp_sync_state.num_tokens_across_dp
            if draft_dp_sync_state is not None
            else None
        )

        generate = (
            self._fused_multi_step_decode
            if self.supports_decode_metadata_update
            else self._multi_step_decode
        )
        generate(
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            batch_desc,
            num_tokens_across_dp,
            input_batch.seq_lens_cpu_upper_bound + 1,
            self.num_speculative_steps,
            start_step=0,
        )
        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_draft_inputs_kernel(
    input_ids_ptr,
    positions_ptr,
    sample_src_positions_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    target_seq_lens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    max_model_len,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_idx == num_reqs:
        for i in range(0, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                query_start_loc_ptr + block,
                tl.minimum(block, num_reqs),
                mask=block < max_num_reqs + 1,
            )
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            tl.store(seq_lens_ptr + block, 0, mask=block < max_num_reqs)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        tl.store(input_ids_ptr + req_idx, 0)
        tl.store(positions_ptr + req_idx, 0)
        tl.store(sample_src_positions_ptr + req_idx, 0)
        tl.store(seq_lens_ptr + req_idx, 0)
        return
    if tl.load(num_sampled_ptr + req_idx) > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
    seq_len = tl.load(target_seq_lens_ptr + req_idx) - tl.load(
        num_rejected_ptr + req_idx
    )
    tl.store(input_ids_ptr + req_idx, next_token)
    tl.store(
        sample_src_positions_ptr + req_idx,
        tl.minimum(seq_len, max_model_len - 1).to(tl.int64),
    )
    tl.store(
        positions_ptr + req_idx, tl.minimum(seq_len, max_model_len - 1).to(tl.int64)
    )
    tl.store(seq_lens_ptr + req_idx, tl.minimum(seq_len + 1, max_model_len))
