# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramLookup
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)


class NgramMTPSpeculator(MTPSpeculator):
    """MTP drafting that copies from the context on an n-gram match.

    The MTP draft prefill always runs, so the draft KV cache stays complete.
    The MTP decode steps are skipped when every request matched. Deciding that
    waits for the lookup, overlapped with the draft prefill, and only while the
    previous round matched every request.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: "RequestState",
    ):
        super().__init__(vllm_config, device)
        spec = self.speculative_config
        assert spec.prompt_lookup_min is not None
        assert spec.prompt_lookup_max is not None
        self.req_states = req_states
        self.ngram = NgramLookup(
            spec.prompt_lookup_min,
            spec.prompt_lookup_max,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_model_len,
            device,
        )
        # Skipping draft forwards on one DP rank would desync the others.
        self.can_skip_decode = self.dp_size == 1
        # Double-buffered so the previous round's flag can be read while this
        # round's copy is in flight.
        self.all_matched_cpu = torch.zeros(2, dtype=torch.bool, pin_memory=True)
        self.all_matched_events = (torch.cuda.Event(), torch.cuda.Event())
        self.round = 0
        self.wait_for_lookup = False

    def num_draft_steps(self, num_speculative_tokens: int) -> int:
        if not self.wait_for_lookup:
            return num_speculative_tokens
        self.wait_for_lookup = False
        cur = self.round % 2
        with gpu_sync_allowed():
            self.all_matched_events[cur].synchronize()
        return 1 if self.all_matched_cpu[cur] else num_speculative_tokens

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
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if not dummy_run:
            ngram_drafts, has_match = self.ngram.lookup(
                self.req_states,
                input_batch.idx_mapping,
                num_sampled,
                last_sampled,
                num_reqs,
            )
            if self.can_skip_decode:
                # Wait for this round's lookup only while the previous round
                # matched every request, read without blocking: batches
                # without matches never wait.
                prev = self.round % 2
                self.wait_for_lookup = bool(
                    self.all_matched_events[prev].query() and self.all_matched_cpu[prev]
                )
                self.round += 1
                cur = self.round % 2
                # Requests that sampled nothing (chunked prefill) need no draft.
                all_matched = (has_match | (num_sampled == 0)).all()
                self.all_matched_cpu[cur].copy_(all_matched, non_blocking=True)
                self.all_matched_events[cur].record()

        draft_tokens = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync=dp_sync,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        self.wait_for_lookup = False
        if dummy_run:
            return draft_tokens
        torch.where(has_match[:, None], ngram_drafts, draft_tokens, out=draft_tokens)
        return draft_tokens
