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
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (
    NgramLookup,
    write_one_hot_draft_logits,
)
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

    Only batches of one request use the lookup: the gain comes from skipping
    the MTP decode steps, which needs every request to match, and a copy that
    does not skip them only displaces the MTP draft. On a match the host waits
    for the lookup (overlapped with the draft prefill, which always runs so the
    draft KV cache stays complete) and skips the decode steps.
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
        self.matched_cpu = torch.zeros(1, dtype=torch.bool, pin_memory=True)
        self.matched_event = torch.cuda.Event()
        self.wait_for_lookup = False

    def num_draft_steps(self, num_speculative_tokens: int) -> int:
        if not self.wait_for_lookup:
            return num_speculative_tokens
        self.wait_for_lookup = False
        with gpu_sync_allowed():
            self.matched_event.synchronize()
        return 1 if self.matched_cpu.item() else num_speculative_tokens

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
        use_lookup = input_batch.num_reqs == 1 and not dummy_run
        if use_lookup:
            ngram_drafts, has_match = self.ngram.lookup(
                self.req_states,
                input_batch.idx_mapping,
                num_sampled,
                last_sampled,
                1,
            )
            self.matched_cpu.copy_(has_match, non_blocking=True)
            self.matched_event.record()
            self.wait_for_lookup = True

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
        if not use_lookup:
            return draft_tokens
        torch.where(has_match[:, None], ngram_drafts, draft_tokens, out=draft_tokens)
        if self.draft_logits is not None:
            write_one_hot_draft_logits(
                self.draft_logits, input_batch.idx_mapping, has_match, ngram_drafts
            )
        return draft_tokens
