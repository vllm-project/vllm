# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.qwen3_dspine import resolve_dspine_head_topk
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    CandidateSampler,
    DFlash2Speculator,
)


class DSpineSpeculator(DFlash2Speculator):
    _speculator_name = "DSpine"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        DFlashSpeculator.__init__(self, vllm_config, device)
        hf_config = self.draft_model_config.hf_config
        self.top_k = resolve_dspine_head_topk(
            hf_config, getattr(hf_config, "num_hidden_layers", None)
        )
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        lm_head = getattr(model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if (
            weight is None
            or weight.dtype not in (torch.bfloat16, torch.float32)
            or weight.dim() != 2
            or weight.stride(1) != 1
        ):
            raise ValueError(
                "DSpine requires an unquantized row-contiguous BF16/FP32 target lm_head"
            )
        shard = getattr(lm_head, "shard_indices", None)
        if shard is not None and (
            int(shard.num_added_elements) or int(shard.org_vocab_start_index)
        ):
            raise ValueError(
                "DSpine does not support added or offset target vocabulary"
            )
        model.materialize_inference_buffers(weight)
        return model

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states, last_raw, last_rms, last_query = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_idx = self.sample_indices[:num_sample]
        hidden_states = last_hidden_states[sample_idx].view(
            num_reqs, self.num_speculative_steps, -1
        )
        raw = last_raw[sample_idx].view(num_reqs, self.num_speculative_steps, -1)
        rms = last_rms[sample_idx].view(num_reqs, self.num_speculative_steps)
        query = last_query[sample_idx].view(num_reqs, self.num_speculative_steps, -1)
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = self._score_candidates(candidate_ids, unary_logits, raw, rms, query)
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        raw: torch.Tensor,
        rms: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        del unary_logits
        num_reqs = candidate_ids.shape[0]
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        return self.model.score_transitions(
            candidate_ids,
            raw,
            rms,
            query,
            anchor_token_ids,
            normalize=self.draft_logits is not None,
        )
