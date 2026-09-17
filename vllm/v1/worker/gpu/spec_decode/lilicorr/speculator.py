# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head


class LiLiCorrSpeculator(DFlashSpeculator):
    _speculator_name = "LiLiCorr"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        self.top_k = int(config.dflash_config["lilicorr_candidate_topk"])
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_speculative_steps, self.top_k, device
        )
        self.anchor_hidden = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self.anchor_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        return torch.float32, -float("inf")

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        inner = getattr(language_model, "model", language_model)
        embedding = getattr(inner, "embed_tokens", None) or getattr(
            inner, "embedding", None
        )
        lm_head = get_target_lm_head(target_model, language_model)
        if not isinstance(embedding, VocabParallelEmbedding) or not isinstance(
            lm_head, VocabParallelEmbedding
        ):
            raise ValueError(
                "LiLiCorr requires the target input embedding and LM head "
                "on the draft rank."
            )
        if embedding.embedding_dim != model.config.hidden_size:
            raise ValueError(
                "LiLiCorr target embedding width must match its trained "
                "token projection input."
            )
        if (
            embedding.shard_indices.num_added_elements
            or lm_head.shard_indices.num_added_elements
        ):
            raise ValueError("LiLiCorr does not support added vocabulary entries.")
        # These are the target tables the correlator was trained against, even
        # when the draft backbone owns different embeddings or an LM head.
        self.target_embeddings = embedding
        model.lm_head = lm_head
        return model

    def prepare_context_anchor(
        self, input_batch: InputBatch, num_rejected: torch.Tensor
    ) -> None:
        num_reqs = input_batch.num_reqs
        starts = input_batch.query_start_loc[:num_reqs]
        ends = input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        valid = ends > starts
        indices = (ends - 1).clamp_min(0).long()
        self.anchor_hidden.zero_()
        self.anchor_valid.zero_()
        self.anchor_hidden[:num_reqs].copy_(
            self.model.model.hidden_norm(self.hidden_states[indices]) * valid[:, None]
        )
        self.anchor_valid[:num_reqs].copy_(valid)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden = hidden[self.sample_indices[:num_sample]]
        candidates, log_probs = self.model.compute_candidates(hidden)
        candidates = candidates.view(num_reqs, self.num_speculative_steps, self.top_k)
        scores = self.model.model.lilicorr(
            self.target_embeddings(candidates),
            log_probs.view_as(candidates),
            hidden.view(num_reqs, self.num_speculative_steps, -1),
            self.anchor_hidden[:num_reqs],
            self.anchor_valid[:num_reqs],
        )
        self.candidate_sampler.sample(
            candidates,
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
