# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head


class LiLiCorrSpeculator(DFlash2Speculator):
    _speculator_name = "LiLiCorr"
    _candidate_top_k_key = "lilicorr_candidate_topk"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        self.anchor_hidden = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self.anchor_valid = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

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
        lm_head = (
            model.lm_head
            if model.has_own_lm_head
            else get_target_lm_head(target_model, language_model)
        )
        if not isinstance(embedding, VocabParallelEmbedding) or not isinstance(
            lm_head, VocabParallelEmbedding
        ):
            raise ValueError(
                "LiLiCorr requires the target input embedding and its selected "
                "LM head on the draft rank."
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
        # Candidate embeddings always come from the target; the checkpoint may
        # supply its own candidate LM head.
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

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = candidate_ids.shape[0]
        return self.model.model.lilicorr(
            self.target_embeddings(candidate_ids),
            unary_logits,
            hidden_states,
            self.anchor_hidden[:num_reqs],
            self.anchor_valid[:num_reqs],
        )
