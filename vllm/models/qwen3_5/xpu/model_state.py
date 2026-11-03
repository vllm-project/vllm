# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5's model runner state on XPU with the mono decode (``--monokernel``):
the hybrid (GatedDeltaNet + attention) state, plus the lifecycle of the
executor (``specialized_decode.runtime.decode``) and its candidate sampling
(``specialized_decode.runtime.sampling``)."""

from typing import Any

import torch

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from .specialized_decode.runtime.sampling import (
    CandidateRejectionSampler,
    Candidates,
    CandidateSampler,
)


class Qwen3_5XpuModelState(MambaHybridModelState):
    """Builds the mono decode once the weights are loaded (with the MTP
    drafter, loaded first), binds it to every KV cache vLLM allocates (the
    memory-profiling cache, then the real one), samples small steps on the
    vocabulary-shard candidates, and runs the executor's per-step check with
    the logits."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        self.decode = model.build_specialized_decode()
        self.candidates = Candidates(
            vllm_config, self.decode.region, self.decode.rank, self.max_num_reqs
        )
        lm_head = getattr(model, "lm_head", None)
        if getattr(lm_head, "num_embeddings_padded", None) != getattr(
            lm_head, "org_vocab_size", None
        ):
            # the shards of a padded vocabulary hold padding rows
            self.candidates.enabled = False

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        self.decode.bind_caches()

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        self.candidates.batch = input_batch
        return super().prepare_inputs(input_batch, req_states)

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        self.candidates.batch = None
        return super().prepare_dummy_inputs(num_reqs, num_tokens)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        if not self.candidates.enabled:
            return None
        sampler = CandidateSampler(sampler, self.candidates)
        spec = self.vllm_config.speculative_config
        rejection = (
            None
            if spec is None
            else CandidateRejectionSampler(sampler, spec, self.device)
        )
        return sampler, rejection

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch = self.candidates.batch
        k = self.candidates.plan(hidden_states.shape[0])
        if k is None or batch is None:
            logits = super().compute_sample_logits(hidden_states)
        else:
            shard = self.language_model.compute_logits_local(hidden_states)
            logits = self.candidates.logits(shard, k, batch.num_draft_tokens > 0)
        self.decode.after_step()
        return logits
