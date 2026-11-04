# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 DSpark draft model on ROCm.

Reuses the shared draft model (including its MLA layers) from
`nvidia/dspark_mla.py` and adds only the vocabulary-sharded Markov sampler
(`ops/dspark_markov.py`)."""

import torch

from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm

from ..nvidia.dspark_mla import K3DSparkForCausalLM as _K3DSparkForCausalLM
from .ops.dspark_markov import ShardedMarkovSampler


class K3DSparkForCausalLM(_K3DSparkForCausalLM):
    _persistent_draft = None

    def setup_sample_sequential(self, speculator) -> None:
        """Called by DSparkSpeculator.load_model on every TP rank, before graph
        capture. Installs the vocabulary-sharded Markov sampler as
        `sample_sequential` when it applies."""
        sampler = ShardedMarkovSampler.create(speculator)
        if sampler is None:
            return
        self.sample_sequential = sampler
        from .ops.dspark_persistent.runtime import PersistentDraft

        self._persistent_draft = PersistentDraft.create(
            self.model, speculator, sampler.failstop.failed
        )
        if self._persistent_draft is not None:
            sampler.failstop.what = "Kimi-K3 DSpark sampling or draft decode"

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        if self._persistent_draft is not None:
            self._persistent_draft.refresh_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._persistent_draft is not None:
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_input_ids(input_ids)
            result = self._persistent_draft.forward(positions, inputs_embeds)
            if result is not None:
                hidden_states, residual = result
                hidden_states, _ = fused_allreduce_rms_norm(
                    hidden_states, residual, self.model.final_norm
                )
                return hidden_states
        return super().forward(input_ids, positions, inputs_embeds)
