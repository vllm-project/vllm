# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DNeo speculator: DFlash's single-pass block drafting with prefix-only attention
and DSpark's left-to-right sampling through DNeo's causal head."""

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class DNeoSpeculator(DSparkSpeculator):
    _speculator_name = "DNeo"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        dneo_config = getattr(self.draft_model_config.hf_config, "dneo_config", {})
        if dneo_config.get("prefix_only_attention", True):
            # A draft query sees the context and its block's anchor only: with
            # non-causal attention, the first query key and no other.
            self.num_query_kv_per_req = 1
            # The rejected suffix's features, attended as K/V only.
            self.rejection_memory = bool(dneo_config.get("rejection_memory", False))
        elif dneo_config.get("rejection_memory", False):
            raise ValueError("DNeo's rejection memory needs prefix-only attention")
        # Query offset of draft step 0, the block slot whose head scales it uses.
        self._first_slot = 0 if self.sample_from_anchor else 1

    def _sample_markov_chain(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        prev: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        out_tokens: torch.Tensor,
        *,
        prev_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Sample the drafts left to right, each conditioned on the previous one
        through the causal head. DNeo has no confidence head."""
        if prev_indices is not None:
            prev = prev[prev_indices]
        num_rows, num_steps = out_tokens.shape
        projected = self.model.project_head_hidden(hidden).view(num_rows, num_steps, -1)
        for i in range(num_steps):
            logits_i = self.model.head_step_logits(
                base_logits[:, i], projected[:, i], prev, self._first_slot + i
            )
            draft_i = self._sample_logits(logits_i, idx_map[:, i], sample_pos[:, i], i)
            out_tokens[:, i] = draft_i
            prev = draft_i
        return None
