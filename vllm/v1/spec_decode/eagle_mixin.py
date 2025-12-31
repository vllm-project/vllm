# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch


class EagleMixin:
    compute_logits: Any

    def sample_chain(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.compute_logits(hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        return draft_token_ids


class Eagle3Mixin:
    lm_head: Any
    logits_processor: Any
    draft_id_to_target_id: Any | None
    config: Any

    def compute_draft_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Computes draft logits from hidden states.
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Maps draft logits to target logits.
        logits = self.compute_draft_logits(hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def sample_chain(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.compute_draft_logits(hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        
        if self.draft_id_to_target_id is None:
            return draft_token_ids
        
        offset = self.draft_id_to_target_id[draft_token_ids]
        draft_token_ids += offset
        return draft_token_ids
