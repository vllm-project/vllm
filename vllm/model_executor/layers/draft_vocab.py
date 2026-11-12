# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft lm_heads that cover a subset of the target vocabulary.

Checkpoints with a pruned draft lm_head (EAGLE3, DFlash, DSpark) ship
`draft_id_to_target_id` offsets. `DraftVocab.compute_logits` scatters the draft
logits into target-vocabulary logits that are `-inf` elsewhere, so speculators
need no special handling.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class DraftVocab:
    """Draft logits over a subset of the target vocabulary.

    Args:
        logits_processor: Logits processor of the draft lm_head.
        vocab_size: Target vocabulary size. A smaller draft vocabulary gets a
            `draft_id_to_target_id` parameter for the model to register and
            load.

    """

    def __init__(self, logits_processor: LogitsProcessor, vocab_size: int):
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size
        self.draft_id_to_target_id: nn.Parameter | None = None
        if logits_processor.vocab_size < vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(logits_processor.vocab_size, dtype=torch.long),
                requires_grad=False,
            )

    def compute_logits(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Target-vocabulary logits, `-inf` outside the draft vocabulary."""
        logits = self.logits_processor(lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(logits.shape[-1], device=logits.device)
        targets = base + self.draft_id_to_target_id
        out = logits.new_full((logits.shape[0], self.vocab_size), float("-inf"))
        out[:, targets] = logits
        return out

    def get_top_tokens(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Greedy target token ids without materializing full logits."""
        top = self.logits_processor.get_top_tokens(lm_head, hidden_states)
        return self.map_draft_to_target(top)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Target ids of draft-vocabulary ids from the pruned lm_head."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]
