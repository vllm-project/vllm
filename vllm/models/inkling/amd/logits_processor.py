# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling logits processor with muP output scaling."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class InklingLogitsProcessor(LogitsProcessor):
    """``LogitsProcessor`` that applies Inkling's muP logits width multiplier.

    Args:
        vocab_size: Padded vocabulary size.
        org_vocab_size: Unpadded vocabulary size (defaults to ``vocab_size``).
        scale: Base logits scale (kept ``1.0`` for the served checkpoint).
        logits_as_input: Whether the input is already logits.
        soft_cap: Optional logit soft cap (``None`` for the served checkpoint).
        logits_mup_width_multiplier: muP width divisor for the final logits;
            ``None`` or ``0`` disables it.
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
        logits_mup_width_multiplier: float | None = None,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            org_vocab_size=org_vocab_size,
            scale=scale,
            logits_as_input=logits_as_input,
            soft_cap=soft_cap,
        )
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self._logits_zero: torch.Tensor | None = None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        mup = self.logits_mup_width_multiplier
        if not mup:
            return super().forward(lm_head, hidden_states, embedding_bias)
        # Fold the muP width divisor into the lm_head GEMM alpha (fp32 epilogue):
        # no separate elementwise kernel, no bf16 rounding of scaled logits, and
        # no weight mutation. Overfit to the served checkpoint: bf16 lm_head, no
        # soft cap, unit logits scale.
        assert self.soft_cap is None
        assert self.scale == 1.0
        w = lm_head.weight
        if self._logits_zero is None:
            self._logits_zero = w.new_zeros(1)
        logits = torch.addmm(
            self._logits_zero,
            hidden_states,
            w.t(),
            beta=0.0,
            alpha=1.0 / mup,
        )
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits
