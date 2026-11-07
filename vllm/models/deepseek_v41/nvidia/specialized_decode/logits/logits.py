# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits of the specialized decode: the low-batch LM head GEMM (``lm_head``)
and vocab-parallel logits gathered with the executor's all-gather.

For decode batches a one-shot all-gather over NVLink multicast is several
times faster than the NCCL ring. The executor's ``LamportAllGather`` takes the
logits when they fit, by size alone, so every TP rank takes the same path;
larger ones go to NCCL. vLLM's custom all-gather is not used: its MNNVL
Lamport kernel never waits for a rank's own multicast copy (see
``ops.cute_dsl.all_gather``). The all-gather's mailbox also serves the
row-sharded indexers, so the logits must be gathered on the forward stream
after any side stream has rejoined it.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

from .lm_head import low_batch_lm_head, low_batch_lm_head_applies

if TYPE_CHECKING:
    from ...ops.cute_dsl import LamportAllGather


class SpecializedDecodeLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vocab_size: int, all_gather: "LamportAllGather | None" = None
    ) -> None:
        super().__init__(vocab_size)
        self.pdl = current_platform.is_arch_support_pdl()
        self.all_gather = all_gather

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            embedding_bias is None
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
            and low_batch_lm_head_applies(lm_head, hidden_states)
        ):
            return low_batch_lm_head(lm_head, hidden_states, self.pdl)
        return super()._apply_head(lm_head, hidden_states, embedding_bias)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if (
            self.use_all_gather
            and self.all_gather is not None
            and self.all_gather.accepts(logits)
        ):
            gathered = self.all_gather(logits)
            # Rank-major rows to rank-major vocab shards, as NCCL lays out.
            return gathered.view(-1, *logits.shape).movedim(0, -2).flatten(-2)
        return super()._gather_logits(logits)
