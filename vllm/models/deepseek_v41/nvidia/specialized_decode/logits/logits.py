# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits of the specialized decode: the low-batch LM head GEMM (``lm_head``)
and vocab-parallel logits gathered with the TP group's custom all-gather.

For decode batches the one-shot custom all-gather is several times faster than
the NCCL ring. The kernel falls back to NCCL per call by dtype and size alone,
so every TP rank takes the same path. It shares its staging buffers and
signals with the group's other custom collectives, so the logits must be
gathered on the forward stream after any side stream has rejoined it.
"""

import torch

from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

from .lm_head import low_batch_lm_head, low_batch_lm_head_applies


class SpecializedDecodeLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.pdl = current_platform.is_arch_support_pdl()

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
        communicator = get_tp_group().device_communicator
        if self.use_all_gather and isinstance(communicator, CudaCommunicator):
            gathered = communicator.custom_all_gather(logits)
            if gathered is not None:
                # Rank-major rows to rank-major vocab shards, as NCCL lays out.
                return gathered.view(-1, *logits.shape).movedim(0, -2).flatten(-2)
        return super()._gather_logits(logits)
