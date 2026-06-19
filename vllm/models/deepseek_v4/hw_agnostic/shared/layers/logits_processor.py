# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LogitsProcessor — hw-agnostic vendored copy.

Vendored from ``vllm/model_executor/layers/logits_processor.py`` with
its only sideways-import (``VocabParallelEmbedding``) rewired to the
local hw-agnostic copy. No vendor dispatch is involved here, so the
file is otherwise verbatim.
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.models.deepseek_v4.hw_agnostic.shared.custom_op import PluggableLayer
from vllm.platforms import current_platform

from .vocab_parallel_embedding import VocabParallelEmbedding


class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        self.logits_as_input = logits_as_input
        self.org_vocab_size = org_vocab_size or vocab_size
        self.soft_cap = soft_cap
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap
            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_all_gather:
            # Gather isn't supported for some devices (e.g. TPUs); use
            # all-gather to keep all ranks in lockstep.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits."""
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
