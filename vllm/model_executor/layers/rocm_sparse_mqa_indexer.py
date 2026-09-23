# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm counterpart of `SparseMQAIndexer`, on aiter's paged MXFP4 kernel.

The candidate consumers of DeepSeek V4.1 score only the candidate pool the
source layer published, read straight from the paged MXFP4 cache. Requires
the `DeepseekV41RocmMxfp4IndexerBackend` metadata
(``indexer_kv_dtype="mxfp4"`` with ``indexer_sparse_logits``, gfx950).
"""

import torch
from torch import nn

from vllm.v1.attention.ops.rocm_mxfp4_indexer import (
    rocm_mxfp4_indexer_unsupported_reason,
    rocm_mxfp4_sparse_mqa_indexer,
)


class RocmSparseMQAIndexer(nn.Module):
    """Candidate-consuming indexer on aiter's paged MXFP4 MQA-logits kernel.

    ``forward`` takes the same arguments as `SparseAttnIndexer`, so the
    attention layer can call either.
    """

    weights_dtype = torch.float32
    """The scorer's weights and logits stay fp32: gfx950 has no packed bf16
    arithmetic, so its head reduce costs the same in either dtype."""

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        topk_indices_buffer: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
    ):
        super().__init__()
        if (reason := rocm_mxfp4_indexer_unsupported_reason()) is not None:
            raise ValueError(f"RocmSparseMQAIndexer: {reason}.")
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.num_candidate_cols = candidate_blocks.shape[1] * candidate_block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert k is None, "the model writes the indexer K cache"
        q_values, q_scale = q_quant
        return rocm_mxfp4_sparse_mqa_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.topk_indices_buffer,
            self.k_cache.compress_ratio,
            self.candidate_blocks,
            self.candidate_block_size,
            self.num_candidate_cols,
        )
