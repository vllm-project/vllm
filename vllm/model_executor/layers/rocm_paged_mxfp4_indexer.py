# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer layers on aiter's paged MXFP4 kernels, for ROCm.

With ``indexer_kv_dtype="mxfp4"`` on gfx950, the model writes the indexer K
cache in the order aiter's MQA-logits kernel reads (see
`rocm_paged_mxfp4_cache_layout`), and the kernel scores the cache in place.
Requires `RocmMxfp4IndexerMetadataBuilder` metadata.
"""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_mxfp4_sparse_attn_indexer,
    rocm_mxfp4_sparse_mqa_indexer,
)


class RocmSparseAttnIndexer(SparseAttnIndexer):
    """`SparseAttnIndexer` whose HIP path walks the paged MXFP4 cache with
    aiter's kernel. A two-level indexer's source layer also publishes its
    candidate pool here."""

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert isinstance(q_quant, tuple) and self.skip_k_cache_insert, (
            "the ROCm MXFP4 indexer takes (values, scales) Q and a K cache "
            "the model already wrote"
        )
        q_values, q_scale = q_quant
        return rocm_mxfp4_sparse_attn_indexer(
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
            compress_ratio=self.compress_ratio,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
        )


class RocmSparseMQAIndexer(nn.Module):
    """DeepSeek-V4.1's candidate-consuming indexer on aiter's paged MXFP4
    MQA-logits kernel: the consumers score only the candidate pool the source
    layer published, read straight from the paged cache.

    Built and called as `SparseMQAIndexer` is, so the model can take either.
    """

    weights_dtype = torch.float32
    """The scorer's weights and logits stay fp32: gfx950 has no packed bf16
    arithmetic, so its head reduce costs the same in either dtype."""

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        # aiter's kernel bounds a row by the model length in compressed
        # positions, not by the prefill buffer SparseMQAIndexer sizes.
        model_config = get_current_vllm_config().model_config
        self.max_model_len = model_config.max_model_len // k_cache.compress_ratio
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
