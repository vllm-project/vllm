# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm counterparts of the DeepSeek V4.1 indexer's ops and layers, on aiter's
paged MXFP4 kernels.

With ``indexer_kv_dtype="mxfp4"`` on gfx950, aiter writes the indexer K cache
in the order its MQA-logits kernel reads, quantizes the query next to it, and
the kernel scores the cache in place. Requires the
`DeepseekV41RocmMxfp4IndexerBackend` metadata.
"""

import functools
from collections.abc import Callable

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_mxfp4_indexer_k_store,
    rocm_mxfp4_indexer_q_quant,
    rocm_mxfp4_indexer_unsupported_reason,
    rocm_mxfp4_sparse_attn_indexer,
    rocm_mxfp4_sparse_mqa_indexer,
)


def rocm_mxfp4_indexer_ops(
    num_heads: int,
) -> tuple[
    Callable[..., None],
    Callable[..., tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
]:
    """The K store and Q quant of an indexer with ``num_heads`` query heads,
    called as `indexer_k_norm_rope_store` and `fused_indexer_q_rope_quant`."""
    return (
        functools.partial(rocm_mxfp4_indexer_k_store, num_heads=num_heads),
        rocm_mxfp4_indexer_q_quant,
    )


class RocmSparseAttnIndexer(SparseAttnIndexer):
    """`SparseAttnIndexer` whose HIP path walks the paged MXFP4 cache with
    aiter's kernel, and publishes the candidate pool from the source layer."""

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
    """Candidate-consuming indexer on aiter's paged MXFP4 MQA-logits kernel:
    the consumers score only the candidate pool the source layer published,
    read straight from the paged cache.

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
        if (reason := rocm_mxfp4_indexer_unsupported_reason()) is not None:
            raise ValueError(f"RocmSparseMQAIndexer: {reason}.")
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
