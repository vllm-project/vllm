# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse Attention Indexer that scores only the candidate blocks.

DeepSeek V4.1 two-level selection: the candidate-source indexer publishes
the top candidate blocks and later indexers pick their top-k inside them.
`SparseAttnIndexer` does that by computing dense logits over the whole
context and masking; this layer instead calls DeepGEMM's sparse MQA-logits
kernels on the candidate blocks only, so the work is O(candidate blocks)
instead of O(context). It requires the `DeepseekV41SparseIndexerBackend`
metadata (see ``AttentionConfig.indexer_sparse_logits``).
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    has_deep_select,
    sparse_mqa_logits_paged_decode,
    sparse_mqa_logits_prefill_chunk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _prefill_k_workspaces(
    total_seq_lens: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed MXFP4 K-gather workspace ``(values, scales)``, shared with
    the dense indexer layers of the same model."""
    values_spec, scales_spec = _gather_workspace_shapes(
        total_seq_lens, head_dim, current_platform.fp8_dtype(), use_fp4_cache=True
    )
    k_quant, k_scale = current_workspace_manager().get_simultaneous(
        values_spec, scales_spec
    )
    return k_quant, k_scale


def _gather_prefill_chunk_k(
    kv_cache: torch.Tensor,
    k_quant_full: torch.Tensor,
    k_scale_full: torch.Tensor,
    chunk: DeepseekV32IndexerPrefillChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one prefill chunk's paged K into the packed workspace."""
    assert chunk.local_cu_seq_lens is not None
    k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
    k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
    if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_quant,
            k_scale,
            chunk.block_table,
            chunk.local_cu_seq_lens,
        )
    return k_quant, k_scale


class SparseMQAIndexer(nn.Module):
    """Candidate-consuming indexer on DeepGEMM's sparse MQA-logits kernels.

    Only valid for indexer layers that read candidate blocks with the MXFP4
    indexer cache on SM100. The K cache is written by the model before this
    runs; ``forward`` takes the same arguments as `SparseAttnIndexer` so the
    attention layer can call either.
    """

    weights_dtype = torch.bfloat16
    """Per-head weights dtype the sparse kernels take. The fused Q RoPE-quant
    kernel writes it directly so no cast runs per step."""

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
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_deep_gemm_sparse_mqa()
        ):
            raise ValueError(
                "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8."
            )
        if not has_deep_select():
            raise ValueError(
                "SparseMQAIndexer requires the DeepSelect top-k extension "
                "(vllm._deepselect_C)."
            )
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size

    def _reserve_workspaces(self, device: torch.device) -> None:
        """Profiling run: claim the K-gather workspace and the peak sparse
        logits allocation so the memory estimate covers them."""
        _prefill_k_workspaces(self.max_total_seq_len, self.head_dim)
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        torch.empty(max_logits_bytes, dtype=torch.uint8, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            self._reserve_workspaces(hidden_states.device)
            return self.topk_indices_buffer

        metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(metadata, DeepseekV41SparseIndexerMetadata), (
            "SparseMQAIndexer needs DeepseekV41SparseIndexerBackend metadata"
        )
        assert k is None, "the model writes the indexer K cache"
        q_values, q_scale = q_quant
        kv_cache = self.k_cache.kv_cache
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer
        topk_indices_buffer[: hidden_states.shape[0]] = -1
        sparse_block_kv = metadata.sparse_block_kv

        if metadata.num_prefills > 0:
            prefill = metadata.prefill
            sparse_chunks = metadata.sparse_prefill
            assert prefill is not None and sparse_chunks is not None
            k_quant_full, k_scale_full = _prefill_k_workspaces(
                self.max_total_seq_len, self.head_dim
            )
            for chunk, rows in zip(prefill.chunks, sparse_chunks):
                k_quant, k_scale = _gather_prefill_chunk_k(
                    kv_cache, k_quant_full, k_scale_full, chunk
                )
                if chunk.local_total_seq_lens == 0:
                    continue  # buffer already holds -1
                start, end = chunk.token_start, chunk.token_end
                rows.kernel_metadata = sparse_mqa_logits_prefill_chunk(
                    q_values[start:end].view(torch.int8),
                    q_scale[start:end],
                    k_quant.view(torch.int8),
                    k_scale.view(torch.int32).squeeze(-1),
                    weights[start:end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    self.candidate_blocks[start:end],
                    self.candidate_block_size,
                    sparse_block_kv,
                    topk_tokens,
                    topk_indices_buffer[start:end, :topk_tokens],
                    sparse_indices=rows.sparse_indices,
                    end=rows.end,
                    col_indices=rows.col_indices,
                    kernel_metadata=rows.kernel_metadata,
                )

        if metadata.num_decodes > 0:
            decode_rows = metadata.sparse_decode
            assert decode_rows is not None and decode_rows.block_table is not None
            assert decode_rows.row_indices is not None
            num_rows = decode_rows.row_ke.shape[0]
            if num_rows > 0:
                decode_rows.kernel_metadata = sparse_mqa_logits_paged_decode(
                    q_values[:num_rows].view(torch.int8).unsqueeze(1),
                    q_scale[:num_rows].unsqueeze(1),
                    kv_cache_as_quant_view(kv_cache, self.head_dim, use_fp4_cache=True),
                    weights[:num_rows],
                    decode_rows.row_ke,
                    decode_rows.block_table,
                    decode_rows.row_indices,
                    self.candidate_blocks[:num_rows],
                    self.candidate_block_size,
                    sparse_block_kv,
                    topk_tokens,
                    topk_indices_buffer[:num_rows, :topk_tokens],
                    row_ks=decode_rows.row_ks,
                    sparse_indices=decode_rows.sparse_indices,
                    end=decode_rows.end,
                    col_indices=decode_rows.col_indices,
                    kernel_metadata=decode_rows.kernel_metadata,
                )
        return topk_indices_buffer
