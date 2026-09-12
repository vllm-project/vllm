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

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    SPARSE_TOPK_KERNEL_SUPPORTED,
    sparse_mqa_logits_paged_decode,
    sparse_mqa_logits_prefill_chunk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    SparseAttnIndexer,
    dense_mha_skips_topk,
    gather_prefill_chunk_k,
    get_prefill_k_workspaces,
    kv_cache_as_quant_view,
    reserve_indexer_workspaces,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager


@eager_break_during_capture
def sparse_mqa_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    dense_mha_metadata_layer_name: LayerNameType,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> torch.Tensor:
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    if not isinstance(attn_metadata, dict):
        # Profiling run: reserve the same workspaces as the dense indexer.
        reserve_indexer_workspaces(
            total_seq_lens, head_dim, use_fp4_cache=True, device=hidden_states.device
        )
        return topk_indices_buffer

    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, DeepseekV41SparseIndexerMetadata), (
        "SparseMQAIndexer needs DeepseekV41SparseIndexerBackend metadata"
    )
    if dense_mha_skips_topk(
        forward_context, attn_metadata, dense_mha_metadata_layer_name
    ):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    sparse_block_kv = metadata.sparse_block_kv
    zero_starts = metadata.zero_starts
    assert zero_starts is not None

    if metadata.num_prefills > 0:
        prefill = metadata.prefill
        sparse_chunks = metadata.sparse_prefill
        assert prefill is not None and sparse_chunks is not None
        k_quant_full, k_scale_full, topk_workspace = get_prefill_k_workspaces(
            total_seq_lens, head_dim, use_fp4_cache=True, with_topk_workspace=True
        )
        for chunk, rows in zip(prefill.chunks, sparse_chunks):
            k_quant, k_scale = gather_prefill_chunk_k(
                kv_cache, k_quant_full, k_scale_full, chunk
            )
            if chunk.local_total_seq_lens == 0:
                continue  # buffer already holds -1
            start, end = chunk.token_start, chunk.token_end
            rows.kernel_metadata = sparse_mqa_logits_prefill_chunk(
                q_quant[start:end].view(torch.int8),
                q_scale[start:end],
                k_quant.view(torch.int8),
                k_scale.view(torch.int32).squeeze(-1),
                weights[start:end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                candidate_blocks[start:end],
                candidate_block_size,
                sparse_block_kv,
                topk_tokens,
                topk_indices_buffer[start:end, :topk_tokens],
                sparse_indices=rows.sparse_indices,
                end=rows.end,
                col_indices=rows.col_indices,
                zero_starts=zero_starts,
                workspace=topk_workspace,
                kernel_metadata=rows.kernel_metadata,
            )

    if metadata.num_decodes > 0:
        decode_rows = metadata.sparse_decode
        assert decode_rows is not None and decode_rows.block_table is not None
        assert decode_rows.row_indices is not None
        num_rows = decode_rows.row_ke.shape[0]
        if num_rows > 0:
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            decode_rows.kernel_metadata = sparse_mqa_logits_paged_decode(
                q_quant[:num_rows].view(torch.int8).unsqueeze(1),
                q_scale[:num_rows].unsqueeze(1),
                kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache=True),
                weights[:num_rows],
                decode_rows.row_ke,
                decode_rows.block_table,
                decode_rows.row_indices,
                candidate_blocks[:num_rows],
                candidate_block_size,
                sparse_block_kv,
                topk_tokens,
                topk_indices_buffer[:num_rows, :topk_tokens],
                row_ks=decode_rows.row_ks,
                sparse_indices=decode_rows.sparse_indices,
                end=decode_rows.end,
                col_indices=decode_rows.col_indices,
                workspace=topk_workspace,
                kernel_metadata=decode_rows.kernel_metadata,
            )
    return topk_indices_buffer


def sparse_mqa_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    dense_mha_metadata_layer_name: LayerNameType,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_mqa_attn_indexer",
    op_func=sparse_mqa_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_mqa_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_mqa_indexer")
class SparseMQAIndexer(SparseAttnIndexer):
    """Candidate-consuming indexer on DeepGEMM's sparse MQA-logits kernels.

    Same constructor as `SparseAttnIndexer`; only valid for layers that read
    candidate blocks (``candidate_write=False``) with the MXFP4 indexer cache
    on SM100. The K cache is written by the model (``skip_k_cache_insert``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.candidate_blocks is None or self.candidate_write:
            raise ValueError(
                "SparseMQAIndexer is only for indexers that consume candidate "
                "blocks; the candidate source stays on SparseAttnIndexer."
            )
        if not self.use_fp4_cache:
            raise ValueError("SparseMQAIndexer requires the MXFP4 indexer cache.")
        if not self.skip_k_cache_insert:
            raise ValueError("SparseMQAIndexer expects the model to write K.")
        if self.dcp_world_size > 1 or self.use_pcp:
            raise NotImplementedError(
                "SparseMQAIndexer does not support context parallel."
            )
        if self.topk_tokens not in SPARSE_TOPK_KERNEL_SUPPORTED:
            raise ValueError(
                f"SparseMQAIndexer requires topk in {SPARSE_TOPK_KERNEL_SUPPORTED}."
            )
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_deep_gemm_sparse_mqa()
        ):
            raise ValueError(
                "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_quant, k, weights)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert isinstance(q_quant, tuple), "MXFP4 Q is a (values, scales) pair"
        q_values, q_scale = q_quant
        return torch.ops.vllm.sparse_mqa_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.candidate_blocks,
            self.candidate_block_size,
        )

    def forward_xpu(self, *args, **kwargs):
        raise NotImplementedError("SparseMQAIndexer is CUDA (SM100) only.")

    def forward_hip(self, *args, **kwargs):
        raise NotImplementedError("SparseMQAIndexer is CUDA (SM100) only.")
