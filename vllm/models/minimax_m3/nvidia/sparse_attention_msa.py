# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) block-sparse attend for MiniMax M3.

Prefill attends with ``fmha_sm100`` (``build_k2q_csr`` + ``sparse_atten_func``);
decode falls back to the Triton split-K kernel (no MSA decode yet). ``fmha_sm100``
imports are function-local, so this module is import-safe on AMD/non-SM100.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import AttentionLayer


class MiniMaxM3SparseMSAImpl(MiniMaxM3SparseImpl):
    """MSA block-sparse attend (``fmha_sm100``); Triton split-K decode."""

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer [total_q, H, MK]; the
        # kernels want [H, tokens, MK], so slice tokens on dim 0 then transpose.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )

        # Decode [:nd]: Triton split-K placeholder (no MSA decode yet).
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            minimax_m3_sparse_attn_decode(
                q[:nd],
                kv_cache,
                topk[:nd].transpose(0, 1),
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
            )

        # Prefill [nd:]: MSA sparse FMHA over the selected blocks.
        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.sparse import (
                build_k2q_csr,
                sparse_atten_func,
            )

            p = main_md.prefill
            assert p is not None
            # [H, prefill, MK] transposed view; build_k2q_csr consumes the
            # strided view directly (topK stays innermost-contiguous).
            prefill_topk = topk[nd:num_tokens].transpose(0, 1)
            qp = q[nd:]
            k_cache, v_cache = kv_cache.split(self.head_size, dim=-1)
            k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
                prefill_topk,
                p.cu_seqlens_q,
                p.cu_seqlens_k,
                SPARSE_BLOCK_SIZE,
                total_k=0,
                max_seqlen_k=p.max_seq_len,
                max_seqlen_q=p.max_query_len,
                total_rows=p.total_kv_blocks,
                qhead_per_kv=qp.shape[1] // self.num_kv_heads,
                return_schedule=True,
            )
            sparse_atten_func(
                qp,
                k_cache,
                v_cache,
                k2q_row_ptr,
                k2q_q_indices,
                topK=self.topk_blocks,
                blk_kv=SPARSE_BLOCK_SIZE,
                causal=True,
                softmax_scale=self.scale,
                cu_seqlens_q=p.cu_seqlens_q,
                cu_seqlens_k=p.cu_seqlens_k,
                max_seqlen_q=p.max_query_len,
                max_seqlen_k=p.max_seq_len,
                page_table=p.block_table,
                seqused_k=p.seq_lens,
                schedule=schedule,
                out=out[nd:],
            )
        return output
