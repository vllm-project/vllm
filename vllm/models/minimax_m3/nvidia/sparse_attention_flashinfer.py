# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer block-sparse attend for MiniMax M3 on SM120/SM121.

K and V are views split from the packed-content KV cache, so this impl
requires flashinfer's packed-KV support (``has_flashinfer_msa_packed_kv``).
Imported only when ``minimax_m3_use_flashinfer_msa`` passes.
"""

import torch
from flashinfer.msa_ops import (
    msa_sparse_attention,
    msa_sparse_decode_attention,
)

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import AttentionLayer


class MiniMaxM3SparseFlashInferImpl(MiniMaxM3SparseImpl):
    """flashinfer msa_ops block-sparse attend (prefill + decode)."""

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
        # The shared top-k buffer is token-major; the kernels need a
        # contiguous head-major copy.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )
        k_cache, v_cache = kv_cache.split(hd, dim=-1)
        softmax_scale = self.scale
        v_scale = None
        if self.use_fp8_kv:
            # Logits are linear in K and the output in V, so the fp8 descales
            # fold exactly into the softmax scale and the kernel output scale.
            softmax_scale *= layer._k_scale_float  # type: ignore[attr-defined]
            v_scale = layer._v_scale_float  # type: ignore[attr-defined]

        # Both kernels use the default q_offset, which right-aligns tokens to
        # match vLLM's positions.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            o = msa_sparse_decode_attention(
                q[:nd],
                k_cache,
                v_cache,
                topk[:nd].transpose(0, 1).contiguous(),
                page_table=d.block_table,
                seqused_k=d.seq_lens,
                seqlen_q=d.decode_query_len,
                causal=True,
                softmax_scale=softmax_scale,
                v_global_scale=v_scale,
            )
            out[:nd].copy_(o)

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            o = msa_sparse_attention(
                q[nd:],
                k_cache,
                v_cache,
                topk[nd:num_tokens].transpose(0, 1).contiguous(),
                p.cu_seqlens_q,
                causal=True,
                softmax_scale=softmax_scale,
                page_table=p.block_table,
                seqused_k=p.seq_lens,
                v_global_scale=v_scale,
            )
            out[nd:].copy_(o)
        return output
