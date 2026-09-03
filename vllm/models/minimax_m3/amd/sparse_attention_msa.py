# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA AITER block-sparse attend for MiniMax M3."""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    PAGE16_SIDES_PER_BLOCK,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import (
    AttentionLayer,
)


class MiniMaxM3SparseAiterPAImpl(MiniMaxM3SparseImpl):
    """ROCm AITER page-16 SHUFFLE sparse paged attention."""

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            PAGES_PER_SPARSE_BLOCK,
            minimax_m3_block_page_stride,
            minimax_m3_sparse_attn_decode_aiter,
            minimax_m3_sparse_attn_prefill_aiter,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        # Set only when the layer's indexer emitted the page table alongside its
        # selection; otherwise these stay None and the table is built below.
        sparse_bt_buf = getattr(layer, "sparse_bt_buffer", None)
        sparse_ctx_buf = getattr(layer, "sparse_ctx_buffer", None)
        kvh = self.num_kv_heads
        if kvh != 1 and sparse_bt_buf is None:
            raise NotImplementedError(
                "MiniMax-M3 AITER sparse PA needs the page table the indexer's "
                f"top-k emits to serve per-rank num_kv_heads == {kvh}; the "
                "Triton builders it falls back to address one head's cache."
            )

        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        k_cache, v_cache = layer.get_aiter_sparse_pa_kv_cache()  # type: ignore[attr-defined]
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        if sparse_bt_buf is not None:
            # An emitted table was scaled by the metadata builder, which reads
            # the packing off the layout rather than off these tensors. The two
            # only disagree if the resolved layout gives each side its own
            # plane, and a silent disagreement reads unrelated pages.
            stride = minimax_m3_block_page_stride(k_cache, v_cache)
            expected = PAGES_PER_SPARSE_BLOCK * PAGE16_SIDES_PER_BLOCK
            if stride != expected:
                raise RuntimeError(
                    "MiniMax-M3 AITER sparse PA: the indexer emitted a page "
                    f"table for a block spanning {expected} pages, but this "
                    f"KV cache lays a block out over {stride}. The page-16 "
                    "rebase assumes both K/V sides share a block."
                )

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            minimax_m3_sparse_attn_decode_aiter(
                q[:nd],
                k_cache,
                v_cache,
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                k_scale=k_scale,
                v_scale=v_scale,
                decode_query_len=d.decode_query_len,
                sparse_bt=None if sparse_bt_buf is None else sparse_bt_buf[: nd * kvh],
                sparse_ctx=(
                    None if sparse_ctx_buf is None else sparse_ctx_buf[: nd * kvh]
                ),
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            minimax_m3_sparse_attn_prefill_aiter(
                q[nd:],
                k_cache,
                v_cache,
                topk[:, nd:num_tokens, :],
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                k_scale=k_scale,
                v_scale=v_scale,
                sparse_bt=(
                    None
                    if sparse_bt_buf is None
                    else sparse_bt_buf[nd * kvh : num_tokens * kvh]
                ),
                sparse_ctx=(
                    None
                    if sparse_ctx_buf is None
                    else sparse_ctx_buf[nd * kvh : num_tokens * kvh]
                ),
            )
        return output
