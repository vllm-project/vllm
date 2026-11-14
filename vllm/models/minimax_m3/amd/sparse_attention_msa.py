# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA AITER block-sparse attend for MiniMax M3."""

from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionLayer,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Opt-in dense MSA prefill: see MiniMaxM3SparseAiterPAImpl._dense_prefill.
_DENSE_PREFILL_MAX = envs.VLLM_MINIMAX_M3_MSA_DENSE_PREFILL
_DENSE_PREFILL_CHECK = envs.VLLM_MINIMAX_M3_MSA_DENSE_PREFILL_CHECK
# Positions below this see every causal block: top-k blocks x SPARSE_BLOCK_SIZE
# (16 x 128 for MiniMax-M3). The builder's tail list uses it; the impl asserts it
# against the layer's top-k width.
_DENSE_TOPK_TOKENS = 2048


class MiniMaxM3SparseAiterPABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend carrying the AITER page-16 block-table rebase."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseAiterPAMetadataBuilder"]:
        return MiniMaxM3SparseAiterPAMetadataBuilder


@dataclass
class MiniMaxM3SparseAiterPAPrefillMetadata(MiniMaxM3SparsePrefillMetadata):
    # ``block_table`` rebased onto AITER's page-16 numbering; see
    # ``MiniMaxM3SparseAiterPAMetadataBuilder.build``.
    page16_block_table: torch.Tensor | None = None
    # Dense prefill path: set when every prefill request of the step fits it
    dense_page_table: torch.Tensor | None = (
        None  # [num_prefills, pages] page ids of the K view
    )
    dense_max_seq_len: int = 0
    dense_tail_idx: torch.Tensor | None = (
        None  # prefill-relative rows at positions >= topk tokens
    )
    dense_num_tail: int = 0
    # 16-byte aligned copy of seq_lens for the dense call (one kernel specialization)
    dense_seq_lens: torch.Tensor | None = None


@dataclass
class MiniMaxM3SparseAiterPADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    page16_block_table: torch.Tensor | None = None


class MiniMaxM3SparseAiterPAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Adds the page-16 rebase of the block table the indexer's top-k needs.

    The AITER indexer's top-k emits the attend's page table, expanding a
    selected block into a compile-time number of pages -- one side's worth,
    while an interleaved block holds both -- so it has to resolve the selection
    through a table pre-scaled to the wider stride. Every sparse layer selects
    through the same table, so it is rebased here once per step rather than
    once per layer, alongside the slot mapping the base rebases for the writer.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Stable address for the same reason as the base's own buffers: a
        # captured decode replay writes through the address it captured.
        self.page16_block_table_buffer = torch.empty(
            (
                vllm_config.scheduler_config.max_num_seqs,
                cdiv(vllm_config.model_config.max_model_len, kv_cache_spec.block_size),
            ),
            dtype=torch.int32,
            device=device,
        )
        # Opt-in dense MSA prefill
        self._dense_prefill_on = _DENSE_PREFILL_MAX > 0
        if self._dense_prefill_on:
            from vllm.models.minimax_m3.amd.ops.sparse_pa import (
                PAGE16_SIDES_PER_BLOCK,
                PAGES_PER_SPARSE_BLOCK,
            )

            # Same page ids the sparse kernels read: a block-table entry is one
            # SPARSE_BLOCK_SIZE block whose K and V sides interleave, so its j-th K
            # page is entry * (pages of both sides) + j and V sits at the same id in
            # the V view (the impl checks the cache is laid out that way).
            self._dense_page_stride = PAGES_PER_SPARSE_BLOCK * PAGE16_SIDES_PER_BLOCK
            self._dense_pages_per_block = PAGES_PER_SPARSE_BLOCK
            self._dense_page_offsets = torch.arange(
                PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device=device
            )
            logger.info(
                "MiniMax-M3 MSA prefill: dense attention for steps whose prefill "
                "requests are all <= %d tokens (queries at positions >= %d "
                "recomputed sparse)",
                _DENSE_PREFILL_MAX,
                _DENSE_TOPK_TOKENS,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_rebase_block_table_to_page16,
        )

        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        block_table = common_attn_metadata.block_table_tensor
        num_reqs, cols = block_table.shape
        assert (
            num_reqs <= self.page16_block_table_buffer.shape[0]
            and cols <= self.page16_block_table_buffer.shape[1]
        ), (
            f"block table {(num_reqs, cols)} exceeds the page-16 rebase buffer "
            f"{tuple(self.page16_block_table_buffer.shape)}"
        )
        page16 = minimax_m3_rebase_block_table_to_page16(
            block_table,
            out=self.page16_block_table_buffer[:num_reqs, :cols],
        )

        # Decode-first batch, so each side takes the rows its own block table
        # was sliced from. Reconstructed from the base's fields rather than
        # listed out, so a field added there does not silently drop here.
        nd = metadata.num_decodes
        if metadata.decode is not None:
            metadata.decode = MiniMaxM3SparseAiterPADecodeMetadata(
                **vars(metadata.decode),
                page16_block_table=page16[:nd],
            )
        if metadata.prefill is not None:
            metadata.prefill = MiniMaxM3SparseAiterPAPrefillMetadata(
                **vars(metadata.prefill),
                page16_block_table=page16[nd:],
            )
            if self._dense_prefill_on:
                self._dense_prefill_metadata(
                    metadata, common_attn_metadata, block_table
                )
        return metadata

    def _dense_prefill_metadata(
        self, metadata, common_attn_metadata, block_table
    ) -> None:
        """Per-page table and tail rows for the dense prefill path (CPU lengths)."""
        seq_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        if seq_cpu is None or qsl_cpu is None:
            return
        nd = metadata.num_decodes
        p_seq = seq_cpu[nd : metadata.num_decodes + metadata.num_prefills].to(
            torch.int64
        )
        if p_seq.numel() == 0:
            return
        pmax = int(p_seq.max())
        if pmax > _DENSE_PREFILL_MAX:
            return
        from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

        p = metadata.prefill
        ncols = cdiv(pmax, SPARSE_BLOCK_SIZE)
        # Even block count, so the page-table row stride (ncols x 8 pages) is a
        # multiple of 16 and the attention kernel keeps one specialization.
        ncols += ncols & 1
        bt = block_table[nd : nd + metadata.num_prefills, :ncols]
        p.dense_page_table = (
            bt.unsqueeze(-1) * self._dense_page_stride + self._dense_page_offsets
        ).reshape(bt.shape[0], ncols * self._dense_pages_per_block)
        p.dense_max_seq_len = pmax
        # seq_lens[num_decodes:] is 16-byte aligned only when num_decodes % 4 == 0.
        p.dense_seq_lens = p.seq_lens.clone()
        if pmax > _DENSE_TOPK_TOKENS:
            q_lo = qsl_cpu[nd : metadata.num_decodes + metadata.num_prefills].to(
                torch.int64
            )
            q_hi = qsl_cpu[
                nd + 1 : metadata.num_decodes + metadata.num_prefills + 1
            ].to(torch.int64)
            ctx = p_seq - (q_hi - q_lo)
            start = q_lo + torch.clamp(_DENSE_TOPK_TOKENS - ctx, min=0)
            base = metadata.num_decode_tokens
            parts = [
                torch.arange(int(s) - base, int(e) - base)
                for s, e in zip(start, q_hi)
                if int(s) < int(e)
            ]
            if parts:
                tail = torch.cat(parts)  # int64: index_select / index_copy_ indices
                p.dense_tail_idx = tail.pin_memory().to(
                    block_table.device, non_blocking=True
                )
                p.dense_num_tail = int(tail.numel())


class MiniMaxM3SparseAiterPAImpl(MiniMaxM3SparseImpl):
    """ROCm AITER page-16 SHUFFLE sparse paged attention."""

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        decode_sparse_table: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            PAGE16_SIDES_PER_BLOCK,
            PAGES_PER_SPARSE_BLOCK,
            minimax_m3_sparse_attn_decode_aiter,
            minimax_m3_sparse_attn_prefill_aiter,
            minimax_m3_sparse_block_page_stride,
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
            stride = minimax_m3_sparse_block_page_stride(k_cache, v_cache)
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
            prepared_decode_table = decode_sparse_table
            if sparse_bt_buf is not None:
                assert sparse_ctx_buf is not None
                prepared_decode_table = (
                    sparse_bt_buf[: nd * kvh],
                    sparse_ctx_buf[: nd * kvh],
                )
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
                sparse_block_table=(
                    prepared_decode_table[0]
                    if prepared_decode_table is not None
                    else None
                ),
                sparse_context_lens=(
                    prepared_decode_table[1]
                    if prepared_decode_table is not None
                    else None
                ),
            )

        if (
            main_md.num_prefills > 0
            and getattr(main_md.prefill, "dense_page_table", None) is not None
        ):
            p = main_md.prefill
            self._dense_prefill(
                layer,
                q[nd:num_tokens],
                out[nd:num_tokens],
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                p,
                topk[:, nd:num_tokens, :],
                None
                if sparse_bt_buf is None
                else sparse_bt_buf[nd * kvh : num_tokens * kvh],
                None
                if sparse_ctx_buf is None
                else sparse_ctx_buf[nd * kvh : num_tokens * kvh],
            )
        elif main_md.num_prefills > 0:
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

    # Dense MSA prefill state
    _dense_q_descale: torch.Tensor | None = None
    # Dense calls left to compare with the sparse kernel, per process:
    # [steps without tail rows, steps with them]
    _dense_checks_left: list[int] = [_DENSE_PREFILL_CHECK, _DENSE_PREFILL_CHECK]

    def _sparse_prefill_rows(
        self,
        q_p,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        p,
        topk_p,
        sparse_bt,
        sparse_ctx,
        out_p,
    ):
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_sparse_attn_prefill_aiter,
        )

        minimax_m3_sparse_attn_prefill_aiter(
            q_p,
            k_cache,
            v_cache,
            topk_p,
            p.block_table,
            p.query_req_id,
            p.query_abs_pos,
            self.num_kv_heads,
            self.scale,
            out_p,
            k_scale=k_scale,
            v_scale=v_scale,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
        )

    def _dense_prefill(
        self,
        layer,
        q_p,
        out_p,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        p,
        topk_p,
        sparse_bt,
        sparse_ctx,
    ) -> None:
        """Dense causal attention for the prefill rows (all of them see every causal
        block below top-k tokens), then the rows at positions >= top-k tokens
        recomputed by the sparse kernel."""
        from aiter.ops.triton.unified_attention import unified_attention

        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            PAGE16_SIDES_PER_BLOCK,
            PAGES_PER_SPARSE_BLOCK,
            _run_gluon_decode,
            _sides_are_packed,
            minimax_m3_build_sparse_block_table_prefill,
            minimax_m3_sparse_block_page_stride,
        )
        from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

        assert topk_p.shape[-1] * SPARSE_BLOCK_SIZE == _DENSE_TOPK_TOKENS, (
            "dense MSA prefill assumes top-"
            f"{_DENSE_TOPK_TOKENS // SPARSE_BLOCK_SIZE} blocks, "
            f"the layer selects {topk_p.shape[-1]}"
        )
        assert (
            minimax_m3_sparse_block_page_stride(k_cache, v_cache)
            == PAGES_PER_SPARSE_BLOCK * PAGE16_SIDES_PER_BLOCK
        ), "dense MSA prefill builds its page table for K/V sides sharing a block"
        kvh = self.num_kv_heads
        k_d = (
            k_cache[: v_cache.shape[0]]
            if _sides_are_packed(k_cache, v_cache)
            else k_cache
        )
        if self._dense_q_descale is None or self._dense_q_descale.device != q_p.device:
            self._dense_q_descale = torch.ones(
                1, dtype=torch.float32, device=q_p.device
            )
        q8 = q_p.to(k_cache.dtype) if self.use_fp8_kv else q_p
        unified_attention(
            q=q8,
            k=k_d,
            v=v_cache,
            out=out_p,
            cu_seqlens_q=p.cu_seqlens_q,
            max_seqlen_q=p.max_query_len,
            seqused_k=p.dense_seq_lens,
            max_seqlen_k=p.dense_max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=p.dense_page_table,
            softcap=0,
            q_descale=self._dense_q_descale if self.use_fp8_kv else None,
            k_descale=k_scale,
            v_descale=v_scale,
            shuffled_kv_cache=True,
        )
        kind = int(p.dense_num_tail > 0)
        check = self._dense_checks_left[kind] > 0
        if p.dense_num_tail == 0 and not check:
            return
        if sparse_bt is None:
            sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
                topk_p,
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                minimax_m3_sparse_block_page_stride(k_cache, v_cache),
            )
        if check:
            ref = torch.empty_like(out_p)
            self._sparse_prefill_rows(
                q_p,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                p,
                topk_p,
                sparse_bt,
                sparse_ctx,
                ref,
            )
            head = torch.ones(out_p.shape[0], dtype=torch.bool, device=out_p.device)
            if p.dense_num_tail:
                head[p.dense_tail_idx] = False
            # Startup dummy runs (all-zero inputs, requests of <= 16 tokens)
            # compare nothing: keep the budget.
            if p.dense_max_seq_len <= 16 or not bool(ref.abs().amax().ne(0)):
                if p.dense_num_tail:
                    out_p.index_copy_(
                        0, p.dense_tail_idx, ref.index_select(0, p.dense_tail_idx)
                    )
                return
            d = (out_p.float() - ref.float()).abs()
            dh = d[head]
            self._dense_checks_left[kind] -= 1
            logger.info(
                "MSA dense prefill check (%s): %d prefill rows, %d tail rows, "
                "max seq %d; head rows |dense - sparse| max %.4g mean %.3g "
                "(|sparse| max %.3g mean %.3g)",
                getattr(layer, "layer_name", "?"),
                out_p.shape[0],
                p.dense_num_tail,
                p.dense_max_seq_len,
                float(dh.max()) if dh.numel() else 0.0,
                float(dh.mean()) if dh.numel() else 0.0,
                float(ref.abs().max()),
                float(ref.float().abs().mean()),
            )
            if p.dense_num_tail:
                idx = p.dense_tail_idx
                out_p.index_copy_(0, idx, ref.index_select(0, idx))
            return
        idx = p.dense_tail_idx
        rows = (
            idx
            if kvh == 1
            else (
                idx.unsqueeze(1) * kvh + torch.arange(kvh, device=idx.device)
            ).reshape(-1)
        )
        q_t = q_p.index_select(0, idx)
        o_t = torch.empty_like(q_t)
        _run_gluon_decode(
            q_t,
            k_cache,
            v_cache,
            sparse_bt.index_select(0, rows),
            sparse_ctx.index_select(0, rows),
            kvh,
            self.scale,
            o_t,
            k_scale,
            v_scale,
        )
        out_p.index_copy_(0, idx, o_t)
