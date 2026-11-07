# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer stage of a K3.1 DSA layer: the stock decode scoring and top-k
(sparse_attn_indexer, its key insert skipped: DsaIndex inserted the key) of the FP8
or MXFP4 (index.py) indexer query and head weights into the model's shared top-k
buffer. Under DCP the
op merges the ranks' local top-k into global token ids.

Under DCP on the FlashInfer core, when the layer's indexer has the one-round DCP
top-k merge (``dcp_local_topk``, Inferact/mum#63), ``merge_rows`` replaces the op
and the local-slot filter: it scores the rows (DeepGEMM paged MQA logits, bf16)
and merges them straight into this rank's compacted physical KV rows and counts.
With the fused DCP indexer (``fused``, indexer.py) the scoring is its scorer (the
same logits layout) and, packed, its select also takes the local top-N (the
merge's B' rows), so the merge only all-gathers and selects."""

from typing import Any

import torch

from vllm.utils.torch_utils import _encode_layer_name

from .index import DIM, fp4_query_views, is_fp4_cache


class IndexerStage:
    """``indexer``: the layer's KimiK3DSAIndexer; ``rows``: the most decode rows
    of a step (sizes the DCP merge's outputs; 0: no direct DCP merge)."""

    def __init__(self, indexer, rows=0):
        self.indexer = indexer
        self.k_cache_name = _encode_layer_name(indexer.k_cache.prefix)
        self.max_model_len = indexer.max_model_len
        self.max_total = indexer.max_total_seq_len
        self.topk = indexer.topk_tokens
        # --- DCP top-k merge (#63): its caller-owned outputs -------------------
        # N, this rank's candidate count; 0 runs the op and the local-slot filter.
        self.local_topk = indexer.dcp_local_topk if rows else 0
        if self.local_topk:
            device = indexer.topk_indices_buffer.device
            # K wide: the exact merge gives a rank up to K of a row's keys (a
            # saturated row: more than its N local candidates).
            shape = (rows, self.topk)
            self.dcp_rows = torch.full(shape, -1, dtype=torch.int32, device=device)
            self.dcp_valid = torch.zeros(rows, dtype=torch.int32, device=device)
        # Optional [>= rows, K] int32: the merge also writes the global ids (tests).
        self.dcp_ids: torch.Tensor | None = None
        # The fused DCP indexer (PreparedDsa.prepare_cache; None: DeepGEMM).
        self.fused = None
        # MXFP4 keys and query (indexer_kv_dtype="mxfp4", kimi_k3_indexer_uses_fp4).
        self.fp4 = indexer.use_fp4_cache

    def query(self, q_fp8, rows):
        """The stock op's (query, its scales or None) of the first ``rows`` rows:
        FP8 [rows, 32, 128] (scales folded into the head weights), or MXFP4 E2M1
        [rows, 32, 64] and UE8M0 x4 [rows, 32] (contiguous copies of the views in
        the query rows, index.py)."""
        if not self.fp4:
            return q_fp8[:rows].view(torch.float8_e4m3fn), None
        packed, scales = fp4_query_views(q_fp8, rows)
        return packed.contiguous(), scales.contiguous()
        # -----------------------------------------------------------------------

    def run(self, projected, q_fp8, head_w):
        """Top-k of the current rows into the shared buffer (stock decode scoring)."""
        indexer, batch = self.indexer, projected.shape[0]
        assert is_fp4_cache(indexer.k_cache.kv_cache) == self.fp4
        query, q_scale = self.query(q_fp8, batch)
        return torch.ops.vllm.sparse_attn_indexer(
            projected,
            self.k_cache_name,
            indexer.k_cache.kv_cache,
            query,
            q_scale,
            None,
            head_w[:batch],
            DIM,
            indexer.scale_fmt,
            self.topk,
            DIM,
            self.max_model_len,
            self.max_total,
            indexer.topk_indices_buffer,
            True,  # the key is inserted by DsaIndex
            False,
            False,
            _encode_layer_name(""),
            self.fp4,
            indexer.dcp_rank,
            indexer.dcp_world_size,
            indexer.cp_kv_cache_interleave_size,
            topk_backend=indexer.topk_backend,
        )

    # --- DCP top-k merge (#63) ---------------------------------------------------
    def merge_rows(self, q_fp8, head_w, metadata, stride_rows, batch, q=1):
        """DCP: this rank's flat KV rows of the merged global top-k of the first
        ``batch`` rows (``q`` per request; compacted to a prefix, at most N per
        row) and their counts, as ``dcp_stage.local_slots`` of the op's global
        ids; ``metadata``: the MLA layer's. None when the step does not fit the
        merge (the caller then runs the op and the filter). One all-gather over
        the DCP group, so every rank takes the same branch. The logits must hold
        no NaN: DeepSelect then returns out-of-range ids, on which the merge's
        pack faults; no per-step check here (finite query, head weights and
        cached keys keep them finite)."""
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.indexer_topk import (
            SparseIndexerTopk,
            is_deep_select_supported,
        )
        from vllm.model_executor.layers.sparse_attn_indexer import (
            kv_cache_as_quant_view,
        )
        from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge import dcp_topk_merge_fits
        from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
            dcp_topk_merge_rows,
        )
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits,
            mqa_logits_mx_inputs,
        )
        from vllm.utils.math_utils import cdiv

        indexer, n = self.indexer, self.local_topk
        index_md: Any = get_forward_context().attn_metadata[indexer.k_cache.prefix]
        decode, rows = index_md.decode, index_md.num_decode_tokens
        if (
            decode is None
            or decode.global_seq_lens is None
            or index_md.num_prefills
            or not 0 < rows <= batch
            or decode.requires_padding
            or rows % decode.decode_lens.shape[0]
            or indexer.cp_kv_cache_interleave_size != 1
            or not dcp_topk_merge_fits(rows, self.topk, n)
        ):
            return None
        slots, valid = self.dcp_rows[:batch], self.dcp_valid[:batch]
        ids = None if self.dcp_ids is None else self.dcp_ids[:rows, : self.topk]
        # As the stock hook: interleave 1 bounds the local lengths.
        max_local_len = cdiv(self.max_model_len, indexer.dcp_world_size)
        merge = dict(
            local_topk=n,
            topk=self.topk,
            req_id=metadata.req_id_per_token,
            block_table=metadata.block_table,
            block_size=metadata.block_size,
            stride_rows=stride_rows,
            dcp_rank=indexer.dcp_rank,
            rows_out=slots[:rows],
            valid_out=valid[:rows],
            ids_out=ids,
        )
        fused = self.fused
        if (
            fused is not None
            and decode.seq_lens.numel() == rows
            and (batch, q) in fused.exchange_kernels
        ):
            # NVLink candidate exchange + exact merge (topk_exchange): every
            # row of the batch (rows past the requests have no keys, so they
            # come out empty, as the fill below).
            # The merge maps ids to rows of fused.page-token MLA blocks.
            assert metadata.block_size == fused.page, (metadata.block_size, fused.page)
            ids = None if self.dcp_ids is None else self.dcp_ids[:batch, : self.topk]
            args = metadata.req_id_per_token, metadata.block_table, stride_rows
            exchange = (*args, slots, valid, max_local_len)
            fused.run_exchange(q_fp8, head_w, decode, batch, q, exchange, ids)
            return slots, valid
        if fused is not None and decode.seq_lens.numel() == rows:
            # The logits stay valid until the merge returns (stream order): a
            # saturated rank re-reads its row in the exact merge's second round.
            packed, logits, ends = fused.run_dcp(q_fp8, head_w, decode, batch, q)
            dcp_topk_merge_rows(
                logits=logits[:rows],
                row_ends=ends[:rows],
                packed=None if packed is None else packed[:rows],
                max_local_len=max_local_len,
                **merge,
            )
            if rows < batch:  # rows past the requests attend nothing
                slots[rows:].fill_(-1)
                valid[rows:].zero_()
            return slots, valid
        # As the stock op's decode branch (uniform rows; FP8 or MXFP4 cache and
        # query).
        reqs = decode.decode_lens.shape[0]
        next_n = rows // reqs
        query, q_scale = self.query(q_fp8, rows)
        query = query.view(reqs, next_n, *query.shape[1:])
        if q_scale is not None:
            q_scale = q_scale.view(reqs, next_n, -1)
        q_scale, weights = mqa_logits_mx_inputs(
            query, q_scale, head_w[:rows], torch.bfloat16
        )
        seq_lens = decode.seq_lens[:reqs]
        logits = fp8_fp4_paged_mqa_logits(
            (query, q_scale),
            kv_cache_as_quant_view(indexer.k_cache.kv_cache, DIM, self.fp4),
            weights,
            seq_lens,
            decode.block_table,
            decode.schedule_metadata,
            max_model_len=self.max_model_len,
            logits_dtype=torch.bfloat16,
            indices=decode.indices,
        )
        # SM100 DeepGEMM emits bf16 at a 1 KiB row stride (DeepSelect's input).
        assert logits.dtype == torch.bfloat16 and is_deep_select_supported(logits, n)
        dcp_topk_merge_rows(
            logits=logits,
            row_ends=SparseIndexerTopk._row_ends(seq_lens, next_n, rows),
            max_local_len=max_local_len,
            **merge,
        )
        if rows < batch:  # rows past the requests attend nothing
            slots[rows:].fill_(-1)
            valid[rows:].zero_()
        return slots, valid

    # -------------------------------------------------------------------------------
