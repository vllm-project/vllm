# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention of a specialized decode step.

``AttentionStep`` runs one ``DeepseekV4FlashInferMLAAttention`` layer through
the layer's own modules and kernels, in a different schedule and with
low-batch kernels where they apply:

- ``fused_wqa_wkv`` reads the MXFP8 layer input the fused all-reduce + mHC
  wrote instead of quantizing it again; up to ``WIDE_LOW_LATENCY_MAX_ROWS``
  rows it, ``wq_b`` and ``wo_b`` run on the low-latency MXFP8 GEMM, and above
  on the stock ``cute-dsl`` GEMM with pinned tactics (``mxfp8``).
- The side projections (compressor ``kv_score``, indexer weights) join their
  side streams at their first reader rather than before the query path, and
  the indexer inputs are built on side streams: the indexer query once ``qr``
  is ready and the index key right after the compressor.
- Up to ``LOW_BATCH_MAX_ROWS`` rows, the side projections and the index key's
  ``wk`` run on ``ll_bf16`` kernels (``aux_projections``), the key store reads
  ``wk``'s FP32 output, and ``wo_b`` publishes its output into the mailbox of
  the fused all-reduce that reduces it next (``boundary.publishing_gemm``).
- Dense indexers row-shard their decode rows (``indexer.decode_rows``) and
  plan them (top-k reset, slice schedule) while the current stream waits for
  the index key; the candidate source starts the candidate consumers'
  expansion on a side stream right after its indexer, and index sources first
  join the side-stream sparse-index builds, which read the top-k indices they
  overwrite.
- The sparse MLA reuses the step's sparse-index tables and runs on persistent
  KV counters (``sparse_mla``).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer
from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

from ...flashinfer_sparse import DeepseekV4FlashInferMLAAttention
from ...ops.o_proj import dsv41_o_proj
from .. import WIDE_LOW_LATENCY_MAX_ROWS
from .aux_projections import aux_projection_gemm, aux_projection_gemm_accepts
from .indexer import (
    CandidateOverlap,
    DecodeRowsPlan,
    finish_decode_rows,
    plan_decode_rows,
    rope_quant_q,
)
from .indexer_k_store import indexer_k_norm_rope_store_fp32
from .mxfp8 import (
    low_latency_mm,
    mxfp8_eligible,
    projection_mm,
    wide_or_pinned_mm,
)
from .sparse_mla import SparseIndexTables, SparseMlaDecode

if TYPE_CHECKING:
    from ...ops.cute_dsl import LamportAllGather
    from ..boundary.publishing_gemm import PublishingMxfp8Linear
    from ..runtime.decode import DecodeStep


def side_weights(attn: DeepseekV4FlashInferMLAAttention) -> list[torch.Tensor]:
    """The weights of ``attn``'s side projections."""
    weights = []
    if attn.compressor is not None:
        weights.append(attn.compressor.fused_wkv_wgate.weight)
    if attn.indexer is not None:
        weights.append(attn.indexer.weights_proj.weight)
    return weights


def key_weight(attn: DeepseekV4FlashInferMLAAttention) -> torch.Tensor | None:
    """``wk`` of ``attn``'s indexer when it produces MXFP4 index keys."""
    indexer = attn.indexer
    if indexer is not None and indexer.owns_k and indexer.use_fp4_kv:
        return indexer.wk.weight
    return None


class AttentionStep:
    """One attention layer of the decode step."""

    def __init__(
        self,
        attn: DeepseekV4FlashInferMLAAttention,
        tables: SparseIndexTables,
        mla: SparseMlaDecode,
        publisher: "PublishingMxfp8Linear | None",
        mxfp8_input: bool,
        source: CandidateOverlap | None,
        consumer: CandidateOverlap | None,
        all_gather: "LamportAllGather | None",
    ) -> None:
        self.attn = attn
        self.tables = tables
        self.mla = mla
        # Gathers the row-sharded indexer's slices.
        self.all_gather = all_gather
        self.publisher = (
            publisher
            if publisher is not None and publisher.supports(attn.wo_b, 1)
            else None
        )
        # The boundary before this layer writes its input as MXFP8 too.
        self.mxfp8_input = mxfp8_input
        # The candidate expansion this layer forks after its indexer, or joins
        # before it.
        self.source = source
        self.consumer = consumer
        self.wqa_mxfp8 = mxfp8_eligible(attn.fused_wqa_wkv)
        self.wq_b_mxfp8 = mxfp8_eligible(attn.wq_b)
        self.wo_b_mxfp8 = mxfp8_eligible(attn.wo_b)
        self.aux_gemm = aux_projection_gemm_accepts(side_weights(attn))
        wk = key_weight(attn)
        self.key_gemm = wk is not None and aux_projection_gemm_accepts([wk])
        # start, latent ready, index key done, index query done.
        self._events = [torch.cuda.Event() for _ in range(4)]

    def run(
        self,
        x: torch.Tensor,
        quant: QuantizedActivation | None,
        positions: torch.Tensor,
        step: "DecodeStep",
    ) -> torch.Tensor:
        """The layer's TP-partial output for input ``x`` (``quant``: the same
        input as MXFP8), published when ``wo_b`` publishes."""
        attn = self.attn
        out = attn._alloc_attn_out(x.shape[0], x)
        assert isinstance(out, torch.Tensor)
        qr_kv, kv_score, indexer_weights = self._input_projections(x, quant, step)
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        q, index_q, index_q_scale, index_weights, plan = self._prepare(
            qr, kv, qr_scale, kv_score, indexer_weights, positions, x.shape[0], step
        )
        indexer = attn.indexer
        if indexer is not None and index_q is not None:
            assert index_weights is not None
            self._indexer(indexer, x, index_q, index_q_scale, index_weights, plan, step)
        self.mla.run(attn, q, out)
        # The fused wo_a triggers wo_b early up to WIDE_LOW_LATENCY_MAX_ROWS,
        # where wo_b runs on the low-latency GEMM (above 8 rows, where its shape
        # has a wide tactic), which loads its weights before its own wait.
        if step.low_batch and (self.publisher is not None or self.wo_b_mxfp8):
            return dsv41_o_proj(
                attn,
                out,
                positions,
                wo_b_proj=self._wo_b_low_batch,
                early_trigger=True,
            )
        if self.wo_b_mxfp8:
            return dsv41_o_proj(
                attn,
                out,
                positions,
                wo_b_proj=self._wo_b_pinned,
                early_trigger=step.rows <= WIDE_LOW_LATENCY_MAX_ROWS,
            )
        return attn._o_proj(out, positions)

    def _input_projections(
        self,
        x: torch.Tensor,
        quant: QuantizedActivation | None,
        step: "DecodeStep",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``_run_parallel_input_projections``, with the side projections left
        unjoined on aux streams 0 and 1; ``_prepare`` joins each at its first
        reader."""
        attn = self.attn
        aux_gemm = self.aux_gemm and step.low_batch
        aux_fns: list[Callable[[], torch.Tensor] | None] = [None, None]
        if attn.compressor is not None:
            kv_weight = attn.compressor.fused_wkv_wgate.weight
            if aux_gemm:
                aux_fns[0] = lambda: aux_projection_gemm(x, kv_weight)
            else:
                aux_fns[0] = lambda: torch.mm(x, kv_weight.T, out_dtype=torch.float32)
        if attn.indexer is not None:
            weights_proj = attn.indexer.weights_proj
            if aux_gemm:
                aux_fns[1] = lambda: aux_projection_gemm(x, weights_proj.weight).to(
                    weights_proj.weight.dtype
                )
            else:
                aux_fns[1] = lambda: weights_proj(x)[0]

        streams = attn.aux_stream_list if step.overlap_inputs else None
        start = attn.ln_events[0]
        start.record()
        results: list[torch.Tensor | None] = [None, None]
        for i, fn in enumerate(aux_fns):
            if fn is None:
                continue
            if streams is None:
                results[i] = fn()
                attn.ln_events[1 + i].record()
                continue
            with torch.cuda.stream(streams[i]):
                start.wait()
                results[i] = fn()
                attn.ln_events[1 + i].record()

        if quant is None:
            qr_kv = attn._fused_wqa_wkv_gemm(x)
        else:
            qr_kv = None
            if self.wqa_mxfp8:
                qr_kv = projection_mm(attn.fused_wqa_wkv, quant, step.low_batch)
            if qr_kv is None:
                qr_kv = attn.fused_wqa_wkv(quant)[0]
        return qr_kv, results[0], results[1]

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor | None, torch.Tensor]:
        """``attn._split_qkv_and_norm``, with the fused norm triggering its
        PDL dependents early up to ``WIDE_LOW_LATENCY_MAX_ROWS`` rows: there
        this step's ``wq_b`` runs on the low-latency GEMM (above 8 rows, where
        its shape has a wide tactic), which loads its weights before its own
        wait, overlapping them with the norm."""
        attn = self.attn
        if attn.q_lora_rank % 32 or not attn._can_fuse_query_quant:
            return attn._split_qkv_and_norm(qr_kv)
        qr, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
        qr_quant, kv = fused_q_kv_rmsnorm_quant(
            qr,
            kv,
            attn.q_norm.weight.data,
            attn.kv_norm.weight.data,
            attn.eps,
            early_trigger=qr.shape[0] <= WIDE_LOW_LATENCY_MAX_ROWS,
        )
        return qr_quant, None, kv

    def _prepare(
        self,
        qr: torch.Tensor | QuantizedActivation,
        kv: torch.Tensor,
        qr_scale: torch.Tensor | None,
        kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        num_tokens: int,
        step: "DecodeStep",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        DecodeRowsPlan | None,
    ]:
        """``_prepare_and_attn`` up to the indexer, with the indexer inputs on
        side streams.

        From the moment ``qr`` is ready, aux stream 2 runs the indexer query
        (``wq_b``, then RoPE/quantize once the weights GEMM is done) and aux
        stream 0, after its ``kv_score`` GEMM, the compressor and the index
        key; the current stream runs the main query and KV insertion, a dense
        indexer's plan (``plan_decode_rows``), then the compressed-cache
        insertion once the latent is ready. Returns the query, the indexer
        query, its scale and weights, and the dense indexer's plan.
        """
        attn = self.attn
        indexer, compressor = attn.indexer, attn.compressor
        streams = attn.aux_stream_list
        assert streams is not None
        start, latent_ready, index_k_done, index_q_done = self._events
        start.record()
        index_q: torch.Tensor | None = None
        index_q_scale: torch.Tensor | None = None
        index_weights: torch.Tensor | None = None
        if indexer is not None:
            assert indexer_weights is not None
            with torch.cuda.stream(streams[2]):
                start.wait()
                index_q_pre = indexer._wq_b_proj(qr, qr_scale)
                attn.ln_events[2].wait()
                index_q, index_q_scale, index_weights = rope_quant_q(
                    indexer,
                    index_q_pre,
                    indexer_weights,
                    positions,
                    attn.indexer_rotary_emb,
                )
                index_q_done.record()
        latent = None
        if compressor is not None:
            assert kv_score is not None
            # After the kv_score GEMM, which ran on this stream.
            with torch.cuda.stream(streams[0]):
                start.wait()
                latent = compressor(kv_score, positions)
                latent_ready.record()
                if indexer is not None and indexer.owns_k:
                    self._produce_k(indexer, latent, positions, step)
                index_k_done.record()

        q = None
        if self.wq_b_mxfp8:
            q = projection_mm(attn.wq_b, qr, step.low_batch)
        if q is None:
            q = attn._wq_b_proj(qr, qr_scale)
        q = q.view(-1, attn.n_local_heads, attn.head_dim)
        q = attn._fused_qnorm_rope_kv_insert(
            q, kv, positions, get_forward_context().attn_metadata
        )
        # The plan reads neither the index query nor the keys: this stream
        # would otherwise wait for them here.
        plan = None
        if indexer is not None and isinstance(indexer.indexer_op, SparseAttnIndexer):
            # The side-stream table builds read the top-k rows it resets.
            self.tables.join()
            plan = plan_decode_rows(indexer.indexer_op, num_tokens)
        if compressor is not None:
            latent_ready.wait()
            compressor.insert_cache(latent, positions, attn.rotary_emb)
            index_k_done.wait()
        if indexer is not None:
            index_q_done.wait()
        return q, index_q, index_q_scale, index_weights, plan

    def _produce_k(
        self,
        indexer: DeepseekV4Indexer,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        step: "DecodeStep",
    ) -> None:
        """``indexer._produce_k``; up to ``LOW_BATCH_MAX_ROWS`` rows, ``wk`` on
        ``ll_bf16`` and the FP32 key store."""
        rotary_emb = self.attn.indexer_rotary_emb
        attn_metadata = get_forward_context().attn_metadata
        if (
            not self.key_gemm
            or not step.low_batch
            or latent is None
            or not isinstance(attn_metadata, dict)
            or latent.dtype != torch.bfloat16
            or not latent.is_contiguous()
        ):
            indexer._produce_k(latent, positions, rotary_emb)
            return
        indexer_metadata = cast(Any, attn_metadata[indexer.k_cache.prefix])
        indexer_k_norm_rope_store_fp32(
            aux_projection_gemm(latent, indexer.wk.weight),
            positions,
            rotary_emb.cos_sin_cache,
            indexer.k_norm.weight,
            indexer.k_norm.variance_epsilon,
            indexer.k_cache.kv_cache,
            indexer_metadata.slot_mapping,
            indexer.compress_ratio,
        )

    def _indexer(
        self,
        indexer: DeepseekV4Indexer,
        x: torch.Tensor,
        index_q: torch.Tensor,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor,
        plan: DecodeRowsPlan | None,
        step: "DecodeStep",
    ) -> None:
        """The layer's indexer: the top-k indices (and in the candidate
        source, the candidate blocks) of the step; a dense indexer finishes
        its ``plan``."""
        q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
        op = indexer.indexer_op
        if isinstance(op, SparseAttnIndexer):
            finish_decode_rows(op, x, q_quant, index_weights, plan, self.all_gather)
        else:
            # The side-stream table builds read the top-k indices this
            # overwrites.
            self.tables.join()
            if self.consumer is not None:
                self.consumer.join()
            op(x, q_quant, None, index_weights)
        if self.source is not None and self.source.warmed:
            self.source.fork()

    def _wo_b_low_batch(self, z: torch.Tensor | QuantizedActivation) -> torch.Tensor:
        """``wo_b`` up to ``LOW_BATCH_MAX_ROWS`` rows: published, or on the
        low-latency MXFP8 GEMM."""
        wo_b = self.attn.wo_b
        if self.publisher is not None:
            return self.publisher(wo_b, z)
        out = low_latency_mm(wo_b, z) if self.wo_b_mxfp8 else None
        return out if out is not None else self.attn._wo_b_proj(z)

    def _wo_b_pinned(self, z: torch.Tensor | QuantizedActivation) -> torch.Tensor:
        """``wo_b`` above ``LOW_BATCH_MAX_ROWS`` rows: on the wide low-latency
        or the pinned ``cute-dsl`` GEMM where one runs."""
        out = wide_or_pinned_mm(self.attn.wo_b, z)
        return out if out is not None else self.attn._wo_b_proj(z)
