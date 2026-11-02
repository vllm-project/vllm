# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The steps of a Hopper executor layer: the executor's attention, MoE and
boundary steps with the Hopper kernel choices (``kernels``)."""

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

from ..attention.step import AttentionStep, StepMla, StepTables
from ..boundary.boundary import Boundaries, BoundaryState
from ..moe.step import MoEStep
from . import attn_chain, fused_indexer, kernels
from .moe import HopperLowBatchMoE
from .sparse_mla import FlashMlaDecode

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    from ...flashmla import DeepseekV4FlashMLAAttention
    from ...model import DeepseekV4MoE
    from ..boundary.all_reduce import BoundaryAllReduce
    from ..runtime.decode import DecodeStep
    from .fused_indexer import FusedDecodeIndexer


class HopperAttentionStep(AttentionStep):
    """``AttentionStep`` of a ``DeepseekV4FlashMLAAttention`` layer: the same
    side-stream schedule, on the Hopper projections (``linears``), with
    ``wo_b`` publishing into ``all_reduce`` where its kernel can. Its layer
    input is BF16 (the dense linears are weight-only W8A16), so ``quant`` is
    always None.

    Steps on the small-batch MLA of up to ``attn_chain.MAX_ROWS`` rows run the
    early-trigger small kernels (``attn_chain``): the q/kv RMSNorm with the KV
    insert, the query RoPE, and the MLA combine with the inverse RoPE, whose
    output ``wo_a`` reads directly."""

    def __init__(
        self,
        attn: "DeepseekV4FlashMLAAttention",
        tables: StepTables,
        mla: StepMla,
        all_reduce: "BoundaryAllReduce",
        fused_indexer: "FusedDecodeIndexer | None" = None,
    ) -> None:
        super().__init__(attn, tables, mla, None, False, None, None)
        self.linears = kernels.AttentionLinears.of(attn)
        self.all_reduce = all_reduce
        self.fused_indexer = fused_indexer
        self.chain = attn_chain.applies(attn) and isinstance(mla, FlashMlaDecode)
        self._chain_step = False

    def run(
        self,
        x: torch.Tensor,
        quant: QuantizedActivation | None,
        positions: torch.Tensor,
        step: "DecodeStep",
    ) -> torch.Tensor:
        rows = x.shape[0]
        mla = self.mla
        if not (
            self.chain
            and isinstance(mla, FlashMlaDecode)
            and mla.chain_step(rows)
            and self.linears.wo_a is not None
            and self.linears.wo_a.serves(rows)
        ):
            return super().run(x, quant, positions, step)
        attn = self.attn
        by_layer = get_forward_context().attn_metadata
        assert isinstance(by_layer, dict)
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata", by_layer[attn.swa_cache_layer.prefix]
        )
        qr_kv, kv_score, indexer_weights = self._input_projections(x, None, step)
        qr = attn_chain.norm_kv_insert(attn, qr_kv, positions, swa_metadata)
        self._chain_step = True
        try:
            q, index_q, index_q_scale, index_weights = self._prepare(
                qr, None, None, kv_score, indexer_weights, positions, step
            )
        finally:
            self._chain_step = False
        indexer = attn.indexer
        if indexer is not None and index_q is not None:
            assert index_weights is not None
            self._indexer(indexer, x, index_q, index_q_scale, index_weights, step)
        rotated = mla.run_rotated(attn, q, positions)
        return kernels.wo_b(
            attn, self.linears, self.linears.wo_a(rotated), self.all_reduce
        )

    def _indexer(
        self,
        indexer: DeepseekV4Indexer,
        x: torch.Tensor,
        index_q: torch.Tensor,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor,
        step: "DecodeStep",
    ) -> None:
        fused = self.fused_indexer
        op = indexer.indexer_op
        if fused is not None and index_q_scale is None and fused_indexer.applies(op):
            by_layer = get_forward_context().attn_metadata
            assert isinstance(by_layer, dict)
            metadata: Any = by_layer.get(op.k_cache.prefix)
            swa: Any = by_layer[self.attn.swa_cache_layer.prefix]
            query_len = swa.max_decode_query_len
            if swa.num_decode_tokens == swa.num_decodes * query_len and fused.serves(
                metadata, query_len
            ):
                # The side-stream table builds read the top-k indices.
                self.tables.join()
                fused(op, index_q, index_weights, metadata, query_len)
                return
        super()._indexer(indexer, x, index_q, index_q_scale, index_weights, step)

    def _q_rope_kv_insert(
        self, q: torch.Tensor, kv: torch.Tensor | None, positions: torch.Tensor
    ) -> torch.Tensor:
        if self._chain_step:
            return attn_chain.q_rope(self.attn, q, positions)
        return super()._q_rope_kv_insert(q, kv, positions)

    def _wqa_wkv(
        self,
        x: torch.Tensor,
        quant: QuantizedActivation | None,
        step: "DecodeStep",
    ) -> torch.Tensor:
        assert quant is None
        return kernels.wqa_wkv(self.attn, self.linears, x)

    def _wq_b(
        self,
        qr: torch.Tensor | QuantizedActivation,
        qr_scale: torch.Tensor | None,
        step: "DecodeStep",
    ) -> torch.Tensor:
        assert isinstance(qr, torch.Tensor) and qr_scale is None
        return kernels.wq_b(self.attn, self.linears, qr)

    def _index_wq_b(
        self,
        indexer: DeepseekV4Indexer,
        qr: torch.Tensor | QuantizedActivation,
        qr_scale: torch.Tensor | None,
        step: "DecodeStep",
    ) -> torch.Tensor:
        assert isinstance(qr, torch.Tensor) and qr_scale is None
        return kernels.index_wq_b(indexer, self.linears, qr)

    def _o_proj(
        self, out: torch.Tensor, positions: torch.Tensor, step: "DecodeStep"
    ) -> torch.Tensor:
        return kernels.o_proj(self.attn, self.linears, out, positions, self.all_reduce)


class HopperMoEStep(MoEStep):
    """``MoEStep`` on the Hopper MoE (``kernels.moe``)."""

    def __init__(self, ffn: "DeepseekV4MoE", max_rows: int) -> None:
        super().__init__(ffn, None, max_rows)
        self.shared = kernels.SharedMlp.of(ffn)
        self.routed = HopperLowBatchMoE.of(ffn)
        # The low-batch kernel leaves the top-k reduction to the boundary.
        self.deferrable = self.routed is not None

    def run(
        self, x: torch.Tensor, input_ids: torch.Tensor | None, step: "DecodeStep"
    ) -> torch.Tensor | MoEOutput:
        return kernels.moe(self.ffn, self.shared, self.routed, x, input_ids)


class HopperBoundaries(Boundaries):
    """``Boundaries`` whose residual-only boundaries (before an Engram layer and
    after the last layer) also reduce a finalized sublayer output in the fused
    kernel. No other TP collective then runs on the forward stream while the
    Engram WKV issues its all-gathers on its side stream.

    Steps above the fused kernel's capacity take the stock decode's unfused
    boundary: the TP all-reduce and the mHC post, after which the mHC pre
    computes the layer input itself."""

    def post_only(
        self, state: BoundaryState, fuse_partial: bool = False
    ) -> torch.Tensor:
        residual, _ = self._reduce(state, None, None, None)
        return residual

    def _reduce(
        self,
        state: BoundaryState,
        pre: torch.Tensor | None,
        norm: RMSNorm | None,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if state.residual.shape[0] <= self.all_reduce.capacity:
            return super()._reduce(state, pre, norm, mxfp8_out)
        assert mxfp8_out is None and isinstance(state.pending, torch.Tensor)
        x = get_tp_group().all_reduce(state.pending)
        return mhc_post_tilelang(x, state.residual, state.post, state.comb), None
