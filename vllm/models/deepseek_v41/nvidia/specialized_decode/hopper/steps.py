# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The steps of a Hopper executor layer: attention, MoE and boundaries with
the Hopper kernel choices (``kernels``)."""

from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm

from ..attention.indexer import rope_quant_q
from ..boundary.boundary import Boundaries, BoundaryState
from ..moe.step import MoEStep
from . import kernels
from .all_reduce import HopperBoundaryAllReduce
from .mhc import SeamPre
from .moe import HopperLowBatchMoE
from .sparse_mla import FlashMlaDecode, FlashMlaTables

if TYPE_CHECKING:
    from ...flashmla import DeepseekV4FlashMLAAttention
    from ...model import DeepseekV4DecoderLayer, DeepseekV4MoE
    from ..runtime.decode import DecodeStep


class HopperAttentionStep:
    """One ``DeepseekV4FlashMLAAttention`` layer of the step, in the GB300
    executor's schedule (``attention.step.AttentionStep``): the side
    projections join their side streams at their first reader, the indexer
    query is built on aux stream 2 once ``qr`` is ready and the compressor
    and index key on aux stream 0; on the Hopper projections (``linears``).
    The layer input is BF16 (the dense linears are weight-only W8A16)."""

    def __init__(
        self,
        attn: "DeepseekV4FlashMLAAttention",
        tables: FlashMlaTables,
        mla: FlashMlaDecode,
    ) -> None:
        self.attn = attn
        self.tables = tables
        self.mla = mla
        self.linears = kernels.AttentionLinears.of(attn)
        # start, latent ready, index key done, index query done.
        self._events = [torch.cuda.Event() for _ in range(4)]

    def run(
        self, x: torch.Tensor, positions: torch.Tensor, step: "DecodeStep"
    ) -> torch.Tensor:
        """The layer's TP-partial output for input ``x``."""
        attn, mla = self.attn, self.mla
        qr_kv, kv_score, indexer_weights = self._input_projections(x, step)
        qr, qr_scale, kv = attn._split_qkv_and_norm(qr_kv)
        assert qr_scale is None
        q, index_q, index_q_scale, index_weights = self._prepare(
            qr, kv, kv_score, indexer_weights, positions
        )
        if attn.indexer is not None and index_q is not None:
            assert index_weights is not None
            q_quant = index_q if index_q_scale is None else (index_q, index_q_scale)
            attn.indexer.indexer_op(x, q_quant, None, index_weights)
        out = attn._alloc_attn_out(x.shape[0], x)
        assert isinstance(out, torch.Tensor)
        mla.run(attn, q, out)
        return kernels.o_proj(attn, self.linears, out, positions)

    def _input_projections(
        self, x: torch.Tensor, step: "DecodeStep"
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``fused_wqa_wkv``, and the side projections (``kv_score``, the
        indexer weights) left unjoined on aux streams 0 and 1; ``_prepare``
        joins each at its first reader."""
        attn = self.attn
        side: list[Any] = [None, None]
        if attn.compressor is not None:
            kv_weight = attn.compressor.fused_wkv_wgate.weight
            side[0] = lambda: torch.mm(x, kv_weight.T, out_dtype=torch.float32)
        if attn.indexer is not None:
            weights_proj = attn.indexer.weights_proj
            side[1] = lambda: weights_proj(x)[0]
        streams = attn.aux_stream_list if step.overlap_inputs else None
        start = attn.ln_events[0]
        start.record()
        results: list[torch.Tensor | None] = [None, None]
        for i, fn in enumerate(side):
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
        return kernels.wqa_wkv(attn, self.linears, x), results[0], results[1]

    def _prepare(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """The query (normalized, rotated, with the KV inserted) and the
        indexer query, its scale and weights, with the indexer inputs,
        compressor and index key on side streams."""
        attn = self.attn
        indexer, compressor = attn.indexer, attn.compressor
        streams = attn.aux_stream_list
        start, latent_ready, index_k_done, index_q_done = self._events
        start.record()
        index_q = index_q_scale = index_weights = None
        if indexer is not None:
            assert indexer_weights is not None and streams is not None
            with torch.cuda.stream(streams[2]):
                start.wait()
                index_q_pre = kernels.index_wq_b(indexer, self.linears, qr)
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
            assert kv_score is not None and streams is not None
            # After the kv_score GEMM, which ran on this stream.
            with torch.cuda.stream(streams[0]):
                start.wait()
                latent = compressor(kv_score, positions)
                latent_ready.record()
                if indexer is not None and indexer.owns_k:
                    indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
                index_k_done.record()
        q = kernels.wq_b(attn, self.linears, qr)
        q = q.view(-1, attn.n_local_heads, attn.head_dim)
        q = attn._fused_qnorm_rope_kv_insert(
            q, kv, positions, get_forward_context().attn_metadata
        )
        if compressor is not None:
            latent_ready.wait()
            compressor.insert_cache(latent, positions, attn.rotary_emb)
            index_k_done.wait()
        if indexer is not None:
            index_q_done.wait()
        return q, index_q, index_q_scale, index_weights


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
    """``Boundaries`` with the executor's own mHC pre (``mhc.SeamPre``) on its
    own side stream (``stream``): the fused all-reduce + mHC up to its
    capacity; above it the TP all-reduce and the mHC post, after which the mHC
    pre also computes the layer input. No DeepGEMM kernel runs."""

    def __init__(
        self, all_reduce: HopperBoundaryAllReduce, stream: torch.cuda.Stream
    ) -> None:
        super().__init__(all_reduce)
        self.all_reduce: HopperBoundaryAllReduce = all_reduce
        self.stream = stream
        self.seams: dict[tuple[DeepseekV4DecoderLayer, bool], SeamPre] = {}

    def add(self, layer: "DeepseekV4DecoderLayer") -> None:
        for ffn in (False, True):
            self.seams[layer, ffn] = SeamPre(layer, ffn)

    def warm_up(self, layer: "DeepseekV4DecoderLayer", rows: list[int]) -> None:
        for (owner, ffn), seam in self.seams.items():
            if owner is layer:
                seam.warm_up(rows, first=not ffn)

    def first(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        streams: torch.Tensor | None = None,
    ) -> tuple[BoundaryState, torch.Tensor]:
        residual = (
            x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            if streams is None
            else streams
        )
        return self._entry(layer, residual, None)

    def pre(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        state: BoundaryState,
    ) -> tuple[BoundaryState, torch.Tensor]:
        return self._entry(layer, residual, state.pre)

    def sublayer(
        self,
        layer: "DeepseekV4DecoderLayer",
        ffn: bool,
        state: BoundaryState,
        *,
        quantize: bool = False,
        capture_aux: bool = False,
    ) -> tuple[
        BoundaryState, torch.Tensor, QuantizedActivation | None, torch.Tensor | None
    ]:
        assert not quantize
        layer_input = None
        if state.residual.shape[0] <= self.all_reduce.capacity:
            norm = layer.ffn_norm if ffn else layer.attn_norm
            residual, layer_input = self._reduce(state, state.pre, norm, None)
            assert layer_input is not None
        else:
            residual, _ = self._reduce(state, None, None, None)
        aux = residual.mean(dim=1) if capture_aux else None
        post, comb, layer_input, next_pre = self.seams[layer, ffn](
            residual, state.pre, self.stream, layer_input
        )
        return BoundaryState(residual, post, comb, next_pre), layer_input, None, aux

    def join(self, layer: "DeepseekV4DecoderLayer") -> None:
        torch.cuda.current_stream().wait_stream(self.stream)

    def _entry(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        pre_mix: torch.Tensor | None,
    ) -> tuple[BoundaryState, torch.Tensor]:
        """An attention entry without a reduction."""
        post, comb, layer_input, next_pre = self.seams[layer, False](
            residual, pre_mix, self.stream
        )
        return BoundaryState(residual, post, comb, next_pre), layer_input

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
