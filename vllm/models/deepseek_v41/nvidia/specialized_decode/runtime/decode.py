# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor.

``SpecializedDecode`` runs the decoder layers of ``DeepseekV4Model`` for the
steps captured into FULL CUDA graphs of pure decode batches of up to
``FUSED_MHC_MAX_ROWS`` rows. Its model state builds it once the weights are
loaded. ``plan`` decides per forward whether it runs; the model's forward then
passes the returned ``DecodeStep`` to the backbone, which runs its embedding
and Engram hashing as usual and hands the layers to ``DecodeStep.run``. Every
other forward (prefill, mixed, eager, PIECEWISE) runs the stock layers, which
the executor leaves unchanged.

Per layer the executor runs the stock modules through its own steps:

- ``Boundaries`` (``boundary``): the mHC between sublayers, on the fused
  all-reduce + mHC kernel;
- ``AttentionStep`` (``attention.step``);
- ``MoEStep`` (``moe.step``), which leaves the MoE finalize to the next
  boundary;
- ``EngramPrefetch`` (``engram.prefetch``) for the Engram layers.

The steps issue TP collectives, so every decision reads host state that all TP
ranks share: the step's rows and the attention metadata's host fields. Side
streams the steps fork are joined before ``run`` returns, so a capture ends
joined.
"""

from typing import TYPE_CHECKING, NamedTuple, TypeAlias, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
from vllm.models.deepseek_v41.common.engram import Engram
from vllm.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant
from vllm.sequence import IntermediateTensors

from ...flashinfer_sparse import DeepseekV4FlashInferMLAAttention
from .. import (
    FUSED_MHC_MAX_ROWS,
    LOW_BATCH_MAX_ROWS,
    MHC_OVERLAP_MAX_ROWS,
    DecodeConfig,
    uses_checked_flashinfer,
)
from ..attention.indexer import OVERLAP_MAX_ROWS, CandidateOverlap
from ..attention.sparse_mla import SparseIndexTables, SparseMlaDecode
from ..attention.step import AttentionStep
from ..boundary.boundary import Boundaries, BoundaryState
from ..engram.prefetch import EngramPrefetch
from ..moe.routed_expert_gemm import build_routed_expert_gemms
from ..moe.step import MoEStep

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    from ...model import DeepseekV4DecoderLayer, DeepseekV4Model

# Most rows of a step whose Engram WKV runs ahead of the layers.
ENGRAM_PREFETCH_MAX_ROWS = 16

DecodeOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


class DecodeStep(NamedTuple):
    """The host-side gates of one captured decode step, by its rows."""

    rows: int
    # The next mHC coefficients and aux means run on the mHC side stream;
    # otherwise inline after the fused all-reduce + mHC.
    side_stream: bool
    # The low-batch GEMMs: routed experts, MXFP8 projections, ll_bf16 side
    # projections, publishing wo_b.
    low_batch: bool
    # The attention side projections run on aux streams.
    overlap_inputs: bool
    # The candidate source forks the consumers' candidate expansion.
    overlap_candidates: bool
    # The Engram WKV runs ahead of the layers.
    engram_prefetch: bool
    executor: "SpecializedDecode"

    def run(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> DecodeOutput:
        """The backbone's output after its embedding."""
        return self.executor.run(
            self, hidden_states, positions, input_ids, engram_hashes, engram_mask
        )


def step_gates(rows: int, executor: "SpecializedDecode") -> DecodeStep:
    """The step of ``rows`` rows (0 < rows <= ``FUSED_MHC_MAX_ROWS``)."""
    return DecodeStep(
        rows=rows,
        side_stream=rows <= MHC_OVERLAP_MAX_ROWS,
        low_batch=rows <= LOW_BATCH_MAX_ROWS,
        overlap_inputs=rows <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        overlap_candidates=rows <= OVERLAP_MAX_ROWS,
        engram_prefetch=rows <= ENGRAM_PREFETCH_MAX_ROWS,
        executor=executor,
    )


def captures_pure_decode(swa_prefixes: list[str]) -> bool:
    """Whether the forward being captured into a CUDA graph is a FULL graph of
    a pure decode step: not a PIECEWISE capture (Model Runner V2 captures FULL
    graphs with the forward context's mode NONE, and PIECEWISE ones, which
    report capturing too, with PIECEWISE), and decode metadata without
    prefills for every SWA KV-cache group (``swa_prefixes``) of the layers."""
    context = get_forward_context()
    attn_metadata = context.attn_metadata
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE or not isinstance(
        attn_metadata, dict
    ):
        return False
    for prefix in swa_prefixes:
        metadata = cast("DeepseekSparseSWAMetadata | None", attn_metadata.get(prefix))
        if metadata is None or metadata.num_prefills > 0:
            return False
    return True


class LayerPlan(NamedTuple):
    layer_id: int
    layer: "DeepseekV4DecoderLayer"
    attention: AttentionStep
    moe: MoEStep
    engram: Engram | None


class SpecializedDecode:
    """The decode executor over one backbone's local layers."""

    def __init__(
        self, backbone: "DeepseekV4Model", config: DecodeConfig, lm_head: nn.Module
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.lm_head = lm_head
        layers = [
            cast("DeepseekV4DecoderLayer", backbone.layers[index])
            for index in range(backbone.start_layer, backbone.end_layer)
        ]
        attns: list[DeepseekV4FlashInferMLAAttention] = []
        for layer in layers:
            if not isinstance(layer.attn, DeepseekV4FlashInferMLAAttention):
                raise ValueError(
                    "decode_backend=latency_specialized needs the FlashInfer "
                    f"sparse MLA attention, not {type(layer.attn).__name__}"
                )
            attns.append(layer.attn)
        self.mhc_stream = layers[0].mhc_stream
        if self.mhc_stream is None or not backbone.fuse_mhc_all_reduce:
            raise ValueError(
                "decode_backend=latency_specialized needs the fused all-reduce + mHC"
            )

        # The CuTe DSL kernels, imported only when the executor is built.
        from ..boundary.all_reduce import BoundaryAllReduce
        from ..boundary.publishing_gemm import PublishingMxfp8Linear

        # Collective: every TP rank builds the executor in lockstep.
        self.all_reduce = BoundaryAllReduce(
            backbone.config,
            uses_checked_flashinfer("the fused kernel's MXFP8 layer input"),
        )
        self.boundaries = Boundaries(self.all_reduce)
        self.publisher = (
            PublishingMxfp8Linear(self.all_reduce)
            if uses_checked_flashinfer("the publishing wo_b GEMM")
            else None
        )
        self.tables = SparseIndexTables(attns)
        # A decode launch has at most one row per token and padded head.
        self.mla = SparseMlaDecode(
            self.tables, FUSED_MHC_MAX_ROWS * max(attn.padded_heads for attn in attns)
        )
        source = next(
            (
                attn
                for attn in attns
                if attn.indexer is not None
                and attn.layer_id == attn.candidate_source_layer
            ),
            None,
        )
        consumers = [
            attn
            for attn in attns
            if source is not None
            and attn is not source
            and attn.indexer is not None
            and attn.candidate_source_layer == source.layer_id
            and isinstance(attn.indexer.indexer_op, SparseMQAIndexer)
        ]
        self.candidates: CandidateOverlap | None = None
        if consumers:
            indexer = consumers[0].indexer
            assert indexer is not None
            assert isinstance(indexer.indexer_op, SparseMQAIndexer)
            self.candidates = CandidateOverlap(indexer.indexer_op)
        self.engram = EngramPrefetch(
            [layer.engram for layer in layers if layer.engram is not None],
            ENGRAM_PREFETCH_MAX_ROWS,
        )
        gemms = build_routed_expert_gemms([layer.ffn.experts for layer in layers])
        quantize = self.all_reduce.quantize_layer_input
        self.plans = [
            LayerPlan(
                layer_id=layer_id,
                layer=layer,
                attention=AttentionStep(
                    attn,
                    self.tables,
                    self.mla,
                    self.publisher,
                    mxfp8_input=quantize and can_fuse_query_quant([attn.fused_wqa_wkv]),
                    source=self.candidates if attn is source else None,
                    consumer=self.candidates if attn in consumers else None,
                ),
                moe=MoEStep(layer.ffn, gemm, FUSED_MHC_MAX_ROWS),
                engram=layer.engram,
            )
            for layer_id, layer, attn, gemm in zip(
                range(backbone.start_layer, backbone.end_layer), layers, attns, gemms
            )
        ]
        self._swa_prefixes = sorted({attn.swa_cache_layer.prefix for attn in attns})

    @classmethod
    def for_target(
        cls, backbone: "DeepseekV4Model", config: DecodeConfig, lm_head: nn.Module
    ) -> "SpecializedDecode":
        """The executor of the target model's backbone."""
        return cls(backbone, config, lm_head)

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers: only
        FULL CUDA graph captures of pure decode steps of up to
        ``FUSED_MHC_MAX_ROWS`` rows run the executor."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= FUSED_MHC_MAX_ROWS
            or not is_forward_context_available()
        ):
            return None
        if not torch.cuda.is_current_stream_capturing():
            # The eager run before each capture.
            if self.candidates is not None:
                self.candidates.warm()
            return None
        if not captures_pure_decode(self._swa_prefixes):
            return None
        return step_gates(rows, self)

    def run(
        self,
        step: DecodeStep,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
    ) -> DecodeOutput:
        """The decoder layers and the final collapse of the step, on the
        embedding ``x``; returns what ``DeepseekV4Model.forward`` returns."""
        backbone = self.backbone
        aux_layers = backbone.aux_hidden_state_layers
        aux: dict[int, torch.Tensor] = {}
        bound = self.boundaries
        self.tables.reset()
        self.all_reduce.reset()
        try:
            if engram_hashes is not None and step.engram_prefetch:
                self.engram.start(x.shape[0])
            state: BoundaryState | None = None
            for plan in self.plans:
                layer = plan.layer
                quant = None
                if state is None:
                    state, layer_input = bound.first(layer, x, step)
                elif plan.engram is not None and engram_hashes is not None:
                    streams = bound.post_only(state)
                    if plan.layer_id in aux_layers:
                        aux[plan.layer_id] = streams.mean(dim=1)
                    injected = self.engram.forward(
                        plan.engram,
                        streams,
                        engram_hashes[:, plan.engram.layer_hash_index],
                        engram_mask,
                    )
                    state, layer_input = bound.pre(layer, injected, state, step)
                else:
                    state, layer_input, quant, layer_aux = bound.sublayer(
                        layer,
                        False,
                        state,
                        step,
                        quantize=plan.attention.mxfp8_input,
                        capture_aux=plan.layer_id in aux_layers,
                    )
                    if layer_aux is not None:
                        aux[plan.layer_id] = layer_aux
                state.pending = plan.attention.run(layer_input, quant, positions, step)
                bound.join(layer, step)
                state, ffn_input, _, _ = bound.sublayer(layer, True, state, step)
                state.pending = plan.moe.run(ffn_input, input_ids, step)
                bound.join(layer, step)
            assert state is not None
            streams = bound.post_only(state)
            if backbone.end_layer in aux_layers:
                aux[backbone.end_layer] = streams.mean(dim=1)
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            self._join_side_work(error=True)
            raise
        self._join_side_work()
        return self._head(streams, state.pre, aux)

    def _join_side_work(self, error: bool = False) -> None:
        """Make the current stream wait for the side streams' pending work."""
        self.tables.join()
        if self.candidates is not None:
            self.candidates.join()
        self.engram.join()
        if error:
            assert self.mhc_stream is not None
            torch.cuda.current_stream().wait_stream(self.mhc_stream)

    def _head(
        self, streams: torch.Tensor, pre: torch.Tensor, aux: dict[int, torch.Tensor]
    ) -> DecodeOutput:
        """``DeepseekV4Model.forward`` after the layers: the MTP buffer, the
        collapse of the hc streams with the last pre-mix and the final norm."""
        backbone = self.backbone
        if backbone._mtp_hidden_buffer is not None:
            backbone._mtp_hidden_buffer[: streams.shape[0]].copy_(streams.flatten(1))
        hidden_states = backbone.norm(hc_collapse_triton(streams, pre))
        aux_hidden_states = [
            aux[layer_id]
            for layer_id in backbone.aux_hidden_state_layers
            if layer_id in aux
        ]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
