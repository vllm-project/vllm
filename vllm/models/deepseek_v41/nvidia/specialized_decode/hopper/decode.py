# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor on Hopper (SM90).

``HopperSpecializedDecode`` is ``SpecializedDecode`` (the same plan, layer
loop, boundaries, side streams and final collapse) over the Hopper stock
layers:

- The executor runs FULL-graph decode steps of up to ``max_rows`` rows (192,
  32 DSpark requests); larger steps run the stock layers.
- Every sublayer boundary runs the fused all-reduce + mHC kernel, the next
  coefficients on the mHC side stream, up to the stock decode's cap
  (``mhc_overlap_max_tokens()``, 144 rows); the residual-only boundaries too.
  Above that a boundary is the stock decode's: the TP all-reduce, the mHC
  post and pre (``HopperBoundaries``).
- ``DeepseekV4FlashMLAAttention`` layers run ``HopperAttentionStep``, whose
  sparse MLA shares one FlashMLA plan per layer type and one global top-k
  table per index source (``sparse_mla``), or up to ``mla_small.MAX_ROWS``
  rows runs the small-batch kernels (``mla_small``).
- The dense FP8 linears (``fused_wqa_wkv``, ``wq_b``, the indexer's ``wq_b``,
  ``wo_a`` on an FP8 copy, ``wo_b``, the shared expert's and the Engram
  ``wkv``) run on the low-batch CuTe DSL or tuned Humming calls up to
  ``dense.MAX_ROWS`` rows (``dense``, ``kernels``); ``wo_b`` publishes into
  the fused all-reduce where its kernel can.
- The MoE runs the low-batch kernel (``moe``) up to ``moe.MAX_ROWS`` rows:
  routed experts, and the shared expert up to ``moe.SHARED_MAX_ROWS``, in one
  launch whose top-k reduction the next boundary's fused all-reduce finalizes.
- The Engram WKV runs ahead on its side stream.
- The model state records the FlashMLA attention metadata builds of the FULL
  decode graphs up to ``max_rows``.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.humming import HummingMxfp8LinearKernel
from vllm.models.deepseek_v41.compressor import CompressorMetadataBuilder
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

from ...flashmla import (
    DeepseekSparseSWAFlashMLAMetadataBuilder,
    DeepseekV4FlashMLAAttention,
)
from ...ops.mhc import mhc_overlap_max_tokens
from .. import DecodeConfig, max_rows
from ..boundary.boundary import Boundaries
from ..engram.prefetch import EngramPrefetch
from ..runtime.decode import (
    ENGRAM_PREFETCH_MAX_ROWS,
    DecodeStep,
    LayerPlan,
    SpecializedDecode,
)
from . import attn_chain, fused_indexer, kernels
from .dense import DISPATCH, MAX_ROWS, DenseFp8Linear
from .sparse_mla import FlashMlaDecode, FlashMlaTables
from .steps import HopperAttentionStep, HopperBoundaries, HopperMoEStep

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.common.engram import Engram

    from ...dspark import DSparkDeepseekV4Model
    from ...model import DeepseekV4DecoderLayer, DeepseekV4Model
    from ..runtime.draft import DraftDecode
    from .draft import HopperDraftDecode

logger = init_logger(__name__)

# The FlashMLA path's builders, capture-safe as the GB300 executor's
# (``runtime.metadata_graphs``): the compressor, the FlashMLA SWA and
# compressed-cache builders, and the FP8 indexer's flattened decode build.
RECORDABLE_BUILDERS = frozenset(
    {
        CompressorMetadataBuilder,
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV4FlashMLAMetadataBuilder,
        DeepseekV32IndexerMetadataBuilder,
    }
)


class HopperSpecializedDecode(SpecializedDecode):
    """The decode executor over one backbone's local layers on Hopper."""

    recordable_builders = RECORDABLE_BUILDERS

    def __init__(
        self, backbone: "DeepseekV4Model", config: DecodeConfig, lm_head: nn.Module
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.lm_head = lm_head
        self.max_rows = self.recorded_max_rows = max_rows()
        layers = [
            cast("DeepseekV4DecoderLayer", backbone.layers[index])
            for index in range(backbone.start_layer, backbone.end_layer)
        ]
        attns: list[DeepseekV4FlashMLAAttention] = []
        for layer in layers:
            if type(layer.attn) is not DeepseekV4FlashMLAAttention:
                raise ValueError(
                    "decode_backend=latency_specialized on Hopper needs the "
                    f"FlashMLA sparse attention, not {type(layer.attn).__name__}"
                )
            attns.append(layer.attn)
        self.mhc_stream = layers[0].mhc_stream
        if self.mhc_stream is None or not backbone.fuse_mhc_all_reduce:
            raise ValueError(
                "decode_backend=latency_specialized needs the fused all-reduce + mHC"
            )

        from ..boundary.all_reduce import BoundaryAllReduce

        # Collective: every TP rank builds the executor in lockstep.
        self.all_reduce = BoundaryAllReduce(
            backbone.config,
            quantize_layer_input=False,
            max_num_tokens=mhc_overlap_max_tokens(),
        )
        self.boundaries: Boundaries = HopperBoundaries(self.all_reduce, self.mhc_stream)
        self.publisher = None
        ops = [
            attn.indexer.indexer_op
            for attn in attns
            if attn.indexer is not None
            and fused_indexer.applies(attn.indexer.indexer_op)
        ]
        self.fused_indexer = (
            fused_indexer.FusedDecodeIndexer.for_ops(ops, attns[0].attn_sink.device)
            if ops
            else None
        )
        self._candidate_k = max(
            (
                op.candidate_blocks.shape[1]
                for op in ops
                if op.candidate_write and op.candidate_blocks is not None
            ),
            default=0,
        )
        self.tables = tables = FlashMlaTables(attns)
        self.small_mla = tables.small
        self.mla = mla = FlashMlaDecode(tables)
        self.candidates = None
        engrams = [layer.engram for layer in layers if layer.engram is not None]
        self.engram_wkv: dict[Engram, kernels.DenseProvider] = {
            engram: linear
            for engram in engrams
            if (linear := DenseFp8Linear.of("engram_wkv", engram.wkv)) is not None
        }
        self.engram = EngramPrefetch(
            engrams,
            ENGRAM_PREFETCH_MAX_ROWS,
            # Humming keeps its split-K locks per layer.
            (HummingMxfp8LinearKernel,),
            kernels.engram_wkv(self.engram_wkv),
        )
        self.plans = [
            LayerPlan(
                layer_id=layer_id,
                layer=layer,
                attention=HopperAttentionStep(
                    attn, tables, mla, self.all_reduce, self.fused_indexer
                ),
                moe=HopperMoEStep(layer.ffn, self.max_rows),
                engram=layer.engram,
            )
            for layer_id, layer, attn in zip(
                range(backbone.start_layer, backbone.end_layer), layers, attns
            )
        ]
        self._swa_prefixes = sorted({attn.swa_cache_layer.prefix for attn in attns})

    def for_draft(self, backbone: "DSparkDeepseekV4Model") -> "HopperDraftDecode":
        """The DSpark draft model's executor, on this executor's fused
        all-reduce + mHC kernel, mailbox and mHC side stream."""
        from .draft import HopperDraftDecode

        return HopperDraftDecode(self, backbone)

    def gates(self, rows: int) -> DecodeStep:
        return DecodeStep(
            rows=rows,
            # The stock cap was measured with the coefficients on the side
            # stream at every row count.
            side_stream=True,
            # The GB300 low-batch kernels; ``kernels`` chooses by rows.
            low_batch=False,
            overlap_inputs=rows <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
            # The FP8 indexer's candidate consumers mask dense logits, with no
            # sparse schedule to build ahead.
            overlap_candidates=False,
            engram_prefetch=rows <= ENGRAM_PREFETCH_MAX_ROWS,
            executor=self,
        )

    def warm_up(
        self, vllm_config: VllmConfig, drafts: "Sequence[DraftDecode]" = ()
    ) -> None:
        """Compile the CuTe DSL kernels and JIT-compile the tuned Humming ones
        for the rows of every FULL decode graph the executor and the DSpark
        draft's (``drafts``) serve, one linear per seam and shape (the kernels
        do not depend on the weights). The stock layers register their mHC
        pre-norm kernels up to the executor's rows, and the fused all-reduce
        compiles when built."""
        sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()

        def graph_rows(query_len: int) -> list[int]:
            cap = min(self.max_rows, MAX_ROWS)
            return sorted(
                {round_up(n, query_len) for n in sizes} & set(range(1, cap + 1))
            )

        rows = graph_rows(vllm_config.num_speculative_tokens + 1)
        executors: list[tuple[list, list[int]]] = [(self.plans, rows)]
        spec = vllm_config.speculative_config
        if drafts and spec is not None:
            from_anchor = getattr(
                spec.draft_model_config.hf_config, "sample_from_anchor", True
            )
            draft_rows = graph_rows(
                spec.num_speculative_tokens + (0 if from_anchor else 1)
            )
            executors += [(draft.plans, draft_rows) for draft in drafts]
        by_shape: dict[tuple, tuple[kernels.DenseProvider, set[int]]] = {}

        def add(linear: kernels.DenseProvider, at: list[int]) -> None:
            key = (linear.seam, linear.n, linear.k)
            by_shape.setdefault(key, (linear, set()))[1].update(at)

        moe_layers = 0
        with torch.inference_mode():
            for plans, at in executors:
                for plan in plans:
                    attention = cast(HopperAttentionStep, plan.attention)
                    moe = cast(HopperMoEStep, plan.moe)
                    for linear in attention.linears.all():
                        add(linear, at)
                    if moe.shared is not None:
                        add(moe.shared.gate_up, at)
                        add(moe.shared.down, at)
                    if moe.routed is not None:
                        moe.routed.warm_up(at)
                        moe_layers += 1
            for linear in self.engram_wkv.values():
                add(linear, rows)
            for draft in drafts:
                draft = cast("HopperDraftDecode", draft)
                if draft.main_proj is not None:
                    add(draft.main_proj, list(DISPATCH["main_proj"]))
                if draft.small_mla is not None:
                    draft.small_mla.warm_up()
            if drafts:
                from ..moe.routed_expert_gemm import _finalize_kernel

                _finalize_kernel(torch.accelerator.current_device_index())
            if self.small_mla is not None:
                self.small_mla.warm_up()
            if self.fused_indexer is not None:
                self.fused_indexer.warm_up(
                    vllm_config.num_speculative_tokens + 1, self._candidate_k or 2048
                )
            chain_eps = {
                cast(HopperAttentionStep, plan.attention).attn.eps
                for plan in self.plans
                if cast(HopperAttentionStep, plan.attention).chain
            }
            for eps in chain_eps:
                attn_chain.warm_up(eps)
            for linear, shape_rows in by_shape.values():
                for m in sorted(shape_rows):
                    linear.warm_up(m)
                    if linear.seam == "wo_b":
                        linear.warm_up(m, publish=True)
        logger.info_once(
            "Hopper specialized decode: dense FP8 kernels ready for %s rows, "
            "low-batch MoE on %d of %d layers.",
            tuple(rows),
            moe_layers,
            sum(len(plans) for plans, _ in executors),
        )
