# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor on Hopper (SM90).

`HopperSpecializedDecode` is `SpecializedDecode` (the same `DecodeStep`
and final collapse) with its own plan, layer loop and steps over the Hopper
stock layers:

- It runs FULL-graph pure-decode steps of up to `MAX_ROWS` rows (192, 32
  DSpark requests); larger steps run the stock layers.
- Every sublayer boundary runs the fused all-reduce + mHC kernel up to
  `FUSED_MAX_ROWS` rows (144), and above that the TP all-reduce and mHC
  post; the next coefficients come from the executor's mHC pre (`mhc`) on
  its own side stream (`HopperBoundaries`). No DeepGEMM kernel runs.
- `DeepseekV4FlashMLAAttention` layers run `HopperAttentionStep`, whose
  sparse MLA shares one FlashMLA plan per layer type and one global top-k
  table per index source (`sparse_mla`), or up to `mla_small.MAX_ROWS`
  uniform rows runs the small-batch chain (`mla_small`, `attn_chain`).
- The dense FP8 linears (`fused_wqa_wkv`, `wq_b`, the indexer's `wq_b`,
  `wo_b` and the shared expert's, and on the chain `wo_a` on an FP8 copy)
  run on the low-batch CuTe DSL or tuned Humming calls up to
  `dense.MAX_ROWS` rows (`dense`, `kernels`).
- The MoE runs the low-batch kernel (`moe`) up to `moe.MAX_ROWS` rows,
  whose top-k reduction the next boundary's fused all-reduce finalizes.
"""

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.forward_context import is_forward_context_available
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up

from ...flashmla import DeepseekV4FlashMLAAttention
from .. import DecodeConfig
from ..boundary.boundary import BoundaryState
from ..runtime.decode import (
    DecodeOutput,
    DecodeStep,
    SpecializedDecode,
    captures_pure_decode,
)
from . import attn_chain
from .dense import MAX_ROWS as DENSE_MAX_ROWS
from .dense import DenseFp8Linear
from .mhc import SeamPre
from .sparse_mla import FlashMlaDecode, FlashMlaTables
from .steps import HopperAttentionStep, HopperBoundaries, HopperMoEStep

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.common.engram import Engram

    from ...model import DeepseekV4DecoderLayer, DeepseekV4Model

logger = init_logger(__name__)

# Rows of the executor's largest step: 32 DSpark requests of 6 rows.
MAX_ROWS = 192
# Rows of the largest step on the fused all-reduce + mHC: on H200 at TP4 its
# one-shot all-reduce wins through 144 rows and falls behind the two-shot one
# at 192.
FUSED_MAX_ROWS = 144


class LayerPlan(NamedTuple):
    layer_id: int
    layer: "DeepseekV4DecoderLayer"
    attention: HopperAttentionStep
    moe: HopperMoEStep
    engram: "Engram | None"


class HopperSpecializedDecode(SpecializedDecode):
    """The decode executor over one backbone's local layers on Hopper."""

    boundaries: HopperBoundaries
    mhc_stream: torch.cuda.Stream

    def __init__(
        self, backbone: "DeepseekV4Model", config: DecodeConfig, lm_head: nn.Module
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.lm_head = lm_head
        self.max_rows = MAX_ROWS
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
        if not all(SeamPre.applies(layer) for layer in layers):
            raise ValueError(
                "decode_backend=latency_specialized on Hopper needs 4 hc streams "
                "of 5120 with FP32 mHC projections"
            )
        # The executor's own mHC side stream: the model's runs only with the
        # DeepGEMM prenorm GEMM.
        self.mhc_stream = torch.cuda.Stream()

        from ..boundary.all_reduce import BoundaryAllReduce

        # Collective: every TP rank builds the executor in lockstep. The
        # low-batch MoE kernel, the grid right before each finalize, writes
        # the route map and weights the finalize reads.
        self.all_reduce = BoundaryAllReduce(
            backbone.config,
            quantize_layer_input=False,
            max_num_tokens=FUSED_MAX_ROWS,
            finalize_metadata_before_pdl=False,
        )
        self.boundaries = HopperBoundaries(self.all_reduce, self.mhc_stream)
        for layer in layers:
            self.boundaries.add(layer)
        # The stock embedding and logits gather.
        self.embedding = None
        self.all_gather = None
        self.candidates = None
        self.mla_tables = tables = FlashMlaTables(attns)
        self.small_mla = tables.small
        mla = FlashMlaDecode(tables)
        self.layer_plans = [
            LayerPlan(
                layer_id=layer_id,
                layer=layer,
                attention=HopperAttentionStep(attn, mla),
                moe=HopperMoEStep(layer.ffn, self.max_rows),
                engram=layer.engram,
            )
            for layer_id, layer, attn in zip(
                range(backbone.start_layer, backbone.end_layer), layers, attns
            )
        ]
        self._swa_prefixes = sorted({attn.swa_cache_layer.prefix for attn in attns})

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
    ) -> DecodeStep | None:
        """As `SpecializedDecode.plan`, up to `max_rows` rows."""
        rows = positions.shape[0]
        if (
            intermediate_tensors is not None
            or not 0 < rows <= self.max_rows
            or not is_forward_context_available()
            or not torch.cuda.is_current_stream_capturing()
            or not captures_pure_decode(self._swa_prefixes)
        ):
            return None
        # The GB300 low-batch kernels never run; ``kernels`` chooses by rows.
        return DecodeStep(
            rows=rows,
            low_batch=False,
            overlap_inputs=rows <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
            executor=self,
        )

    def run(
        self,
        step: DecodeStep,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
        streams: torch.Tensor | None = None,
    ) -> DecodeOutput:
        """`SpecializedDecode.run` on the Hopper steps and boundaries."""
        backbone = self.backbone
        aux_layers = backbone.aux_hidden_state_layers
        aux: dict[int, torch.Tensor] = {}
        bound = self.boundaries
        self.mla_tables.reset()
        self.all_reduce.reset()
        try:
            state: BoundaryState | None = None
            for plan in self.layer_plans:
                layer = plan.layer
                if state is None:
                    state, layer_input = bound.first(layer, x, streams)
                elif plan.engram is not None and engram_hashes is not None:
                    residual = bound.post_only(state)
                    if plan.layer_id in aux_layers:
                        aux[plan.layer_id] = residual.mean(dim=1)
                    injected = plan.engram(
                        residual,
                        engram_hashes[:, plan.engram.layer_hash_index],
                        engram_mask,
                    )
                    state, layer_input = bound.pre(layer, injected, state)
                else:
                    state, layer_input, _, layer_aux = bound.sublayer(
                        layer, False, state, capture_aux=plan.layer_id in aux_layers
                    )
                    if layer_aux is not None:
                        aux[plan.layer_id] = layer_aux
                state.pending = plan.attention.run(layer_input, positions, step)
                bound.join(layer)
                state, ffn_input, _, _ = bound.sublayer(layer, True, state)
                state.pending = plan.moe.run(ffn_input, input_ids, step)
                bound.join(layer)
            assert state is not None
            residual = bound.post_only(state)
            if backbone.end_layer in aux_layers:
                aux[backbone.end_layer] = residual.mean(dim=1)
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            self._join_side_work(error=True)
            raise
        self._join_side_work()
        return self._head(residual, state.pre, aux)

    def _join_side_work(self, error: bool = False) -> None:
        self.mla_tables.join()
        if error:
            torch.cuda.current_stream().wait_stream(self.mhc_stream)

    def warm_up(self, vllm_config: VllmConfig) -> None:
        """Compile the CuTe DSL kernels and JIT-compile the tuned Humming ones
        for the rows of every FULL decode graph the executor serves, one linear
        per seam and shape (the kernels do not depend on the weights)."""
        sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
        query_len = vllm_config.num_speculative_tokens + 1
        cap = min(self.max_rows, DENSE_MAX_ROWS)
        rows = sorted({round_up(n, query_len) for n in sizes} & set(range(1, cap + 1)))
        by_shape: dict[tuple, tuple[DenseFp8Linear, set[int]]] = {}

        def add(linear: DenseFp8Linear) -> None:
            key = (linear.seam, linear.n, linear.k)
            by_shape.setdefault(key, (linear, set()))[1].update(rows)

        moe_layers = 0
        with torch.inference_mode():
            for plan in self.layer_plans:
                for linear in plan.attention.linears.all():
                    add(linear)
                moe = plan.moe
                if moe.shared is not None:
                    add(moe.shared.gate_up)
                    add(moe.shared.down)
                if moe.routed is not None:
                    moe.routed.warm_up(rows)
                    moe_layers += 1
            # The mHC pre's kernels are the same in every layer.
            self.boundaries.warm_up(self.layer_plans[0].layer, rows)
            if self.small_mla is not None:
                self.small_mla.warm_up()
            chained = [p.attention for p in self.layer_plans if p.attention.chain]
            for eps in {attention.attn.eps for attention in chained}:
                attn_chain.warm_up(eps)
            for linear, shape_rows in by_shape.values():
                for m in sorted(shape_rows):
                    linear.warm_up(m)
        logger.info_once(
            "Hopper specialized decode: kernels ready for %s rows, low-batch MoE on "
            "%d of %d layers.",
            tuple(rows),
            moe_layers,
            len(self.layer_plans),
        )
