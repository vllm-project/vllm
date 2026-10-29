# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode of the DSpark draft model.

With DSpark speculative decoding, a few extra DeepSeek-V4.1 decoder layers
(``DSparkDeepseekV4Model``) draft the next tokens from the target's hidden
states after every target step, in CUDA graphs of their own. ``DraftDecode``
runs those layers for the FULL-graph pure decode steps, as ``SpecializedDecode``
runs the target's, through the same per-layer steps and on the kernels and
streams the target's executor owns: the fused all-reduce + mHC kernel and its
Lamport mailbox, the publishing ``wo_b`` GEMM and the mHC side stream. The
draft runs between target steps on the same stream, so sharing them needs no
synchronization beyond the stream order.

The draft's layers differ from the target's in ways the steps take as
parameters:

- the first layer takes ``[n, hc, H]`` streams, the embedding expanded to the
  hc copies, through its own mHC pre (``Boundaries.first``);
- the layers are built as stock layers, which reduce ``wo_b`` and the MoE
  themselves; ``AttentionStep`` projects ``wo_b`` without that all-reduce and
  ``MoEStep`` runs the runner's body, so both stay TP-partial for the next
  boundary's fused kernel;
- each layer finalizes its MoE (the fused kernel's finalize is built for the
  target's top-k): up to ``LOW_BATCH_MAX_ROWS`` rows on the low-batch GEMMs,
  without the target's fused front, and the finalize kernel;
- the last boundary reduces that plain tensor in the fused kernel's
  residual-only variant, and the streams are collapsed with the last pre-mix,
  before the final norm, which ``compute_logits`` applies.

The layers are sliding-window only: they share each SWA KV-cache group's
sparse-index table (``SparseIndexTables``), and the sparse MLA keeps the stock
decode's wide non-causal window of the draft tokens.

The target's model state builds this executor after the target's, on every TP
rank in the same order, and warms its GEMMs up with the target's. Every
forward that ``plan`` refuses runs the stock layers.
"""

from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.models.deepseek_v41.common.ops.query_quant import can_fuse_query_quant

from .. import FUSED_MHC_MAX_ROWS, LOW_BATCH_MAX_ROWS
from ..attention.sparse_mla import SparseIndexTables, SparseMlaDecode
from ..attention.step import AttentionStep
from ..boundary.boundary import BoundaryState
from ..moe.routed_expert_gemm import build_routed_expert_gemms
from ..moe.step import MoEStep
from .decode import (
    DecodeStep,
    SpecializedDecode,
    captures_pure_decode,
    flashinfer_attentions,
    step_gates,
)

if TYPE_CHECKING:
    from ...dspark import DSparkDeepseekV4Model
    from ...model import DeepseekV4DecoderLayer

logger = init_logger(__name__)


def check_low_batch_graph(vllm_config: VllmConfig) -> None:
    """Log when no DSpark draft CUDA graph is small enough for the low-batch
    GEMMs: the drafter rounds each capture size up to whole requests, so a
    one-request graph needs a capture size of one request's draft rows."""
    spec = vllm_config.speculative_config
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes
    if spec is None or not sizes:
        return
    sample_from_anchor = getattr(
        spec.draft_model_config.hf_config, "sample_from_anchor", True
    )
    rows = spec.num_speculative_tokens + (0 if sample_from_anchor else 1)
    smallest = min(-(-size // rows) * rows for size in sizes)
    if smallest > LOW_BATCH_MAX_ROWS:
        logger.info_once(
            "The smallest DSpark draft CUDA graph has %d rows, so the draft "
            "layers never take the low-batch GEMMs; add %d to "
            "cudagraph_capture_sizes for a one-request draft graph.",
            smallest,
            rows,
        )


class DraftLayerPlan(NamedTuple):
    layer: "DeepseekV4DecoderLayer"
    attention: AttentionStep
    moe: MoEStep


class DraftDecode:
    """The decode executor over the DSpark draft model's layers."""

    def __init__(
        self, target: SpecializedDecode, backbone: "DSparkDeepseekV4Model"
    ) -> None:
        if backbone.use_sequence_parallel or backbone.use_native_mega_moe:
            raise ValueError(
                "decode_backend=latency_specialized runs the DSpark draft without "
                "sequence parallelism or the native mega MoE"
            )
        self.backbone = backbone
        layers = [cast("DeepseekV4DecoderLayer", layer) for layer in backbone.layers]
        attns = flashinfer_attentions(layers)
        self.mhc_stream = target.mhc_stream
        self.all_reduce = target.all_reduce
        self.boundaries = target.boundaries
        self.tables = SparseIndexTables(attns)
        # The persistent KV counters are the target's: the draft's decode
        # launches run after the target's on the same stream.
        self.mla = SparseMlaDecode(
            self.tables, FUSED_MHC_MAX_ROWS * max(attn.padded_heads for attn in attns)
        )
        gemms = build_routed_expert_gemms(
            [layer.ffn.experts for layer in layers], fused_front=False
        )
        quantize = self.all_reduce.quantize_layer_input
        self.plans = [
            DraftLayerPlan(
                layer=layer,
                attention=AttentionStep(
                    attn,
                    self.tables,
                    self.mla,
                    target.publisher,
                    mxfp8_input=quantize and can_fuse_query_quant([attn.fused_wqa_wkv]),
                    source=None,
                    consumer=None,
                ),
                moe=MoEStep(layer.ffn, gemm, FUSED_MHC_MAX_ROWS, finalize=True),
            )
            for layer, attn, gemm in zip(layers, attns, gemms)
        ]
        self._swa_prefixes = sorted({attn.swa_cache_layer.prefix for attn in attns})

    def plan(self, positions: torch.Tensor) -> DecodeStep | None:
        """The step this forward runs, or None to run the stock layers: as for
        the target, only FULL CUDA graph captures of pure decode steps of up to
        ``FUSED_MHC_MAX_ROWS`` rows run the executor."""
        rows = positions.shape[0]
        if (
            not 0 < rows <= FUSED_MHC_MAX_ROWS
            or not is_forward_context_available()
            or not torch.cuda.is_current_stream_capturing()
            or not captures_pure_decode(self._swa_prefixes)
        ):
            return None
        return step_gates(rows, self)

    def run(
        self,
        step: DecodeStep,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The draft layers and the collapse of the step on the hc streams
        ``x`` ([n, hc, H]); returns the pre-norm hidden state, as
        ``DSparkDeepseekV4Model.forward`` does. The draft has no Engram
        layers, so ``engram_hashes`` and ``engram_mask`` stay None."""
        assert engram_hashes is None and engram_mask is None
        bound = self.boundaries
        self.tables.reset()
        self.all_reduce.reset()
        state: BoundaryState | None = None
        try:
            for plan in self.plans:
                quant = None
                if state is None:
                    state, layer_input = bound.first(plan.layer, x, step)
                else:
                    state, layer_input, quant, _ = bound.sublayer(
                        plan.layer,
                        False,
                        state,
                        step,
                        quantize=plan.attention.mxfp8_input,
                    )
                state.pending = plan.attention.run(layer_input, quant, positions, step)
                bound.join(step)
                state, ffn_input, _, _ = bound.sublayer(plan.layer, True, state, step)
                state.pending = plan.moe.run(ffn_input, input_ids, step)
                bound.join(step)
            assert state is not None
            # No successor boundary: reduce the last layer's output into the
            # streams only.
            streams = bound.post_only(state, fuse_partial=True)
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            self.tables.join()
            torch.cuda.current_stream().wait_stream(self.mhc_stream)
            raise
        self.tables.join()
        return hc_collapse_triton(streams, state.pre)
