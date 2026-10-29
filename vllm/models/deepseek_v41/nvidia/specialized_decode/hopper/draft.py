# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft model's specialized decode on Hopper.

``HopperDraftDecode`` is ``DraftDecode`` (the same layer loop, first-layer mHC
pre of the expanded embedding, residual-only last boundary and collapse) over
the draft's sliding-window layers with the Hopper kernel choices, on the
target executor's fused all-reduce + mHC kernel, mailbox and mHC side stream:

- attention: ``HopperAttentionStep`` (the dense FP8 projections, ``wo_b``
  publishing where its kernel can) on the draft's own FlashMLA tables, so the
  small-batch kernels take steps whose SWA window they hold;
- MoE (128 experts, top-3): the low-batch MoE kernel up to ``moe.MAX_ROWS``
  rows, finalized in the layer (the fused kernel's finalize is built for the
  target's top-k) with the routed-expert finalize kernel; above that the
  runner's body, TP-partial;
- ``main_proj``, which prepares the draft's context, on tuned Humming calls.

The draft's stock layers reduce ``wo_b`` and the MoE themselves; every call
here leaves them TP-partial for the next boundary.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import is_forward_context_available
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack

from ...flashmla import DeepseekV4FlashMLAAttention
from ..runtime.decode import DecodeStep, captures_pure_decode
from ..runtime.draft import DraftDecode, DraftLayerPlan
from . import kernels
from .dense import DenseFp8Linear
from .sparse_mla import FlashMlaDecode, FlashMlaTables
from .steps import HopperAttentionStep, HopperMoEStep

if TYPE_CHECKING:
    from ...dspark import DSparkDeepseekV4Model
    from ...model import DeepseekV4DecoderLayer, DeepseekV4MoE
    from .decode import HopperSpecializedDecode


class HopperDraftMoEStep(HopperMoEStep):
    """A draft MoE layer, finalized in the layer."""

    def __init__(self, ffn: "DeepseekV4MoE", max_rows: int) -> None:
        super().__init__(ffn, max_rows)
        if self.runner.routed_scaling_factor != 1.0:
            # The runner's body adds the shared and routed outputs without the
            # stock forward's routed scaling.
            raise ValueError(
                "decode_backend=latency_specialized requires the draft's MoE to "
                "have routed_scaling_factor == 1.0"
            )
        self.deferrable = False

    def run(
        self, x: torch.Tensor, input_ids: torch.Tensor | None, step: DecodeStep
    ) -> torch.Tensor:
        rows = x.shape[0]
        routed = self.routed
        if routed is not None and routed.serves(rows):
            out = kernels.moe(self.ffn, self.shared, routed, x, input_ids)
            assert isinstance(out, MoEOutput)
            assert isinstance(out.routed, UnfinalizedMoEOutput)
            assert out.shared_output is not None
            return out.shared_output + finalize(out.routed)
        with kernels.shared_installed(self.shared, rows):
            shared_output, finalized = _unpack(
                self.runner._forward_impl(x, x, x, input_ids)
            )
        assert shared_output is not None and isinstance(finalized, torch.Tensor)
        return shared_output + finalized


def finalize(routed: UnfinalizedMoEOutput) -> torch.Tensor:
    """The top-k reduction of the low-batch MoE kernel's rows: BF16 [rows,
    5120] (``routed_expert_finalize_cute``)."""
    import cutlass
    from cutlass import Int32

    from ...ops.cute_dsl.primitives import current_cu_stream
    from ..moe.routed_expert_gemm import _finalize_kernel, _pointer

    rows, top_k = routed.expert_weights.shape
    device = routed.gemm2_permuted.device
    output = torch.empty(
        (rows, routed.gemm2_permuted.shape[-1]), dtype=torch.bfloat16, device=device
    )
    _finalize_kernel(device.index)(
        _pointer(cutlass.BFloat16, routed.gemm2_permuted, 16),
        _pointer(cutlass.Int32, routed.expanded_idx_to_permuted_idx, 4),
        _pointer(cutlass.Float32, routed.expert_weights, 4),
        _pointer(cutlass.BFloat16, output, 16),
        Int32(rows),
        Int32(top_k),
        current_cu_stream(),
    )
    return output


class HopperDraftDecode(DraftDecode):
    """The decode executor over the DSpark draft model's layers on Hopper."""

    def __init__(
        self, target: "HopperSpecializedDecode", backbone: "DSparkDeepseekV4Model"
    ) -> None:
        if backbone.use_sequence_parallel or backbone.use_native_mega_moe:
            raise ValueError(
                "decode_backend=latency_specialized runs the DSpark draft without "
                "sequence parallelism or the native mega MoE"
            )
        self.backbone = backbone
        self.target = target
        layers = [cast("DeepseekV4DecoderLayer", layer) for layer in backbone.layers]
        attns: list[DeepseekV4FlashMLAAttention] = []
        for layer in layers:
            if type(layer.attn) is not DeepseekV4FlashMLAAttention:
                raise ValueError(
                    "decode_backend=latency_specialized on Hopper needs the "
                    f"FlashMLA sparse attention, not {type(layer.attn).__name__}"
                )
            attns.append(layer.attn)
        self.max_rows = target.max_rows
        self.mhc_stream = target.mhc_stream
        self.all_reduce = target.all_reduce
        self.boundaries = target.boundaries
        self.tables = tables = FlashMlaTables(attns)
        self.small_mla = tables.small
        self.mla = mla = FlashMlaDecode(tables)
        self.plans = [
            DraftLayerPlan(
                layer=layer,
                attention=HopperAttentionStep(attn, tables, mla, self.all_reduce),
                moe=HopperDraftMoEStep(layer.ffn, self.max_rows),
            )
            for layer, attn in zip(layers, attns)
        ]
        self._swa_prefixes = sorted({attn.swa_cache_layer.prefix for attn in attns})
        self.main_proj = DenseFp8Linear.of("main_proj", backbone.main_proj)
        if self.main_proj is not None:
            install_main_proj(backbone.main_proj, self.main_proj)

    def plan(self, positions: torch.Tensor) -> DecodeStep | None:
        """As the target's: only FULL CUDA graph captures of pure decode steps
        of up to ``max_rows`` rows run the executor."""
        rows = positions.shape[0]
        if (
            not 0 < rows <= self.max_rows
            or not is_forward_context_available()
            or not torch.cuda.is_current_stream_capturing()
            or not captures_pure_decode(self._swa_prefixes)
        ):
            return None
        return self.target.gates(rows)._replace(executor=self)


def install_main_proj(main_proj: torch.nn.Module, linear: DenseFp8Linear) -> None:
    """Run ``main_proj``'s forward on ``linear`` for the rows it serves."""
    stock = main_proj.forward
    return_bias = getattr(main_proj, "return_bias", True)

    def forward(x: torch.Tensor):
        if x.dim() != 2 or not linear.serves(x.shape[0]):
            return stock(x)
        out = linear(x.contiguous())
        return (out, None) if return_bias else out

    main_proj.forward = forward
