# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mHC boundaries of the specialized decode.

Every decoder sublayer ends in a TP-partial output (the attention's ``wo_b``,
the MoE's routed and shared experts). The next boundary reduces it across the
TP group, mixes it into the hc residual streams (mHC post) and collapses and
normalizes the streams into the next sublayer's input (mHC pre). Here the
fused all-reduce + mHC kernel (``BoundaryAllReduce``) does all of that for every
step the executor runs; the next sublayer's mHC coefficients then come from the
stock ``mhc_pre_delayed_overlap`` on the shared mHC side stream.

The fused kernel also finalizes a MoE output left unfinalized, collects a
``wo_b`` output the publishing GEMM already stored into its mailbox, writes the
attention input as MXFP8 for ``fused_wqa_wkv``, and writes only the streams
(residual-only) where the next consumer is not a sublayer's pre: an Engram
injection and the final collapse.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v41.common.ops.query_quant import mxfp8_scale_bytes

from ...ops.mhc import mhc_pre_delayed_overlap

if TYPE_CHECKING:
    from ...model import DeepseekV4DecoderLayer
    from .all_reduce import BoundaryAllReduce


@dataclass
class BoundaryState:
    """The hc residual streams and the mixes carried to the next boundary."""

    residual: torch.Tensor
    # Post-mix and residual mix of the sublayer that produces ``pending``.
    post: torch.Tensor
    comb: torch.Tensor
    # The pre-mix the next collapse uses.
    pre: torch.Tensor
    # The sublayer's TP-partial output: a tensor (maybe published) or a MoE
    # output left unfinalized.
    pending: torch.Tensor | MoEOutput | None = None


def empty_quantized_layer_input(
    num_tokens: int, hidden_size: int, device: torch.device
) -> QuantizedActivation:
    """An unfilled MXFP8 layer input with FlashInfer's swizzled 128x4 scales."""
    shape = torch.Size((num_tokens, hidden_size))
    return QuantizedActivation(
        data=torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        scale=torch.empty(
            mxfp8_scale_bytes(num_tokens, hidden_size), dtype=torch.uint8, device=device
        ),
        orig_dtype=torch.bfloat16,
        orig_shape=shape,
        quant_key=kMxfp8Dynamic,
    )


class Boundaries:
    """The boundaries of one decode step's layer loop, on one fused kernel."""

    def __init__(self, all_reduce: "BoundaryAllReduce") -> None:
        self.all_reduce = all_reduce

    def first(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        streams: torch.Tensor | None = None,
    ) -> tuple[BoundaryState, torch.Tensor]:
        """The first layer's attention entry: the embedding broadcast to the
        hc streams (or written with them, ``streams``), whose identity pre-mix
        selects stream zero."""
        assert layer.hc_attn_fn_broadcast is not None
        residual = (
            x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            if streams is None
            else streams
        )
        return self._pre(layer, residual, layer.hc_attn_fn_broadcast, None, x)

    def pre(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        state: BoundaryState,
    ) -> tuple[BoundaryState, torch.Tensor]:
        """An Engram layer's attention entry, on the injected streams."""
        return self._pre(layer, residual, layer.hc_attn_fn, state.pre, None)

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
        """Reduce ``state.pending`` into the streams and prepare the attention
        (or with ``ffn``, the MoE) input of ``layer``.

        Returns the new state, the layer input, the same input as MXFP8 with
        ``quantize``, and with ``capture_aux`` the mean of the streams (on the
        side stream, after the coefficients: join it before reading).
        """
        if ffn:
            fn, scale, base = layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
            norm = layer.ffn_norm
        else:
            fn, scale, base = layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
            norm = layer.attn_norm
        rows, _, hidden = state.residual.shape
        quant = (
            empty_quantized_layer_input(rows, hidden, state.residual.device)
            if quantize
            else None
        )
        residual, layer_input = self._reduce(
            state,
            state.pre,
            norm,
            None if quant is None else (quant.data, quant.scale),
        )
        assert layer_input is not None
        stream = layer.mhc_stream
        assert stream is not None
        aux = None
        if capture_aux:
            streams_ready = torch.cuda.Event()
            streams_ready.record()
        post, comb, layer_input, next_pre = mhc_pre_delayed_overlap(
            residual,
            fn,
            scale,
            base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=state.pre,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
            stream=stream,
            layer_input=layer_input,
        )
        if capture_aux:
            # residual.mean(dim=1), after the coefficients on the side stream.
            aux = residual.new_empty(rows, hidden)
            stream.wait_event(streams_ready)
            with torch.cuda.stream(stream):
                torch.mean(residual, dim=1, out=aux)
            residual.record_stream(stream)
            aux.record_stream(stream)
        return BoundaryState(residual, post, comb, next_pre), layer_input, quant, aux

    def post_only(self, state: BoundaryState) -> torch.Tensor:
        """Reduce ``state.pending`` into the streams and return them, for an
        Engram injection or the final collapse."""
        if isinstance(state.pending, MoEOutput):
            residual, _ = self._reduce(state, None, None, None)
            return residual
        assert isinstance(state.pending, torch.Tensor)
        x = tensor_model_parallel_all_reduce(state.pending)
        return mhc_post_tilelang(x, state.residual, state.post, state.comb)

    def join(self, layer: "DeepseekV4DecoderLayer") -> None:
        """Make the current stream wait for the side-stream coefficients."""
        assert layer.mhc_stream is not None
        torch.cuda.current_stream().wait_stream(layer.mhc_stream)

    def _pre(
        self,
        layer: "DeepseekV4DecoderLayer",
        residual: torch.Tensor,
        fn: torch.Tensor,
        pre_mix: torch.Tensor | None,
        x: torch.Tensor | None,
    ) -> tuple[BoundaryState, torch.Tensor]:
        """The stock attention entry without a reduction: mHC pre of
        ``residual`` on the side stream."""
        post, comb, layer_input, next_pre = mhc_pre_delayed_overlap(
            residual,
            fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            x=x,
            norm_weight=layer.attn_norm.weight,
            norm_eps=layer.attn_norm.variance_epsilon,
            stream=layer.mhc_stream,
        )
        return BoundaryState(residual, post, comb, next_pre), layer_input

    def _reduce(
        self,
        state: BoundaryState,
        pre: torch.Tensor | None,
        norm: RMSNorm | None,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The fused kernel on ``state.pending``; residual-only without ``pre``."""
        norm_weight = None if norm is None else norm.weight
        norm_eps = 0.0 if norm is None else norm.variance_epsilon
        x = state.pending
        if isinstance(x, MoEOutput):
            routed = x.routed
            assert isinstance(routed, UnfinalizedMoEOutput)
            assert x.shared_output is not None
            return self.all_reduce.finalize(
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
                x.shared_output,
                state.residual,
                state.post,
                state.comb,
                pre,
                norm_weight,
                norm_eps,
                mxfp8_out,
            )
        assert x is not None
        return self.all_reduce(
            x,
            state.residual,
            state.post,
            state.comb,
            pre,
            norm_weight,
            norm_eps,
            mxfp8_out,
        )
