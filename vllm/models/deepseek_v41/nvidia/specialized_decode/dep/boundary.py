# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local mHC boundaries for TP1: no reduction across unrelated DP requests."""

import torch

from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_pre_delayed_overlap
from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
    Boundaries,
    BoundaryState,
    empty_quantized_layer_input,
)

from .local_boundary import LocalPostPre
from .owner import OwnerOutput
from .shared_boundary import FusedPostPre, LocalMoEOutput


class LocalBoundaries(Boundaries):
    def __init__(self, layers, stream):
        self.mhc_stream = stream
        self.local_reduce = LocalPostPre()
        self.shared_reduce = FusedPostPre()

    def _reduce(self, state, pre, norm, mxfp8_out):
        return self._reduce_local(state, pre, norm, mxfp8_out, False)[:2]

    def sublayer(self, layer, ffn, state, *, quantize=False, capture_aux=False):
        fn = layer.hc_ffn_fn if ffn else layer.hc_attn_fn
        scale = layer.hc_ffn_scale if ffn else layer.hc_attn_scale
        base = layer.hc_ffn_base if ffn else layer.hc_attn_base
        norm = layer.ffn_norm if ffn else layer.attn_norm
        rows, _, hidden = state.residual.shape
        quant = (
            empty_quantized_layer_input(rows, hidden, state.residual.device)
            if quantize
            else None
        )
        residual, layer_input, aux = self._reduce_local(
            state,
            state.pre,
            norm,
            None if quant is None else (quant.data, quant.scale),
            capture_aux,
        )
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
            stream=self.mhc_stream,
            layer_input=layer_input,
        )
        return BoundaryState(residual, post, comb, next_pre), layer_input, quant, aux

    def _reduce_local(self, state, pre, norm, mxfp8_out, capture_aux):
        pending = state.pending
        if isinstance(pending, OwnerOutput):
            if not torch.cuda.is_current_stream_capturing():
                raise RuntimeError("deferred DEP output escaped FULL graph capture")
            return pending.collective.prepare(
                pending.partials,
                state.residual,
                state.post,
                state.comb,
                pre,
                norm,
                None
                if mxfp8_out is None
                else (mxfp8_out[0].view(torch.uint8), mxfp8_out[1]),
                capture_aux,
                shared=pending.shared,
                publish_override=pending.publish,
            )()
        if isinstance(state.pending, LocalMoEOutput):
            return self.shared_reduce(
                state.pending.routed,
                state.residual,
                state.post,
                state.comb,
                pre,
                norm,
                None
                if mxfp8_out is None
                else (mxfp8_out[0].view(torch.uint8), mxfp8_out[1]),
                capture_aux,
                shared=state.pending.shared,
            )
        if not isinstance(state.pending, torch.Tensor):
            raise TypeError("DEP boundary requires finalized EP output")
        result = self.local_reduce(
            state.pending,
            state.residual,
            state.post,
            state.comb,
            pre,
            norm,
            None
            if mxfp8_out is None
            else (mxfp8_out[0].view(torch.uint8), mxfp8_out[1]),
            capture_aux,
        )
        if result is not None:
            return result
        residual = self.post_only(state)
        collapsed = (residual.float() * pre.unsqueeze(-1)).sum(1).to(residual.dtype)
        layer_input = norm(collapsed)
        if mxfp8_out is not None:
            from flashinfer import mxfp8_quantize

            quant, scales = mxfp8_quantize(layer_input, is_sf_swizzled_layout=True)
            mxfp8_out[0].copy_(quant)
            mxfp8_out[1].copy_(scales.reshape(-1))
        return residual, layer_input, residual.mean(dim=1) if capture_aux else None

    def post_only(self, state, fuse_partial=False):
        pending = state.pending
        if isinstance(pending, LocalMoEOutput):
            pending = pending.routed + pending.shared
        if not isinstance(pending, torch.Tensor):
            raise TypeError("cannot carry an unfinished EP combine across layers")
        return mhc_post_tilelang(pending, state.residual, state.post, state.comb)
