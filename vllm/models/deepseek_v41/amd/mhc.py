# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap the DSV4.1 mHC gate projection with the next sublayer on ROCm."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.triton import mhc_post_collapse_rms_norm_triton
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
    MHCPreDelayedOp,
)
from vllm.platforms import current_platform

# MI355X TP2, TP4 and TP8 improve through 128 tokens, shrinking from about +5%
# output throughput at 1 token to +0.3-2% at 96-128.
MHC_OVERLAP_MAX_TOKENS = 128


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    config = vllm_config.model_config.hf_config
    return (
        (on_gfx942() or on_gfx950())
        and HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM
        and config.hc_mult == 4
        and not vllm_config.parallel_config.use_ubatching
    )


def mhc_pre_delayed_overlap(
    pre_delayed: MHCPreDelayedOp,
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    *,
    stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``pre_delayed`` with the gate projection moved onto ``stream``.

    Returns residual, post_mix, comb_mix, layer_input, next_pre_mix. Only the
    residual and layer input are ready on the caller stream; join ``stream``
    before using the gates. The residual and gates match the fused AITER seam
    bit for bit, since the gates are ``pre_delayed`` applied to the same new
    residual; the layer input's RMSNorm sums its row in another order, which
    can move it by up to one bf16 ulp. Seams without a post block, a carried
    pre-mix or a fused norm run ``pre_delayed`` on the caller stream.
    """
    main = torch.cuda.current_stream()
    if (
        x is not None
        or pre_mix is None
        or norm_weight is None
        or sublayer_out is None
        or post_layer_mix is None
        or comb_res_mix is None
    ):
        # Fork anyway: the caller joins after every seam, and a join needs the
        # stream to be part of the capture.
        stream.wait_stream(main)
        return pre_delayed(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            x=x,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            sublayer_out=sublayer_out,
            post_layer_mix=post_layer_mix,
            comb_res_mix=comb_res_mix,
        )
    residual, layer_input = mhc_post_collapse_rms_norm_triton(
        residual,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
    )
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        _, post_mix, comb_mix, _, next_pre_mix = pre_delayed(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    for tensor in (residual, pre_mix):
        tensor.record_stream(stream)
    for tensor in (post_mix, comb_mix, next_pre_mix):
        tensor.record_stream(main)
    return residual, post_mix, comb_mix, layer_input, next_pre_mix
