# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap the DSV4.1 mHC gate projection with the next sublayer on ROCm."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.triton import mhc_post_collapse_rms_norm_triton
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
)
from vllm.platforms import current_platform

# MI355X TP4 improves through 128 tokens, shrinking from +5% output throughput
# at 1 token to +1.7% at 128.
MHC_OVERLAP_MAX_TOKENS = 128


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    return (
        current_platform.is_rocm()
        and HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM
        and config.hc_mult == 4
        and not vllm_config.parallel_config.use_ubatching
    )


def mhc_seam_overlap(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    pre_mix: torch.Tensor,
    sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``MHCPreDelayedOp`` with the gate projection moved onto ``stream``.

    Returns residual, post_mix, comb_mix, layer_input, next_pre_mix. Only the
    residual and layer input are ready on the caller stream; join ``stream``
    before using the gates. The residual and gates match the fused AITER seam
    bit for bit, since the gates are AITER's kernel applied to the same new
    residual; the layer input's RMSNorm sums its row in another order, which
    can move it by up to one bf16 ulp.
    """
    residual, layer_input = mhc_post_collapse_rms_norm_triton(
        residual,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
    )
    main = torch.cuda.current_stream()
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        post_mix, comb_mix, _, next_pre_mix = (
            torch.ops.vllm.mhc_fused_post_pre_delayed_rms_norm_aiter(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix,
                None,
                None,
                None,
                norm_weight,
                norm_eps,
                None,
            )
        )
    for tensor in (residual, pre_mix):
        tensor.record_stream(stream)
    for tensor in (post_mix, comb_mix, next_pre_mix):
        tensor.record_stream(main)
    return residual, post_mix, comb_mix, layer_input, next_pre_mix
