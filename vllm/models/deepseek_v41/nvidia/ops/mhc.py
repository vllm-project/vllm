# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""

from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.
MHC_OVERLAP_MAX_TOKENS = 16


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Check kernel requirements and safety of sharing the coefficient stream."""
    config = vllm_config.model_config.hf_config
    # DeepGEMM's prenorm kernel requires K % 64 == 0, N % 8 == 0, and N <= 32.
    mix_size = config.hc_mult * (config.hc_mult + 2)
    return (
        current_platform.is_device_capability_family(100)
        and is_deep_gemm_supported()
        and config.hidden_size % 64 == 0
        and 0 < mix_size <= 32
        and mix_size % 8 == 0
        and not vllm_config.parallel_config.use_ubatching
    )


def mhc_pre_delayed_overlap(
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
    *,
    stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the input on the caller stream and coefficients on another stream.

    Returns post mix, residual mix, normalized input, and next pre-mix. Only the
    input is ready on the caller stream; join stream before using coefficients.
    The caller must retain the weights and coefficient outputs until the join.
    """
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    assert residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert norm_weight is not None
    if x is None:
        x = residual.view(n, hc * hidden)
    assert x.is_contiguous()
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    layer_input = torch.empty((n, hidden), device=residual.device, dtype=torch.bfloat16)
    outputs = post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre
    if n == 0:
        return outputs
    splits = compute_mhc_pre_num_splits(x.shape[1], n)
    mix = torch.empty(
        (splits, n, hc * (hc + 2)), device=residual.device, dtype=torch.float32
    )
    sqr = torch.empty((splits, n), device=residual.device, dtype=torch.float32)
    epilogue = partial(
        MHC_PRE_NORM_KERNEL,
        mix,
        sqr,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre,
        layer_input,  # Unused aux output in split modes.
        hidden_size=hidden,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
    )
    main = torch.cuda.current_stream()
    # Prioritize input readiness for small batches before releasing statistics.
    if n <= 8:
        epilogue(split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        epilogue(split_mode="stats")
    for tensor in (x, mix, sqr):
        tensor.record_stream(stream)
    if n > 8:
        epilogue(split_mode="input")
    return outputs
