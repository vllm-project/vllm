# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split shifted mHC pre so coefficient generation can overlap the sublayer."""

import torch


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

    Args:
        residual: Contiguous BF16 residual streams, shaped [tokens, 4, 5120].
        fn: FP32 projection weights.
        hc_scale: Three coefficient scales.
        hc_base: Bias for the 24 coefficients.
        rms_eps: Projection RMS normalization epsilon.
        hc_pre_eps: Pre-mix sigmoid epsilon.
        hc_sinkhorn_eps: Sinkhorn normalization epsilon.
        hc_post_mult_value: Post-mix multiplier.
        sinkhorn_repeat: Sinkhorn iteration count.
        pre_mix: Previous sublayer's pre-mix; None selects residual stream zero.
        x: Optional broadcast projection input for the first layer.
        norm_weight: BF16 RMSNorm weight for the collapsed sublayer input.
        norm_eps: Sublayer RMSNorm epsilon.
        stream: Dedicated coefficient stream, distinct from the caller stream.

    Returns:
        Post mix, residual mix, normalized input, and the next pre-mix. Only the
        normalized input is ready on the caller stream: the caller must wait
        for stream before using the other outputs or crossing a graph boundary.
    """
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    assert hc == 4 and hidden == 5120
    assert residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert norm_weight is not None
    if x is None:
        x = residual.view(n, hc * hidden)
    assert x.is_contiguous()
    splits = compute_mhc_pre_num_splits(x.shape[1], n) if n else 1
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    layer_input = torch.empty((n, hidden), device=residual.device, dtype=torch.bfloat16)
    mix = torch.empty(
        (splits, n, hc * (hc + 2)), device=residual.device, dtype=torch.float32
    )
    sqr = torch.empty((splits, n), device=residual.device, dtype=torch.float32)
    outputs = post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre
    if n == 0:
        return outputs
    args = (
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
    )
    fields = dict(
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
    # Prioritize input readiness for small SP shards before releasing statistics.
    if n <= 8:
        MHC_PRE_NORM_KERNEL(*args, **fields, split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        MHC_PRE_NORM_KERNEL(*args, **fields, split_mode="stats")
    for tensor in (residual, x, fn, hc_scale, hc_base, mix, sqr, post, comb, next_pre):
        tensor.record_stream(stream)
    if n > 8:
        MHC_PRE_NORM_KERNEL(*args, **fields, split_mode="input")
    return outputs
