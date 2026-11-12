# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The delayed mHC seam as one AITER launch on gfx942.

AITER's `mhc_fused_post_pre_delayed` applies the previous sublayer's post,
collapses the residual with the carried pre-mix, and computes the next
gates. The decoder calls it from `MIN_TOKENS` tokens, on seams that have a
post block and no fused norm; shorter steps and the other seams keep
`MHCPreDelayedOp`.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mhc import HAS_AITER_MHC, MHCPreDelayedOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# AITER's few-token kernel runs below 60 tokens (MHC_SEAM_SMALL_MAX_T), timed
# on 304-CU gfx942. 64 is where vLLM starts calling the seam, not a measured
# crossover: the seam was faster at 32, 60, 63, 64, 128 and 8192 tokens.
MIN_TOKENS = 64


@functools.cache
def supported(hidden_size: int, hc_mult: int) -> bool:
    """True when AITER's probe accepts this shape. AITER builds without the
    gfx942 seam (v0.1.27 and earlier) do not export the probe."""
    if not HAS_AITER_MHC or not current_platform.is_rocm():
        return False
    try:
        from aiter.ops.mhc import mhc_fused_post_pre_delayed_asm_supported
    except Exception:
        return False
    return bool(mhc_fused_post_pre_delayed_asm_supported(hidden_size, hc_mult))


def _seam(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns residual_out, post_mix, comb_mix, layer_input, next_pre_mix.

    `post_layer_mix` and `pre_mix` must already be fp32 of shape `(T, hc)` or
    `(T, hc, 1)`; AITER checks that and raises instead of casting.
    """
    from aiter.ops.mhc import mhc_fused_post_pre_delayed

    return mhc_fused_post_pre_delayed(
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
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
    )


def _seam_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    sublayer_out: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = residual.shape
    device = residual.device
    # AITER copies residual contiguous before the store. empty_like would
    # keep a permuted residual's strides, and torch.compile rejects that.
    return (
        torch.empty(residual.shape, dtype=residual.dtype, device=device),
        torch.empty(num_tokens, hc_mult, 1, dtype=torch.float32, device=device),
        torch.empty(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device),
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device),
        torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device),
    )


direct_register_custom_op(
    op_name="mhc_fused_post_pre_delayed_gfx942",
    op_func=_seam,
    mutates_args=[],
    fake_impl=_seam_fake,
)


def mhc_pre_delayed_seam(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`pre_delayed` with the seam run as one AITER launch where it applies.

    Same arguments and returns as `MHCPreDelayedOp`. Layer 0 (`x`), seams
    without a post block (Engram) and seams with a fused norm keep
    `pre_delayed`. The caller checks `supported` once.
    """
    if (
        sublayer_out is None
        or post_layer_mix is None
        or comb_res_mix is None
        or x is not None
        or norm_weight is not None
        or residual.shape[0] < MIN_TOKENS
    ):
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
    logger.info_once(
        "mhc_seam: aiter mhc_fused_post_pre_delayed from %d tokens", MIN_TOKENS
    )
    return torch.ops.vllm.mhc_fused_post_pre_delayed_gfx942(
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
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
    )


_warmed_up = False


def warmup(hidden_size: int, hc_mult: int, device: torch.device) -> None:
    """Load `module_mhc_seam_asm` before CUDA-graph capture.

    One launch compiles the module. `pre_mix` None fills AITER's cached
    identity gate, which a later graph keeps reading.
    """
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True
    num_tokens = MIN_TOKENS
    mix_hc = 2 * hc_mult + hc_mult * hc_mult
    f32 = dict(dtype=torch.float32, device=device)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    comb_res_mix = torch.zeros(num_tokens, hc_mult, hc_mult, **f32)
    comb_res_mix[..., 0] = 1
    _seam(
        torch.zeros(num_tokens, hc_mult, hidden_size, **bf16),
        torch.zeros(mix_hc, hc_mult * hidden_size, **f32),
        torch.ones(3, **f32),
        torch.zeros(mix_hc, **f32),
        1e-6,
        1e-6,
        1e-6,
        1.0,
        20,
        None,
        torch.zeros(num_tokens, hidden_size, **bf16),
        torch.zeros(num_tokens, hc_mult, 1, **f32),
        comb_res_mix,
    )
    torch.accelerator.synchronize()
