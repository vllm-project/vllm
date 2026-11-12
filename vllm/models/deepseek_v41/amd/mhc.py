# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSV4.1 delayed mHC seams on ROCm.

On gfx950, `mhc_pre_delayed_overlap` moves the gate projection onto a side
stream, overlapped with the next sublayer. On gfx942, `mhc_pre_delayed_seam`
runs AITER's `mhc_fused_post_pre_delayed`, which applies the previous
sublayer's post, collapses the residual with the carried pre-mix and computes
the next gates in one call, from `MHC_SEAM_MIN_TOKENS` tokens on seams that
have a post block and no fused norm.
"""

import functools

import torch

from vllm.config import VllmConfig
from vllm.model_executor.kernels.mhc.triton import mhc_post_collapse_rms_norm_triton
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
    MHCPreDelayedOp,
)
from vllm.platforms.rocm import get_cdna_version
from vllm.utils.torch_utils import aux_stream, direct_register_custom_op

MHC_OVERLAP_MAX_TOKENS = 128


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    return (
        get_cdna_version() == 4
        and HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM
        and config.hc_mult == 4
        and not vllm_config.parallel_config.use_ubatching
    )


class MHCOverlapStream:
    """The global aux stream with the fork and join events of the overlap.

    Every fork and join waits on its event before the next records it, so all
    layers share the two events.
    """

    def __init__(self) -> None:
        stream = aux_stream()
        assert stream is not None
        self.stream = stream
        self._fork_event = torch.cuda.Event()
        self._join_event = torch.cuda.Event()

    def fork(self) -> None:
        self._fork_event.record()
        self.stream.wait_event(self._fork_event)

    def join(self) -> None:
        self._join_event.record(self.stream)
        torch.cuda.current_stream().wait_event(self._join_event)


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
    overlap: MHCOverlapStream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`pre_delayed` with the gate projection moved onto `overlap.stream`.

    Returns residual, post_mix, comb_mix, layer_input, next_pre_mix. Only the
    residual and layer input are ready on the caller stream; call
    `overlap.join()` before using the gates. Seams without a post block, a
    carried pre-mix or a fused norm run `pre_delayed` on the caller stream.
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
        overlap.fork()
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
    stream = overlap.stream
    overlap.fork()
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


# Chosen so decode-only steps at low concurrency keep their current path. With
# DSpark (1 + 5 tokens per request) decode steps reach it from 11 requests.
MHC_SEAM_MIN_TOKENS = 64


@functools.cache
def supports_mhc_seam(hidden_size: int, hc_mult: int) -> bool:
    """True when AITER's probe accepts this shape."""
    if not HAS_AITER_MHC:
        return False
    from aiter.ops.mhc import mhc_fused_post_pre_delayed_asm_supported

    return bool(mhc_fused_post_pre_delayed_asm_supported(hidden_size, hc_mult))


def _mhc_seam(
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


def _mhc_seam_fake(
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
    op_name="mhc_fused_post_pre_delayed_aiter",
    op_func=_mhc_seam,
    mutates_args=[],
    fake_impl=_mhc_seam_fake,
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
    """`pre_delayed` with the seam run as one AITER call where it applies.

    Same arguments and returns as `MHCPreDelayedOp`. Layer 0 (`x`), seams
    without a post block (Engram) and seams with a fused norm keep
    `pre_delayed`. The caller checks `supports_mhc_seam` once.
    """
    if (
        sublayer_out is None
        or post_layer_mix is None
        or comb_res_mix is None
        or x is not None
        or norm_weight is not None
        or residual.shape[0] < MHC_SEAM_MIN_TOKENS
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
    return torch.ops.vllm.mhc_fused_post_pre_delayed_aiter(
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


_mhc_seam_warmed_up = False


def warmup_mhc_seam(hidden_size: int, hc_mult: int, device: torch.device) -> None:
    """Load `module_mhc_seam_asm` before CUDA-graph capture.

    One call compiles the module. `pre_mix` None fills AITER's cached
    identity gate, which a later graph keeps reading.
    """
    global _mhc_seam_warmed_up
    if _mhc_seam_warmed_up:
        return
    _mhc_seam_warmed_up = True
    num_tokens = MHC_SEAM_MIN_TOKENS
    mix_hc = 2 * hc_mult + hc_mult * hc_mult
    f32 = dict(dtype=torch.float32, device=device)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    comb_res_mix = torch.zeros(num_tokens, hc_mult, hc_mult, **f32)
    comb_res_mix[..., 0] = 1
    _mhc_seam(
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
