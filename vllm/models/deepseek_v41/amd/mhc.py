# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSV4.1 delayed mHC seams on ROCm.

On gfx950, `mhc_pre_delayed_overlap` moves the gate projection onto a side
stream, overlapped with the next sublayer, and with
VLLM_ROCM_USE_AITER_FLYDSL_MHC `mhc_pre_delayed_flydsl` runs the seams AITER's
FlyDSL kernel is faster at, and `mhc_tail_flydsl` the step-end tail. On
gfx942, `mhc_pre_delayed_seam` runs AITER's `mhc_fused_post_pre_delayed`, which
applies the previous sublayer's post, collapses the residual with the carried
pre-mix and computes the next gates in one call, from `MHC_SEAM_MIN_TOKENS`
tokens on seams that have a post block and no fused norm.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.triton import mhc_post_collapse_rms_norm_triton
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM,
    MHCPreDelayedOp,
    _aiter_mhc_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.deepseek_v4.amd.model import attach_mxfp8_input
from vllm.platforms.rocm import get_cdna_version
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

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
        self._mega_mhc_ready = False

    def prepare_mega_mhc(self, hidden_size: int, device: torch.device) -> None:
        """Allocate AITER Mega-MHC's per-stream scratch on the side stream.

        The gates run there only inside FULL graph capture, which cannot
        allocate it; sizing it for `MHC_OVERLAP_MAX_TOKENS` covers every
        batch the overlap captures.
        """
        if self._mega_mhc_ready:
            return
        with torch.cuda.stream(self.stream):
            rocm_aiter_ops.mega_mhc_prepare_stream(
                MHC_OVERLAP_MAX_TOKENS, hidden_size, device
            )
        self._mega_mhc_ready = True

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
    fn_packed: torch.Tensor | None = None,
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
            fn_packed=fn_packed,
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
            fn_packed=fn_packed,
        )
    for tensor in (residual, pre_mix):
        tensor.record_stream(stream)
    for tensor in (post_mix, comb_mix, next_pre_mix):
        tensor.record_stream(main)
    return residual, post_mix, comb_mix, layer_input, next_pre_mix


def mhc_pre_delayed_mxfp8(
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
    fn_packed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`pre_delayed` whose Mega-MHC seam also writes the layer input as MXFP8.

    The sublayer's first GEMM then reads that copy (see `attach_mxfp8_input`)
    instead of quantizing the bf16 layer input in a separate launch. Seams
    `pre_delayed` would not run on Mega-MHC are left to it.
    """
    if (
        not HAS_AITER_MHC_FUSED_POST_PRE_DELAYED_RMS_NORM
        or x is not None
        or norm_weight is None
        or fn_packed is None
        or norm_weight.dtype != torch.bfloat16
        or not _aiter_mhc_supported(residual, norm_weight, supports_norm=True)
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
            fn_packed=fn_packed,
        )
    residual_out = torch.empty_like(residual) if sublayer_out is not None else None
    post_mix, comb_mix, layer_input, next_pre_mix, values, scales = (
        rocm_aiter_ops.mhc_mega_post_pre_delayed_rms_norm_mxfp8(
            residual,
            fn_packed,
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
            norm_weight,
            norm_eps,
            residual_out,
        )
    )
    attach_mxfp8_input(
        layer_input,
        QuantizedActivation(
            values, scales, layer_input.dtype, layer_input.shape, kMxfp8Dynamic
        ),
    )
    return (
        residual if residual_out is None else residual_out,
        post_mix,
        comb_mix,
        layer_input,
        next_pre_mix,
    )


def get_flydsl_mhc_ops(hidden_size: int, hc_mult: int) -> Any | None:
    """AITER's FlyDSL mHC kernels (`aiter.ops.flydsl`) if
    VLLM_ROCM_USE_AITER_FLYDSL_MHC is on and they take this model's shape,
    else None."""
    ops = rocm_aiter_ops.flydsl_mhc_ops()
    if ops is None or not ops.flydsl_mhc_post_pre_rmsnorm_supported(
        0, hidden_size, hc_mult
    ):
        return None
    logger.info_once("Using AITER's FlyDSL mHC seam and step-end tail kernels.")
    return ops


def flydsl_seam_takes(
    ops: Any, residual_shape: tuple[int, ...], aux_mean: bool = False
) -> bool:
    """Whether the FlyDSL seam takes a `[T, hc_mult, hidden_size]` residual
    and is faster than Mega-MHC at its row count (AITER's measured rule). With
    `aux_mean` the seam also returns the mean of the post-mixed streams, which
    replaces an `mhc_post` and a mean launch."""
    num_tokens, hc_mult, hidden_size = residual_shape
    return bool(
        ops.flydsl_mhc_post_pre_rmsnorm_supported(num_tokens, hidden_size, hc_mult)
        and ops.flydsl_mhc_post_pre_rmsnorm_preferred(num_tokens, aux_mean=aux_mean)
    )


@functools.cache
def _flydsl_seam_max_tokens(
    ops: Any, hidden_size: int, hc_mult: int, aux_mean: bool
) -> int:
    """The most rows at which `flydsl_seam_takes` holds (AITER's rule)."""
    num_tokens = most = 0
    while ops.flydsl_mhc_post_pre_rmsnorm_supported(
        num_tokens + 1, hidden_size, hc_mult
    ):
        num_tokens += 1
        if ops.flydsl_mhc_post_pre_rmsnorm_preferred(num_tokens, aux_mean=aux_mean):
            most = num_tokens
    return most


@functools.cache
def warmup_flydsl_seam(
    ops: Any,
    device: torch.device,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_repeat: int,
    *,
    mxfp8: bool,
    aux_mean: bool,
    has_post: bool,
    identity_pre: bool,
) -> None:
    """Compile one FlyDSL seam variant for every split configuration AITER's
    rule can pick, once per process, so no seam compiles while serving: each
    (split configuration, variant) pair is a separate kernel that otherwise
    compiles on its first call, which under DP/EP would stall the other ranks
    in the next collective. Runs on the current stream, outside capture."""
    ops.warmup_flydsl_mhc_post_pre_rmsnorm(
        _flydsl_seam_max_tokens(ops, hidden_size, hc_mult, aux_mean),
        out_dtype="mxfp8" if mxfp8 else "bf16",
        aux_mean=aux_mean,
        has_post=has_post,
        identity_pre=identity_pre,
        sinkhorn_repeat=sinkhorn_repeat,
        device=device,
    )


@functools.cache
def warmup_flydsl_tail(
    ops: Any, device: torch.device, hidden_size: int, hc_mult: int, keep_residual: bool
) -> None:
    """Compile the FlyDSL tail variants the model can launch (with the
    post-mix, keeping the residual or not, and collapse + norm only), once per
    process, so none compiles while serving."""
    residual = torch.zeros(1, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    pre_mix = torch.zeros(1, hc_mult, device=device)
    weight = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    post = (
        torch.zeros(1, hidden_size, dtype=torch.bfloat16, device=device),
        torch.zeros(1, hc_mult, 1, device=device),
        torch.zeros(1, hc_mult, hc_mult, device=device),
    )
    ops.flydsl_mhc_post_collapse_rmsnorm(residual, pre_mix, weight, 1e-6)
    ops.flydsl_mhc_post_collapse_rmsnorm(
        residual, pre_mix, weight, 1e-6, *post, keep_residual=keep_residual
    )


def mhc_pre_delayed_flydsl(
    fallback: Callable[..., tuple[torch.Tensor, ...]],
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
    fn_packed: torch.Tensor | None = None,
    *,
    ops: Any,
    fn_flydsl: torch.Tensor,
    mxfp8: bool,
    aux_out: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A Mega-MHC seam on AITER's FlyDSL kernel where `flydsl_seam_takes`.

    Same arguments and returns as `MHCPreDelayedOp`; `fn_flydsl` is `fn`
    packed by `flydsl_mhc_pack_fn` (not Mega-MHC's layout). With `mxfp8` the
    layer input carries its MXFP8 copy, as in `mhc_pre_delayed_mxfp8`. Other
    seams run `fallback`. With `aux_out`, which the caller gives only to a
    post-mix seam `flydsl_seam_takes` with `aux_mean`, the mean of the
    post-mixed streams is appended to it.

    The kernel's split-K scratch is keyed by the stream a seam is launched or
    captured on, so all seams captured on one stream share it: replays of
    graphs captured on the same stream must not overlap. The model runner
    replays the target's and the DSpark drafter's graphs one after the other on
    one stream, and DBO (separate ubatch streams) keeps Mega-MHC. The eager
    warm-up before capture, on the capture stream, allocates the scratch.
    """
    if aux_out is None and (
        x is not None
        or norm_weight is None
        or norm_weight.dtype != torch.bfloat16
        or residual.dim() != 3
        or not flydsl_seam_takes(ops, tuple(residual.shape))
    ):
        return fallback(
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
            fn_packed=fn_packed,
        )
    assert aux_out is None or sublayer_out is not None
    post: tuple[torch.Tensor | None, ...] = (None, None, None)
    if sublayer_out is not None:
        assert post_layer_mix is not None and comb_res_mix is not None
        post = (
            sublayer_out.contiguous(),
            post_layer_mix.contiguous(),
            comb_res_mix.contiguous(),
        )
    out = ops.flydsl_mhc_post_pre_rmsnorm(
        residual.contiguous(),
        fn_flydsl,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        None if pre_mix is None else pre_mix.contiguous(),
        *post,
        norm_weight,
        norm_eps,
        out_dtype="mxfp8" if mxfp8 else "bf16",
        aux_mean=aux_out is not None,
    )
    residual, post_mix, comb_mix, layer_input, next_pre_mix = out[:5]
    if aux_out is not None:
        aux_out.append(out[5])
    if mxfp8:
        layer_input, values, scales = layer_input
        attach_mxfp8_input(
            layer_input,
            QuantizedActivation(
                values, scales, layer_input.dtype, layer_input.shape, kMxfp8Dynamic
            ),
        )
    return residual, post_mix, comb_mix, layer_input, next_pre_mix


def mhc_tail_flydsl(
    ops: Any,
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    *,
    residual_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The step-end tail in one FlyDSL launch: the last post-mix (when
    `sublayer_out` is given), the collapse with `pre_mix` and the final
    RMSNorm. Returns the normed `[T, hidden_size]` hidden states.

    `residual_out` (`[T, hc_mult, hidden_size]`, e.g. a view of the MTP hidden
    buffer) receives the post-mixed residual, or a copy of `residual` when
    there is no post-mix.
    """
    residual, pre_mix = residual.contiguous(), pre_mix.contiguous()
    if sublayer_out is None:
        if residual_out is not None:
            residual_out.copy_(residual)
        out, _, _ = ops.flydsl_mhc_post_collapse_rmsnorm(
            residual, pre_mix, norm_weight, norm_eps
        )
        return out
    assert post_layer_mix is not None and comb_res_mix is not None
    out, _, _ = ops.flydsl_mhc_post_collapse_rmsnorm(
        residual,
        pre_mix,
        norm_weight,
        norm_eps,
        sublayer_out.contiguous(),
        post_layer_mix.contiguous(),
        comb_res_mix.contiguous(),
        residual_out=residual_out,
    )
    return out


# Chosen so decode-only steps at low concurrency keep their current path. With
# DSpark (1 + 5 tokens per request) decode steps reach it from 11 requests.
MHC_SEAM_MIN_TOKENS = 64


@functools.cache
def supports_mhc_seam(hidden_size: int, hc_mult: int) -> bool:
    """True on gfx942 when AITER's probe accepts this shape."""
    if not HAS_AITER_MHC or get_cdna_version() != 3:
        return False
    from aiter.ops.mhc import mhc_fused_post_pre_delayed_asm_supported

    return bool(mhc_fused_post_pre_delayed_asm_supported(hidden_size, hc_mult))


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
    fn_packed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`pre_delayed` with the seam run as one AITER call where it applies.

    Same arguments and returns as `MHCPreDelayedOp`. Layer 0 (`x`), seams
    without a post block (Engram), and seams with a fused norm or a packed
    `fn` (Mega-MHC) keep `pre_delayed`. The caller checks `supports_mhc_seam`
    once.
    """
    if (
        sublayer_out is None
        or post_layer_mix is None
        or comb_res_mix is None
        or x is not None
        or norm_weight is not None
        or fn_packed is not None
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
            fn_packed=fn_packed,
        )
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
