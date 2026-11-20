# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""

from functools import cache, partial
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_delayed_tilelang,
    mhc_post_tilelang,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.import_utils import has_cutedsl

from .mega_mhc import (
    can_use_mega_mhc,
    mhc_shifted_post_pre_deep_gemm,
)

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    from ..specialized_decode.ampere.all_reduce import AmpereAllReduce
    from .cute_dsl import AllReduceMHC

logger = init_logger(__name__)

# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.
MHC_OVERLAP_MAX_TOKENS = 16

# The fused all-reduce + mHC kernel of the overlap path, built once by
# init_mhc_all_reduce.
_all_reduce_mhc: "AllReduceMHC | None" = None
# The SM80 boundaries' two-shot Lamport all-reduce, built once by
# init_ampere_all_reduce.
_ampere_all_reduce: "AmpereAllReduce | None" = None


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


# The SM80 mHC kernels (specialized_decode/ampere/mhc.py) take boundaries of
# up to this many rows; larger ones keep the TileLang kernels.
AMPERE_MHC_MAX_TOKENS = 256
# SM90 runs the same kernels up to this many rows, with the coefficients on
# CUDA cores (see ``_uses_mma``): on H100 at TP8 with five draft tokens they
# beat the stock boundary, whose projection runs on DeepGEMM's tensor-core
# kernel, at 48 rows and lost from 96.
SM90_MHC_MAX_TOKENS = 48
# The fused all-reduce is one-shot, as FlashInfer's MNNVL one-shot kernel,
# which vLLM's all-reduce leaves for its two-shot one above 1 MiB of all
# ranks' rows (MNNVL_ONE_SHOT_THRESHOLD): 12 rows of DSV4.1 at TP8. On H100 at
# TP8 with five draft tokens the fused boundary takes 2.5% off a decode step
# at 6 rows, and is 1.2% and 3.0% slower than the two-shot one at 24 and 48.
SM90_MHC_ALL_REDUCE_MAX_TOKENS = 12


@cache
def ampere_mhc_max_tokens() -> int:
    """The most rows an mHC boundary takes the SM80 CuTe DSL kernels for:
    all of them on SM80, small decode batches on SM90, none elsewhere."""
    # Imported here: the specialized_decode package imports this module.
    from ..specialized_decode.ampere import is_ampere

    if is_ampere():
        return AMPERE_MHC_MAX_TOKENS
    if current_platform.is_cuda() and current_platform.is_device_capability(90):
        return SM90_MHC_MAX_TOKENS
    return 0


def use_ampere_mhc() -> bool:
    """Whether any mHC boundary takes the SM80 CuTe DSL kernels."""
    return ampere_mhc_max_tokens() > 0


def supports_ampere_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Whether FULL graphs on SM80 and SM90 compute the mHC coefficients on
    a side stream: only for a hidden size the SM80 kernels take (otherwise
    the TileLang kernels run in order), and not with micro-batching, which
    the single ticket workspace rules out."""
    if not use_ampere_mhc() or vllm_config.parallel_config.use_ubatching:
        return False
    from ..specialized_decode.ampere.mhc import HC, supports

    return supports(1, vllm_config.model_config.hf_text_config.hidden_size, HC)


def prepare_ampere_mhc(hidden_size: int) -> None:
    """Allocate the SM80 mHC kernels' persistent buffers on the current
    device and compile them, before any capture."""
    from ..specialized_decode.ampere.mhc import HC, MAX_ROWS, prepare, supports

    assert MAX_ROWS == AMPERE_MHC_MAX_TOKENS
    if supports(1, hidden_size, HC):
        prepare(torch.accelerator.current_device_index(), hidden_size)


def _ampere_mhc(residual: torch.Tensor, norm_weight: torch.Tensor | None) -> bool:
    """Whether a boundary over `residual` takes the SM80 CuTe DSL kernels."""
    if norm_weight is None or not 0 < residual.shape[0] <= ampere_mhc_max_tokens():
        return False
    from ..specialized_decode.ampere.mhc import supports

    return supports(residual.shape[0], residual.shape[-1], residual.shape[1])


def ampere_projection_splits() -> tuple[int, ...]:
    """The split counts of the SM80 prefill projection (`_mhc_pre_split`),
    for the stock epilogue's warmup: none off SM80."""
    from ..specialized_decode.ampere import is_ampere

    if not is_ampere():
        return ()
    from ..specialized_decode.ampere.mhc_projection import SPLITS

    return (SPLITS,)


def _split_projection(
    residual: torch.Tensor, width: int, norm_weight: torch.Tensor | None
) -> bool:
    """Whether a boundary over `residual` projects its `width`-wide rows
    on the SM80 split-k kernel: on SM80, above the rows the SM80 boundary
    kernels take."""
    if norm_weight is None or residual.shape[0] <= AMPERE_MHC_MAX_TOKENS:
        return False
    if not ampere_projection_splits():
        return False
    from ..specialized_decode.ampere.mhc_projection import supports

    return supports(width)


def _mhc_pre_split(
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
    x: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_eps: float,
    capture_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mhc_pre_delayed_tilelang` with a norm weight on SM80 prefill: the
    projection on the split-k tensor-core kernel (mhc_projection.py), then
    the stock epilogue. Returns the post-, comb-mix, layer input, next
    pre-mix and the stream mean (empty unless `capture_aux`)."""
    from vllm.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL

    from ..specialized_decode.ampere.mhc_projection import mhc_projection

    num_tokens, hc_mult, hidden_size = residual.shape
    if x is None:
        x = residual.view(num_tokens, hc_mult * hidden_size)
    mixes, sqrsum = mhc_projection(x, fn)
    next_pre_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    post = torch.empty_like(next_pre_mix)
    comb = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    aux = torch.empty(
        num_tokens if capture_aux else 0,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    MHC_PRE_NORM_KERNEL(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre_mix,
        aux if capture_aux else layer_input,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc_mult,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
        write_aux=capture_aux,
    )
    return (
        post.unsqueeze(-1),
        comb.view(num_tokens, hc_mult, hc_mult),
        layer_input,
        next_pre_mix,
        aux,
    )


def mhc_pre_delayed_ampere(
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
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`mhc_pre_delayed_tilelang` on SM80: the CuTe DSL kernels, with the
    coefficients on `stream` when given (join it before reading them)."""
    if not _ampere_mhc(residual, norm_weight):
        width = residual.shape[1] * residual.shape[2] if x is None else x.shape[1]
        if _split_projection(residual, width, norm_weight):
            assert norm_weight is not None
            return _mhc_pre_split(
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
                x,
                norm_weight,
                norm_eps,
            )[:4]
        return mhc_pre_delayed_tilelang(
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
        )
    from ..specialized_decode.ampere.mhc import mhc_pre

    assert norm_weight is not None
    return mhc_pre(
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
        x,
        norm_weight,
        norm_eps,
        stream=stream,
    )


def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:
    """Whether the mHC boundaries fuse the TP all-reduce (`AllReduceMHC`):
    at TP4 on SM100, whose overlap path it serves, and at TP8 on SM90, where
    the boundaries of the SM80 kernels' path take it; both need NVLink
    multicast."""
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if current_platform.is_device_capability(90):
        tp_size = 8
    elif current_platform.is_device_capability_family(100):
        tp_size = 4
    else:
        return False
    if (
        parallel.tensor_parallel_size != tp_size
        or parallel.enable_expert_parallel
        or config.hidden_size != 5120
        or config.hc_mult != 4
    ):
        return False
    # The custom all-reduce's MNNVL buffers succeed exactly when NVLink
    # multicast does, which the fused kernel's own mailbox needs too.
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)


def mhc_all_reduce_max_tokens() -> int:
    """The most rows the fused all-reduce takes, and so the most a MoE may
    leave its top-k reduction to it for: `SM90_MHC_ALL_REDUCE_MAX_TOKENS`
    on SM90 (its boundaries up to `SM90_MHC_MAX_TOKENS` rows all-reduce
    first), the two-shot kernel's on SM80, `MHC_OVERLAP_MAX_TOKENS` on
    SM100."""
    if _ampere_all_reduce is not None:
        return min(ampere_mhc_max_tokens(), _ampere_all_reduce.capacity)
    if use_ampere_mhc():
        return min(ampere_mhc_max_tokens(), SM90_MHC_ALL_REDUCE_MAX_TOKENS)
    return MHC_OVERLAP_MAX_TOKENS


def init_mhc_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the fused all-reduce + mHC kernel the overlap path reduces with.

    Collective over the TP group, so every rank must call it, and only when
    `supports_mhc_all_reduce` holds.
    """
    global _all_reduce_mhc
    from .cute_dsl import AllReduceMHC

    config = vllm_config.model_config.hf_config
    max_num_tokens = mhc_all_reduce_max_tokens()
    _all_reduce_mhc = AllReduceMHC(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        max_num_tokens=max_num_tokens,
        top_k=config.num_experts_per_tok,
        device=current_platform.current_device(),
    )
    logger.info_once(
        "DSV4.1 mHC: CuTe DSL Lamport all-reduce fused with mHC for up to %d tokens.",
        max_num_tokens,
    )


def supports_ampere_all_reduce(vllm_config: VllmConfig) -> bool:
    """Whether the SM80 mHC boundaries take the TP all-reduce with their own
    kernel (`AmpereAllReduce`, a two-shot all-reduce that stores through
    peer mappings): on SM80, which has no NVLink multicast, at TP 2, 4 or 8
    where vLLM's custom all-reduce runs, so the GPUs are fully connected
    with peer access."""
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    tp_size = parallel.tensor_parallel_size
    if (
        not current_platform.is_device_capability(80)
        or tp_size not in (2, 4, 8)
        or parallel.enable_expert_parallel
        or config.hidden_size % (8 * tp_size)
    ):
        return False
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and not comm.disabled and comm.fully_connected


def init_ampere_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the SM80 boundaries' all-reduce. Collective over the TP group,
    so every rank must call it, and only when `supports_ampere_all_reduce`
    holds."""
    global _ampere_all_reduce
    from ..specialized_decode.ampere.all_reduce import AmpereAllReduce

    group = get_tp_group()
    config = vllm_config.model_config.hf_config
    _ampere_all_reduce = AmpereAllReduce(
        group.device_group,
        config.hidden_size,
        group.device,
        top_k=config.num_experts_per_tok,
    )
    logger.info_once(
        "DSV4.1 mHC: CuTe DSL two-shot Lamport all-reduce for up to %d tokens.",
        _ampere_all_reduce.capacity,
    )


def mhc_all_reduce_ready() -> bool:
    """Whether `init_mhc_all_reduce` has built the fused all-reduce (the
    target model does; its DSpark draft model reuses it)."""
    return _all_reduce_mhc is not None


def ampere_all_reduce_ready() -> bool:
    """Whether `init_ampere_all_reduce` has built the SM80 two-shot kernel
    (the target model does; its DSpark draft model reuses it)."""
    return _ampere_all_reduce is not None


def boundary_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """The TP all-reduce of a partial sum the model reduces itself (a
    sublayer output, the embedding's vocabulary shards): the SM80 two-shot
    Lamport kernel for the rows it takes, else the TP group's."""
    if _ampere_all_reduce is not None and _ampere_all_reduce.supports(x):
        return _ampere_all_reduce(x)
    return get_tp_group().all_reduce(x)


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
    layer_input: torch.Tensor | None = None,
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
    compute_input = layer_input is None
    if layer_input is None:
        layer_input = torch.empty(
            (n, hidden), device=residual.device, dtype=torch.bfloat16
        )
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
    if compute_input and n <= 8:
        epilogue(split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        epilogue(split_mode="stats")
    for tensor in (x, mix, sqr):
        tensor.record_stream(stream)
    if compute_input and n > 8:
        epilogue(split_mode="input")
    return outputs


def mhc_boundary_weight(
    hidden_size: int, hc_mult: int, device: torch.device
) -> torch.Tensor | None:
    """An empty `prepare_boundary_weight` buffer, or None where the CuTe DSL mHC
    boundary (`mhc_boundary.py`) cannot run."""
    if not (current_platform.is_device_capability_family(100) and has_cutedsl()):
        return None
    from .mhc_boundary import MhcBoundaryKernel, boundary_weight_shape

    if not MhcBoundaryKernel.supports(hidden_size, hc_mult):
        return None
    return torch.empty(
        boundary_weight_shape(hidden_size), dtype=torch.int32, device=device
    )


@cache
def _mhc_shifted_post_pre_boundary():
    from .mhc_boundary import mhc_shifted_post_pre_boundary

    return mhc_shifted_post_pre_boundary


def mhc_shifted_post_pre(
    x: torch.Tensor | MoEOutput,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
    boundary_weight: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Dispatch shifted post/pre to overlap, Mega-mHC, or fused TileLang.

    `boundary_weight` (`fn` laid out by `prepare_boundary_weight`, see
    `mhc_boundary_weight`) runs decode-size Mega-mHC batches on the CuTe DSL kernel.

    A MoE output left unfinalized is finalized inside the fused all-reduce.
    When stream is supplied, join it before consuming the returned coefficients.
    """
    if (
        _ampere_mhc(residual, norm_weight)
        and _all_reduce_mhc is not None
        and (
            isinstance(x, MoEOutput)
            or (
                reduce_results
                and stream is not None
                and residual.shape[0] <= mhc_all_reduce_max_tokens()
            )
        )
    ):
        # SM90 in FULL graphs up to the fused kernel's rows (and wherever a
        # MoE left its top-k reduction open, which it does only for those):
        # the all-reduce, post-mix, collapse and RMSNorm in the fused kernel,
        # then the next coefficients from its streams on the side stream, or
        # inline without one.
        from ..specialized_decode.ampere.mhc import mhc_coefficients

        assert pre_mix is not None and norm_weight is not None
        if isinstance(x, MoEOutput):
            routed = x.routed
            assert isinstance(routed, UnfinalizedMoEOutput)
            assert x.shared_output is not None
            residual, layer_input = _all_reduce_mhc.finalize(
                routed.gemm2_permuted,
                routed.expert_weights,
                routed.expanded_idx_to_permuted_idx,
                x.shared_output,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        else:
            residual, layer_input = _all_reduce_mhc(
                x.contiguous(),
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        assert layer_input is not None
        post, comb, pre = mhc_coefficients(
            None,
            residual,
            None,
            None,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            stream=stream,
        )
        aux = (
            residual.mean(dim=1)
            if capture_aux
            else residual.new_empty(0, residual.shape[-1])
        )
        return residual, post.unsqueeze(-1), comb, layer_input, pre, aux
    if _ampere_mhc(residual, norm_weight) and (
        isinstance(x, torch.Tensor) or _ampere_all_reduce is not None
    ):
        from ..specialized_decode.ampere.mhc import mhc_coefficients, mhc_post_pre

        assert norm_weight is not None
        if isinstance(x, MoEOutput) or (
            reduce_results
            and _ampere_all_reduce is not None
            and _ampere_all_reduce.supports(x)
        ):
            # The two-shot all-reduce, post-mix, collapse and RMSNorm in one
            # kernel (with the MoE's top-k sum and shared-expert add where
            # the MoE left them open), then the next coefficients from its
            # streams.
            assert _ampere_all_reduce is not None
            if isinstance(x, MoEOutput):
                routed = x.routed
                assert isinstance(routed, UnfinalizedMoEOutput) and routed.route_major
                assert x.shared_output is not None
                residual, layer_input, aux = _ampere_all_reduce.finalize_post_collapse(
                    routed.gemm2_permuted,
                    routed.expert_weights,
                    x.shared_output,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    pre_mix,
                    norm_weight,
                    norm_eps,
                    capture_aux,
                )
            else:
                residual, layer_input, aux = _ampere_all_reduce.post_collapse(
                    x.contiguous(),
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    pre_mix,
                    norm_weight,
                    norm_eps,
                    capture_aux,
                )
            post, comb, pre = mhc_coefficients(
                None,
                residual,
                None,
                None,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                stream=stream,
            )
            return residual, post.unsqueeze(-1), comb, layer_input, pre, aux
        if reduce_results:
            x = boundary_all_reduce(x)
        return mhc_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix,
            norm_weight,
            norm_eps,
            capture_aux,
            stream=stream,
        )
    layer_input = None
    if isinstance(x, MoEOutput):
        assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
        assert pre_mix is not None and norm_weight is not None
        routed = x.routed
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert x.shared_output is not None
        residual, layer_input = _all_reduce_mhc.finalize(
            routed.gemm2_permuted,
            routed.expert_weights,
            routed.expanded_idx_to_permuted_idx,
            x.shared_output,
            residual,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            norm_weight,
            norm_eps,
        )
        if stream is None:
            # Outside FULL graphs, compute the next mixes inline.
            stream = torch.cuda.current_stream()
    elif reduce_results:
        if (
            stream is not None
            and _all_reduce_mhc is not None
            and 0 < x.shape[0] <= MHC_OVERLAP_MAX_TOKENS
        ):
            assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
            assert pre_mix is not None and norm_weight is not None
            residual, layer_input = _all_reduce_mhc(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        else:
            x = get_tp_group().all_reduce(x)
    if stream is not None:
        if layer_input is None:
            assert isinstance(x, torch.Tensor)
            residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        aux = (
            residual.mean(dim=1)
            if capture_aux
            else residual.new_empty(0, residual.shape[-1])
        )
        pre_outputs = mhc_pre_delayed_overlap(
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
            stream=stream,
            layer_input=layer_input,
        )
        return residual, *pre_outputs, aux

    assert isinstance(x, torch.Tensor)
    if _split_projection(residual, residual.shape[1] * residual.shape[2], norm_weight):
        # SM80 prefill: the stock post kernel, then the split-k projection
        # and the stock epilogue.
        assert norm_weight is not None
        residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        return residual, *_mhc_pre_split(
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
            norm_weight,
            norm_eps,
            capture_aux,
        )
    if can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux):
        assert pre_mix is not None and norm_weight is not None
        args = (
            x,
            residual,
            pre_mix,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_post_mult_value,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        outputs = None
        if boundary_weight is not None:
            # Decode-size batches: one CuTe DSL kernel, the same TF32 products.
            outputs = _mhc_shifted_post_pre_boundary()(
                *args[:5], boundary_weight, *args[6:]
            )
        if outputs is None:
            outputs = mhc_shifted_post_pre_deep_gemm(*args)
        return *outputs, x.new_empty(0, x.shape[1])

    return mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
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
        capture_aux=capture_aux,
    )


@triton.jit
def _hc_expand_kernel(
    x,
    out,
    numel,
    H: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    row = index // H
    col = index % H
    value = tl.load(x + row * STRIDE_M + col * STRIDE_H, mask=mask, other=0)
    for stream in tl.static_range(N):
        tl.store(out + (row * N + stream) * H + col, value, mask=mask)


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """`x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()`, reading
    `x` once instead of once per copy."""
    m, h = x.shape
    out = x.new_empty(m, hc_mult, h)
    if out.numel():
        _hc_expand_kernel[(triton.cdiv(m * h, 1024),)](
            x, out, m * h, h, hc_mult, *x.stride(), BLOCK=1024, num_warps=4
        )
    return out
