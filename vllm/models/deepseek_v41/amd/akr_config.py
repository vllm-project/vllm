# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which DeepSeek-V4.1-Flash routes run `amd_kernels` (`kernel_config.amd_kernels`).

A route is on when `amd_kernels` imports on ROCm gfx950, the route is not in
`disabled_routes` and its prerequisite routes are on. Per call, `serves(route, m)`
also needs `m` inside the route's row range (`GATES`), where `m` is the static
(captured, padded) token count of the call, so a captured shape takes one branch.
A collective route (`fused_ep8`) passes `dp_rows(m)`, the DP-group maximum, so
that every rank takes the same branch. The `amd_kernels` op entries stay the
authority on shapes: each call also asks the entry, and a declined call runs the
stock kernel.
"""

from collections.abc import Callable
from typing import get_args

from vllm.config.kernel import DeepseekV41AmdKernelsRoute
from vllm.logger import init_logger

logger = init_logger(__name__)

ROUTES: tuple[str, ...] = get_args(DeepseekV41AmdKernelsRoute)

# prerequisites, each listed before the routes that need it
_NEEDS: dict[str, tuple[str, ...]] = {
    # the fused wo_b epilogue reads wo_a's MXFP8 output and replaces the FFN seam
    "mhc_wo_b": ("mhc_seam", "wo_a"),
    # the fused shared-expert down epilogue replaces the next attention seam
    "mhc_ffn": ("mhc_seam", "shared_expert"),
}


def _rows(lo: int, hi: int) -> Callable[[int], bool]:
    return lambda m: lo <= m <= hi


# Row ranges where the kernel beat the stock path on MI355X (per-kernel
# measurements against the served kernels, graph replay).
GATES: dict[str, Callable[[int], bool]] = {
    "rope_insert": _rows(1, 1024),
    "norm_dual": _rows(1, 1024),
    "compressor": _rows(1, 768),
    "kv_score": _rows(1, 768),
    "dense_gemm": _rows(1, 768),
    "indexer": _rows(1, 768),
    "sparse_mla": _rows(1, 768),
    "wo_a": _rows(1, 1024),
    "router": _rows(1, 768),
    "shared_expert": _rows(1, 768),
    "fused_ep8": _rows(1, 1024),
    "mhc_seam": _rows(1, 384),
    "mhc_tail": _rows(1, 1024),
    "mhc_wo_b": _rows(1, 128),
    "mhc_ffn": _rows(1, 192),
    # vs vLLM's lm_head dispatch: wins 9-13% at 113-128, 3-4% at 496-512 rows
    "lm_head": lambda m: 113 <= m <= 128 or 496 <= m <= 512,
    "main_proj": _rows(640, 1024),
    "draft_router": _rows(1, 1024),
    "markov": _rows(1, 160),
}

# The mHC seam variant that also emits the mean of the new streams (the DSpark
# auxiliary hidden state) replaces two launches and wins at any row count.
MHC_SEAM_AUX_MAX_ROWS = 4096
# The indexer handles rows up to 2^20 compressed entries; rows longer than
# 262144 take its long path, which loses to AITER below 128 decode rows on the
# dense (non-candidate) layers.
INDEXER_MAX_ENTRIES = 1 << 20
INDEXER_SHORT_ENTRIES = 262144
INDEXER_LONG_MIN_ROWS = 128

_routes: frozenset[str] | None = None
_failed: dict[str, str] = {}


def resolve_routes(
    enabled: bool, disabled: set[str], available: bool
) -> frozenset[str]:
    if not enabled or not available:
        return frozenset()
    on = set(ROUTES) - disabled
    for r in ROUTES:
        if r in on and not set(_NEEDS.get(r, ())) <= on:
            on.discard(r)
    return frozenset(on)


def routes() -> frozenset[str]:
    """The routes on in this process, read from the current vLLM config at the
    first call under one (model build) and kept."""
    global _routes
    if _routes is not None:
        return _routes
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    akr = None if cfg is None else cfg.kernel_config.amd_kernels
    result = resolve_routes(
        akr is None or akr.enabled,
        set(() if akr is None else akr.disabled_routes),
        _available(),
    )
    if cfg is not None:
        _routes = result
        logger.info(
            "amd_kernels routes: %s",
            ", ".join(r for r in ROUTES if r in result) or "none",
        )
    return result


def on(route: str) -> bool:
    """Whether `route` is on in this process (any row count)."""
    return route in routes() and route not in _failed


def serves(route: str, m: int) -> bool:
    """Whether `route` is on and `m` rows are inside its row range."""
    return on(route) and GATES[route](m)


def fail(route: str, reason: str) -> None:
    """Take `route` out for the rest of the process. Only called at load time,
    before CUDA graph capture fixes the branches."""
    if route not in _failed:
        _failed[route] = reason
        logger.warning("amd_kernels route %s disabled: %s", route, reason)


def failed() -> dict[str, str]:
    return dict(_failed)


def reset() -> None:
    global _routes
    _routes = None
    _failed.clear()


def dp_rows(m: int) -> int:
    """The DP-group maximum token count of this step (`m` outside a forward
    context or without DP)."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if is_forward_context_available():
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            return max(m, int(dp_metadata.num_tokens_across_dp_cpu.max()))
    return m


def indexer_bound(max_model_len: int, compress_ratio: int, override: int = 0) -> int:
    """Longest row of an indexer layer in compressed entries."""
    return override or -(-max_model_len // max(compress_ratio, 1))


def indexer_rows_ok(bound: int, rows: int, candidate_consumer: bool) -> bool:
    """Candidate consumers always run the kernel; dense layers with a long bound
    only from `INDEXER_LONG_MIN_ROWS` decode rows."""
    return (
        candidate_consumer
        or bound <= INDEXER_SHORT_ENTRIES
        or rows >= INDEXER_LONG_MIN_ROWS
    )


def takes_mxfp8_codes(linear) -> bool:
    """Whether a stock linear reads row-major MXFP8 codes and E8M0 scales as
    they are (FlashInfer kernels use the same key with swizzled scales)."""
    from vllm.model_executor.kernels.linear.mxfp8.aiter_flydsl import (
        AiterFlyDSLMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    kernel = getattr(linear.quant_method, "kernel", None)
    supports = getattr(kernel, "supports_quantized_input", None)
    return (
        get_input_quant_key(linear) == kMxfp8Dynamic
        and isinstance(
            kernel, (RocmDotScaledMxfp8LinearKernel, AiterFlyDSLMxfp8LinearKernel)
        )
        and (supports is None or bool(supports(linear)))
    )


def quant_activation(codes, scale, orig_dtype):
    """MXFP8 codes and row-major E8M0 scales as the activation stock linears take."""
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    return QuantizedActivation(codes, scale, orig_dtype, codes.shape, kMxfp8Dynamic)


def _available() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return False
    try:
        import amd_kernels  # noqa: F401
    except ImportError:
        return False
    return True
