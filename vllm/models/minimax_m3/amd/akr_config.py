# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which MiniMax-M3.1 routes run amd_kernels (``kernel_config.amd_kernels``): all by
default on ROCm gfx950 with the package installed."""

from typing import get_args

from vllm.config.kernel import AmdKernelsRoute
from vllm.logger import init_logger

logger = init_logger(__name__)

ROUTES: tuple[str, ...] = get_args(AmdKernelsRoute)
# prerequisites, each listed before the routes that need it
_NEEDS = {
    "indexer": ("attention",),
    "norm": ("gemm", "moe"),
    "sampling": ("draft",),
    "sp_prefill": ("gemm", "moe"),
}
_routes: frozenset[str] | None = None


def routes() -> frozenset[str]:
    """The routes on in this process, read from the current vLLM config at the first
    call under one (model build) and kept."""
    global _routes
    if _routes is not None:
        return _routes
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    akr = None if cfg is None else cfg.kernel_config.amd_kernels
    on: set[str] = set()
    if (akr is None or akr.enabled) and _available():
        on = set(ROUTES) - set(() if akr is None else akr.disabled_routes)
        for r in ROUTES:
            if r in on and not set(_NEEDS.get(r, ())) <= on:
                on.discard(r)
    result = frozenset(on)
    if cfg is not None:
        _routes = result
        logger.info(
            "amd_kernels routes: %s", ", ".join(r for r in ROUTES if r in on) or "none"
        )
    return result


def on(route: str) -> bool:
    """Whether ``route`` (one of ``ROUTES``) runs amd_kernels."""
    return route in routes()


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
