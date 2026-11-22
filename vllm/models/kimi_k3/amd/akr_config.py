# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which Kimi-K3.1 routes run amd_kernels (`kernel_config.amd_kernels`): all by
default on ROCm gfx950 with the package installed.

`VLLM_KIMI_AKR` overrides for A/B runs: `0`/`false` turns every route off, `1`/`true`
(or unset) follows the config, and a comma list (`gemm,moe`) keeps only those routes.
"""

import os
from typing import get_args

import regex as re

from vllm.config.kernel import AmdKernelsRoute
from vllm.logger import init_logger

logger = init_logger(__name__)

ROUTES: tuple[str, ...] = get_args(AmdKernelsRoute)
# prerequisites, each listed before the routes that need it
_NEEDS: dict[str, tuple[str, ...]] = {
    "indexer": ("attention",),
    "sampling": ("draft",),
}
_routes: frozenset[str] | None = None
_warned: set[str] = set()


def _env_routes() -> set[str] | None:
    """Routes allowed by VLLM_KIMI_AKR, or None when it does not restrict."""
    v = os.environ.get("VLLM_KIMI_AKR", "").strip().lower()
    if v in ("", "1", "true", "on", "yes"):
        return None
    if v in ("0", "false", "off", "no", "none"):
        return set()
    return {r.strip() for r in v.split(",") if r.strip()}


def _resolve(akr, available: bool) -> frozenset[str]:
    on: set[str] = set()
    if (akr is None or akr.enabled) and available:
        on = set(ROUTES) - set(() if akr is None else akr.disabled_routes)
        allowed = _env_routes()
        if allowed is not None:
            on &= allowed
        for r in ROUTES:
            if r in on and not set(_NEEDS.get(r, ())) <= on:
                on.discard(r)
    return frozenset(on)


def routes() -> frozenset[str]:
    """The routes on in this process, read from the current vLLM config at the first
    call under one (model build) and kept."""
    global _routes
    if _routes is not None:
        return _routes
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    akr = None if cfg is None else cfg.kernel_config.amd_kernels
    result = _resolve(akr, _available())
    if cfg is not None:
        _routes = result
        logger.info(
            "amd_kernels routes: %s",
            ", ".join(r for r in ROUTES if r in result) or "none",
        )
    return result


def on(route: str) -> bool:
    """Whether `route` (one of `ROUTES`) runs amd_kernels."""
    assert route in ROUTES, route
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


def warn_once(key: str, msg: str, *args) -> None:
    """Log a warning once per process per `key`."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


def unsupported_reason(route: str, reason: str | None) -> bool:
    """Fallback helper: True when the route must keep the stock path because
    `reason` is set (warned once); False when `reason` is None."""
    if reason is None:
        return False
    warn_once(
        f"{route}:{reason}",
        "amd_kernels route %r unsupported (%s); using the stock kernels",
        route,
        reason,
    )
    return True


# the entries' out-of-range reasons: "<op>: <n> rows (the kernel takes 1..<hi>)", ...
_OUT_OF_RANGE = re.compile(
    r"\(the (kernel|communicator|runner) (takes|holds) (\d|<|>|at least)"
)


def call_unsupported(route: str, reason: str | None) -> bool:
    """Per-call fallback: True when an entry's `*_unsupported_reason` keeps this call on
    the stock path. A size outside the kernel's range (more rows or tokens than it
    serves) is an expected case of the call and logged at debug level; any other
    reason is warned once."""
    if reason is None:
        return False
    if _OUT_OF_RANGE.search(reason):
        logger.debug(
            "amd_kernels route %r: stock kernels for this call (%s)", route, reason
        )
        return True
    return unsupported_reason(route, reason)
