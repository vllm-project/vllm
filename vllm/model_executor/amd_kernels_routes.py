# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which `amd_kernels` routes run in this process (`kernel_config.amd_kernels`).

`amd_kernels` (https://github.com/Inferact/amd-kernels) holds FlyDSL kernels for
ROCm gfx950. A route replaces one stock kernel, or a chain of them, with an
`amd_kernels` op. A model declares its routes once:

    ROUTES = AmdKernelsRoutes("dsv41", ("wo_a", "wo_b"), needs={"wo_b": ("wo_a",)})

and asks `ROUTES.on("wo_a")` where it would call the op. Whether the op serves a
given call (shapes, dtypes, the row counts where it beats the stock path) is the
library's answer (`<op>_blocker`, `prepare_<op>`); a route only holds vLLM's
side: the user's switches, the platform, and prerequisites between routes.
Everything is off unless the package imports on ROCm gfx950.
"""

import functools
from collections.abc import Iterable, Mapping

from vllm.logger import init_logger

logger = init_logger(__name__)

_REGISTRY: dict[str, "AmdKernelsRoutes"] = {}


@functools.cache
def available() -> bool:
    """ROCm gfx950 with `amd_kernels` importable."""
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


class AmdKernelsRoutes:
    """One model's routes: their names and the prerequisites between them."""

    def __init__(
        self,
        model: str,
        routes: Iterable[str],
        needs: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.model = model
        self.names: tuple[str, ...] = tuple(routes)
        self.needs = {r: tuple(n) for r, n in (needs or {}).items()}
        unknown = {x for r, n in self.needs.items() for x in (r, *n)} - set(self.names)
        if unknown:
            raise ValueError(f"{model}: prerequisites name unknown routes {unknown}")
        self._on: frozenset[str] | None = None
        self._failed: dict[str, str] = {}
        _REGISTRY[model] = self

    def resolve(
        self, enabled: bool, disabled: Iterable[str], available: bool
    ) -> frozenset[str]:
        """The routes on for these switches: all but the disabled ones and the
        ones whose prerequisite is off."""
        if not enabled or not available:
            return frozenset()
        on = set(self.names) - set(disabled)
        changed = True
        while changed:
            changed = False
            for route in list(on):
                if not set(self.needs.get(route, ())) <= on:
                    on.discard(route)
                    changed = True
        return frozenset(on)

    def routes(self) -> frozenset[str]:
        """The routes on in this process, read from the current vLLM config at
        the first call under one (model construction) and kept."""
        if self._on is not None:
            return self._on
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        akr = None if cfg is None else cfg.kernel_config.amd_kernels
        disabled = () if akr is None else akr.disabled_routes
        on = self.resolve(akr is None or akr.enabled, disabled, available())
        if cfg is not None:
            self._on = on
            _warn_unknown(disabled)
            logger.info(
                "amd_kernels routes (%s): %s",
                self.model,
                ", ".join(r for r in self.names if r in on) or "none",
            )
        return on

    def on(self, route: str) -> bool:
        """Whether `route` runs `amd_kernels` in this process."""
        assert route in self.names, f"{self.model} declares no route {route!r}"
        return route in self.routes() and not self._failed_with_needs(route)

    def fail(self, route: str, reason: str) -> None:
        """Take `route`, and the routes that need it, out for the rest of the
        process. Only at load time: a captured CUDA graph keeps its branch."""
        if route not in self._failed:
            self._failed[route] = reason
            logger.warning("amd_kernels route %s disabled: %s", route, reason)

    def failed(self) -> dict[str, str]:
        return dict(self._failed)

    def reset(self) -> None:
        """Forget the resolved routes and failures (tests)."""
        self._on = None
        self._failed.clear()

    def _failed_with_needs(self, route: str) -> bool:
        return route in self._failed or any(
            self._failed_with_needs(n) for n in self.needs.get(route, ())
        )


def covered(op: str, reason: str | None) -> bool:
    """True when an `amd_kernels` blocker returned None; else logs the reason
    once and the caller runs its stock path."""
    if reason is None:
        return True
    logger.info_once("amd_kernels %s: stock path (%s)", op, reason)
    return False


def dp_max_tokens(num_tokens: int) -> int:
    """The DP group's largest token count of this step (`num_tokens` outside a
    forward context or without DP): what a collective route decides on, so
    that every rank takes the same branch."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if is_forward_context_available():
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            return max(num_tokens, int(dp_metadata.num_tokens_across_dp_cpu.max()))
    return num_tokens


@functools.cache
def _warn_unknown_once(names: tuple[str, ...]) -> None:
    logger.warning(
        "amd_kernels: disabled_routes %s are not routes of any loaded model (%s)",
        list(names),
        ", ".join(f"{m}: {', '.join(r.names)}" for m, r in sorted(_REGISTRY.items())),
    )


def _warn_unknown(disabled: Iterable[str]) -> None:
    known = {n for routes in _REGISTRY.values() for n in routes.names}
    unknown = tuple(sorted(set(disabled) - known))
    if unknown:
        _warn_unknown_once(unknown)
