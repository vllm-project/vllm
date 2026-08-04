# SPDX-License-Identifier: Apache-2.0
"""No-throw guarantee for usage telemetry.

Every usage-telemetry entry point that serving code calls (engine startup,
MP server startup, KV-cache registration, the stats-logger loop) is wrapped
with :func:`swallow_telemetry_errors`, so a bug or environment failure in
telemetry can never propagate into the caching or serving path.
"""

# Future
from __future__ import annotations

# Standard
from typing import Callable, ParamSpec, TypeVar
import functools

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def swallow_telemetry_errors(func: Callable[P, R]) -> Callable[P, R | None]:
    """Wrap *func* so that every exception is logged at debug and swallowed.

    Args:
        func: The telemetry entry point to protect.

    Returns:
        A wrapper with the same signature that returns ``None`` instead of
        raising when *func* fails.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug(
                "Usage telemetry failure in %s (suppressed)",
                func.__qualname__,
                exc_info=True,
            )
            return None

    return wrapper
