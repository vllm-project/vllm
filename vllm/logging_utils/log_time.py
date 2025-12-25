# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Provides a timeslice logging decorator
"""

import functools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable)


def logtime(logger, msg: str | None = None) -> Callable[[T], T]:
    """
    Logs the execution time of the decorated function.
    Always place it beneath other decorators.
    """

    def _inner(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            prefix = (
                f"Function '{func.__module__}.{func.__qualname__}'"
                if msg is None
                else msg
            )
            logger.debug("%s: Elapsed time %.7f secs", prefix, elapsed)
            return result

        return _wrapper

    return _inner
