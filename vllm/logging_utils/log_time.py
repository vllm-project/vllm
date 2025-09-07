# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Provides a timeslice logging decorator
"""

import functools
import time


def logtime(logger):
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

            logger.debug("Function '%s.%s' took %.7f secs", func.__module__,
                         func.__qualname__, elapsed)
            return result

        return _wrapper

    return _inner
