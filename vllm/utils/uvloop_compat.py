# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for uvloop.

uvloop is not available on Windows. Keep call sites using uvloop.run while
falling back to the standard asyncio event loop where uvloop cannot be imported.
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

try:
    import uvloop as _uvloop
except ModuleNotFoundError:
    _uvloop = None


def run(awaitable: Awaitable[Any]) -> Any:
    if _uvloop is not None:
        return _uvloop.run(awaitable)
    return asyncio.run(awaitable)
