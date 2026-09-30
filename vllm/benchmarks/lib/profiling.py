# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any


@contextmanager
def profile(llm: Any, enabled: bool) -> Iterator[None]:
    """Flush the configured profiler even when a benchmark request fails."""
    if enabled:
        llm.start_profile()
    try:
        yield
    finally:
        if enabled:
            llm.stop_profile()


@asynccontextmanager
async def profile_async(llm: Any, enabled: bool) -> AsyncIterator[None]:
    """Flush the configured async engine profiler on completion or failure."""
    if enabled:
        await llm.start_profile()
    try:
        yield
    finally:
        if enabled:
            await llm.stop_profile()
