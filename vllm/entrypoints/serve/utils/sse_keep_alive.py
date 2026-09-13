# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSE keep-alive comments for idle streaming responses.

Reverse proxies and tunnels (Cloudflare Tunnel, NGINX ``proxy_read_timeout``,
AWS ALB, ...) can close a streaming connection when no bytes are sent for a
while. That happens while a request is queued, during prefill, or in gaps
between tokens. ``with_sse_keep_alive`` wraps the final SSE generator and emits
a comment line (``": keep-alive\\n\\n"``) whenever it has been idle for
``interval`` seconds. Comments are ignored by SSE-compliant clients but still
count as bytes for proxy read-timeout logic.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator

SSE_KEEP_ALIVE_COMMENT = ": keep-alive\n\n"


def with_sse_keep_alive(
    generator: AsyncGenerator[str, None],
    interval: float,
) -> AsyncGenerator[str, None]:
    """Emit an SSE keep-alive comment when ``generator`` is idle.

    A non-positive or non-finite ``interval`` returns ``generator`` unchanged,
    so the default path has no overhead. Otherwise a keep-alive comment is
    yielded whenever no chunk arrives within ``interval`` seconds.
    """
    if not 0 < interval < float("inf"):
        return generator
    return _keep_alive_stream(generator, interval)


async def _keep_alive_stream(
    generator: AsyncGenerator[str, None],
    interval: float,
) -> AsyncGenerator[str, None]:
    # A single retained ``anext()`` task advances the generator; asyncio forbids
    # concurrent ``__anext__`` calls, and ``wait_for`` would cancel the task
    # mid-advance on timeout. ``asyncio.wait`` observes both outcomes without
    # cancelling the pending advance.
    next_task: asyncio.Task[str] = asyncio.create_task(anext(generator))
    try:
        while True:
            done, _ = await asyncio.wait({next_task}, timeout=interval)
            if not done:
                yield SSE_KEEP_ALIVE_COMMENT
                continue
            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                return
            yield chunk
            next_task = asyncio.create_task(anext(generator))
    finally:
        next_task.cancel()
        # The task outcome was already handled in the loop; suppress so cleanup
        # never corrupts it (matching ``merge_async_iterators``).
        with contextlib.suppress(BaseException):
            await next_task
        # Close the upstream generator so its cleanup runs deterministically.
        with contextlib.suppress(BaseException):
            await generator.aclose()
