# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic idle sleep for the OpenAI-compatible API server.

When ``--sleep-idle-ttl`` is set (requires ``--enable-sleep-mode``), the
server puts the engine to sleep after the configured idle period and
wakes it up when the next inference request arrives.
"""

import asyncio
import time

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


class IdleSleepManager:
    """Puts the engine to sleep when idle and wakes it on demand.

    Transitions are serialized by ``_lock``: the idle loop holds it while
    putting the engine to sleep, and an inference request arriving during
    or after the transition waits on it, then triggers wake-up. In-flight
    accounting is maintained by :class:`IdleSleepMiddleware` via
    :meth:`on_request_start` / :meth:`on_request_end`.

    Only sleeps initiated by this manager are auto-woken; an engine put to
    sleep through the dev ``/sleep`` endpoint is left alone so existing
    RLHF-style orchestration keeps its semantics.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        ttl: float,
        level: int = 1,
    ) -> None:
        self.engine_client = engine_client
        self.ttl = ttl
        self.level = level
        self.in_flight = 0
        self.auto_slept = False
        self._last_activity = time.monotonic()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._poll_interval = min(max(ttl / 10, 0.01), 5.0)

    def start(self) -> None:
        """Start the idle monitoring loop on the running event loop."""
        self._task = asyncio.create_task(self._idle_loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def on_request_start(self) -> None:
        """Register an inference request, waking the engine if needed.

        Waits on any in-progress sleep transition, so a request can never
        be forwarded to an engine this manager is putting to sleep.
        """
        async with self._lock:
            if self.auto_slept:
                await self._wake()
                self.auto_slept = False
            self.in_flight += 1
            self._touch()

    def on_request_end(self) -> None:
        self.in_flight -= 1
        self._touch()

    async def _wake(self) -> None:
        logger.info("Idle sleep: waking up engine (level %d)", self.level)
        start = time.monotonic()
        if self.level >= 2:
            # Level 2 discarded the weights; wake them first, reload
            # in-place from the model source, then restore the KV cache
            # (mirrors the documented level-2 wake-up sequence).
            await self.engine_client.wake_up(tags=["weights"])
            await self.engine_client.collective_rpc("reload_weights")
            await self.engine_client.wake_up(tags=["kv_cache"])
        else:
            await self.engine_client.wake_up()
        logger.info("Idle sleep: engine awake in %.2fs", time.monotonic() - start)

    async def _idle_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._idle_step()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Idle sleep: transition failed")
                # Avoid a hot retry loop; wait a full TTL before retrying.
                self._touch()

    async def _idle_step(self) -> None:
        if self.auto_slept:
            # Resync in case something woke the engine directly (e.g. the
            # dev /wake_up endpoint) without going through this manager.
            if not await self.engine_client.is_sleeping():
                self.auto_slept = False
                self._touch()
            return
        if self.in_flight > 0 or self.idle_seconds < self.ttl:
            return
        async with self._lock:
            # Re-check under the lock; a request may have arrived while
            # we were waiting for it.
            if self.auto_slept or self.in_flight > 0 or self.idle_seconds < self.ttl:
                return
            if await self.engine_client.is_sleeping():
                # Slept externally; leave the wake decision to whoever
                # initiated it.
                return
            logger.info(
                "Idle sleep: no inference requests for %.0fs, "
                "putting engine to sleep (level %d)",
                self.idle_seconds,
                self.level,
            )
            await self.engine_client.sleep(level=self.level, mode="wait")
            self.auto_slept = True
