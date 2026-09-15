from __future__ import annotations

import asyncio
from contextlib import suppress

from vllm.logger import init_logger

from .disk import SQLiteSessionStore

logger = init_logger(__name__)


class PeriodicDatabaseKeyRotation:
    """Check periodically for a due database-wide key rotation."""

    def __init__(
        self,
        store: SQLiteSessionStore,
        check_interval_seconds: float = 3600.0,
    ) -> None:
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be greater than 0")
        self._store = store
        self._check_interval_seconds = check_interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(
            self._run_periodically(),
            name="responses-store-key-rotation",
        )
        logger.info(
            "Automatic Responses store database-wide key rotation enabled "
            "with a 90-day interval"
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return

        self._stop_event.set()
        await task
        self._task = None

    async def run_once(self, now: int | None = None) -> bool:
        return await self._store.rotate_key_if_due(now)

    async def _run_periodically(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("Automatic Responses store key rotation failed")

            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._check_interval_seconds,
                )


__all__ = ["PeriodicDatabaseKeyRotation"]
