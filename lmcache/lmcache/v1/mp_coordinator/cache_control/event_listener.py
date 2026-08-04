# SPDX-License-Identifier: Apache-2.0
"""MP-server-side L2 event client: batches adapter store/lookup/delete
events and flushes them to the coordinator on a timer."""

# Standard
import asyncio
import threading

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.mp_coordinator.schemas import (
    EventType,
    ReportUsageRequest,
    ReportUsageResponse,
    UsageEvent,
)

logger = init_logger(__name__)

_DEFAULT_FLUSH_INTERVAL = 1.0


class L2EventListener(L2AdapterListener):
    """L2 adapter listener that batches events and flushes to the
    coordinator. Run :meth:`run` as a background task.

    Args:
        client: HTTP client.
        coordinator_url: Coordinator base URL.
        instance_id: This MP server's id (sent with every batch).
        flush_interval: Seconds between flushes.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        coordinator_url: str,
        instance_id: str,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        self._client = client
        self._base_url = coordinator_url.rstrip("/")
        self._instance_id = instance_id
        self._flush_interval = flush_interval
        self._seq = 0
        self._lock = threading.Lock()
        self._buffer: list[UsageEvent] = []

    # -- L2AdapterListener implementation ------------------------------------

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        """Buffer store events for each key. Thread-safe."""
        for obj, size in zip(keys, sizes, strict=True):
            event = UsageEvent(
                type=EventType.STORE,
                key=obj.to_encoded_object_key(),
                bytes=size,
            )
            with self._lock:
                self._buffer.append(event)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        """Buffer lookup events for each key. Thread-safe."""
        for obj in keys:
            event = UsageEvent(
                type=EventType.LOOKUP,
                key=obj.to_encoded_object_key(),
                bytes=0,
            )
            with self._lock:
                self._buffer.append(event)

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        """Buffer DELETE events for each key. Thread-safe."""
        for obj in keys:
            event = UsageEvent(
                type=EventType.DELETE,
                key=obj.to_encoded_object_key(),
                bytes=0,
            )
            with self._lock:
                self._buffer.append(event)

    # -- Flush loop ----------------------------------------------------------

    async def run(self) -> None:
        """Drain the buffer on a timer until cancelled. Flush failures
        drop the batch to bound buffer growth when the coordinator is
        down."""
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered events to the coordinator."""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []
            self._seq += 1
            seq = self._seq

        body = ReportUsageRequest(
            instance_id=self._instance_id,
            seq=seq,
            events=batch,
        )
        try:
            resp = await self._client.post(
                f"{self._base_url}/quota/events",
                json=body.model_dump(),
            )
            resp.raise_for_status()
            result = ReportUsageResponse.model_validate(resp.json())
            logger.debug("Flushed %d L2 events to coordinator", result.recorded)
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "Failed to flush %d L2 events to coordinator: %s",
                len(batch),
                e,
            )
