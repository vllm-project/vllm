# SPDX-License-Identifier: Apache-2.0

"""L0↔L1 throughput metrics subscriber.

Emits two OTel histograms in GB/s, labeled by ``engine_id``, ``device``,
and ``model_name``:
  - ``lmcache_mp.l0_l1_store_throughput``  — GPU→CPU (L0→L1) store
  - ``lmcache_mp.l0_l1_load_throughput``   — CPU→GPU (L1→L0) load

Implementation:
  - Correlates ``MP_STORE_START`` → ``MP_STORE_END`` and
    ``MP_RETRIEVE_START`` → ``MP_RETRIEVE_END`` pairs by the compound key
    ``(session_id, device)``.  ``session_id`` alone is not sufficient
    because one MP server process serves multiple vLLM workers, so TP/PP
    replicas of the same request fire concurrent START/END pairs on
    different GPUs.
  - START/END events fire on the GPU cupy stream (``publish_on_stream``),
    so their timestamps reflect true GPU-stream time for the D2H/H2D
    copies — not Python/lock overhead.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class L0L1ThroughputSubscriber(EventSubscriber):
    """Records L0↔L1 throughput by correlating MP_*_START→MP_*_END pairs."""

    def __init__(self) -> None:
        # (session_id, device) -> t_start.
        # Compound key avoids collisions when one MP server
        # handles the same request_id from multiple GPUs (TP/PP).
        self._pending_store: dict[tuple[str, str], float] = {}
        self._pending_load: dict[tuple[str, str], float] = {}

        meter = metrics.get_meter("lmcache_mp.perf")
        self._store_hist = meter.create_histogram(
            "lmcache_mp.l0_l1_store_throughput",
            description=(
                "Histogram of L0→L1 (GPU→CPU) store throughput in GB/s, "
                "measured per request as total_bytes / (end_ts - start_ts) "
                "on the GPU cupy stream."
            ),
            unit="GB/s",
        )
        self._load_hist = meter.create_histogram(
            "lmcache_mp.l0_l1_load_throughput",
            description=(
                "Histogram of L1→L0 (CPU→GPU) load throughput in GB/s, "
                "measured per request as total_bytes / (end_ts - start_ts) "
                "on the GPU cupy stream."
            ),
            unit="GB/s",
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: self._on_retrieve_start,
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
        }

    # -- Store path (L0→L1, GPU→CPU) ---------------------------------------

    def _on_store_start(self, event: Event) -> None:
        key = self._correlation_key(event)
        if key is not None:
            self._pending_store[key] = event.timestamp

    def _on_store_end(self, event: Event) -> None:
        self._record(
            event=event,
            pending=self._pending_store,
            hist=self._store_hist,
        )

    # -- Retrieve path (L1→L0, CPU→GPU) ------------------------------------

    def _on_retrieve_start(self, event: Event) -> None:
        key = self._correlation_key(event)
        if key is not None:
            self._pending_load[key] = event.timestamp

    def _on_retrieve_end(self, event: Event) -> None:
        self._record(
            event=event,
            pending=self._pending_load,
            hist=self._load_hist,
        )

    # -- Core computation --------------------------------------------------

    @staticmethod
    def _correlation_key(event: Event) -> tuple[str, str] | None:
        """Build the ``(session_id, device)`` correlation key.

        Returns ``None`` if either field is missing — such events cannot
        be paired safely and are dropped.
        """
        device = event.metadata.get("device")
        if not event.session_id or device is None:
            return None
        return (event.session_id, str(device))

    @classmethod
    def _record(
        cls,
        event: Event,
        pending: dict[tuple[str, str], float],
        hist: Any,
    ) -> None:
        key = cls._correlation_key(event)
        if key is None:
            return
        t_start = pending.pop(key, None)
        if t_start is None:
            return  # No matching START event

        total_bytes = event.metadata.get("total_bytes", 0)
        if total_bytes <= 0:
            return

        dt = event.timestamp - t_start
        if dt <= 0:
            return

        engine_id = event.metadata.get("engine_id")
        model_name = event.metadata.get("model_name")
        attrs: dict[str, Any] = {"device": key[1]}
        if engine_id is not None:
            attrs["engine_id"] = str(engine_id)
        if model_name is not None:
            attrs["model_name"] = str(model_name)

        hist.record(total_bytes / dt / 1e9, attributes=attrs)
