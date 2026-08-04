# SPDX-License-Identifier: Apache-2.0

"""L1↔L2 throughput metrics subscriber.

Emits two OTel histograms in GB/s, labeled by ``l2_name`` (the registered
adapter type, e.g. ``"fs"``, ``"nixl_store"``):
  - ``lmcache_mp.l2_store_throughput``  — L1→L2 store
  - ``lmcache_mp.l2_load_throughput``   — L2→L1 load

Implementation:
  - Store path correlates ``L2_STORE_SUBMITTED`` → ``L2_STORE_COMPLETED``
    by the compound key ``(adapter_index, task_id)``.
  - Load path correlates ``L2_LOAD_TASK_SUBMITTED`` → ``L2_LOAD_TASK_COMPLETED``
    by ``(request_id, adapter_index)``.  One prefetch request may fan out
    across multiple adapters, so per-adapter correlation is required to
    attribute throughput to the right ``l2_name``.
  - ``total_bytes`` is read from the SUBMITTED event and cached in the
    subscriber's pending dict alongside the start timestamp, so the
    COMPLETED event does not need to carry it.  This keeps the byte
    accounting out of the controllers' in-flight task state.
  - ``(end_ts - start_ts)`` spans submit -> complete and therefore
    includes adapter queue, network, and disk time — not just transfer.
    The histogram is "bytes / end-to-end latency", not raw transfer rate.
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


class L2ThroughputSubscriber(EventSubscriber):
    """Records L1↔L2 throughput by correlating SUBMITTED→COMPLETED pairs."""

    def __init__(self) -> None:
        # (adapter_index, task_id) -> (t_start, total_bytes).
        self._pending_store: dict[tuple[int, int], tuple[float, int]] = {}
        # (request_id, adapter_index) -> (t_start, total_bytes).
        self._pending_load: dict[tuple[int, int], tuple[float, int]] = {}

        meter = metrics.get_meter("lmcache_mp.perf")
        self._store_hist = meter.create_histogram(
            "lmcache_mp.l2_store_throughput",
            description=(
                "Histogram of L1->L2 store throughput in GB/s, measured "
                "per task as total_bytes / (completed_ts - "
                "submitted_ts).  Spans adapter queue + network/disk I/O, "
                "so this is end-to-end latency-based throughput."
            ),
            unit="GB/s",
        )
        self._load_hist = meter.create_histogram(
            "lmcache_mp.l2_load_throughput",
            description=(
                "Histogram of L2->L1 load throughput in GB/s, measured "
                "per (request, adapter) pair as total_bytes / "
                "(completed_ts - submitted_ts).  Spans adapter queue + "
                "network/disk I/O."
            ),
            unit="GB/s",
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L2_STORE_SUBMITTED: self._on_store_submitted,
            EventType.L2_STORE_COMPLETED: self._on_store_completed,
            EventType.L2_LOAD_TASK_SUBMITTED: self._on_load_submitted,
            EventType.L2_LOAD_TASK_COMPLETED: self._on_load_completed,
        }

    # -- Store path (L1->L2) -----------------------------------------------

    def _on_store_submitted(self, event: Event) -> None:
        key = self._store_key(event)
        if key is not None:
            total_bytes = int(event.metadata.get("total_bytes", 0))
            self._pending_store[key] = (event.timestamp, total_bytes)

    def _on_store_completed(self, event: Event) -> None:
        key = self._store_key(event)
        if key is None:
            return

        # ``bytes_transferred`` is always present in L2_STORE_COMPLETED
        # events (populated from L2StoreResult).
        bytes_transferred = event.metadata.get("bytes_transferred")
        self._record(
            event=event,
            correlation_key=key,
            pending=self._pending_store,
            hist=self._store_hist,
            real_bytes_transferred=bytes_transferred,
        )

    # -- Load path (L2->L1) ------------------------------------------------

    def _on_load_submitted(self, event: Event) -> None:
        key = self._load_key(event)
        if key is not None:
            total_bytes = int(event.metadata.get("total_bytes", 0))
            self._pending_load[key] = (event.timestamp, total_bytes)

    def _on_load_completed(self, event: Event) -> None:
        key = self._load_key(event)
        if key is None:
            return
        self._record(
            event=event,
            correlation_key=key,
            pending=self._pending_load,
            hist=self._load_hist,
        )

    # -- Correlation-key helpers ------------------------------------------

    @staticmethod
    def _store_key(event: Event) -> tuple[int, int] | None:
        """Build the ``(adapter_index, task_id)`` correlation key.

        Returns ``None`` if either field is missing.
        """
        adapter_index = event.metadata.get("adapter_index")
        task_id = event.metadata.get("task_id")
        if adapter_index is None or task_id is None:
            return None
        return (int(adapter_index), int(task_id))

    @staticmethod
    def _load_key(event: Event) -> tuple[int, int] | None:
        """Build the ``(request_id, adapter_index)`` correlation key.

        Returns ``None`` if either field is missing.
        """
        request_id = event.metadata.get("request_id")
        adapter_index = event.metadata.get("adapter_index")
        if request_id is None or adapter_index is None:
            return None
        return (int(request_id), int(adapter_index))

    # -- Core computation --------------------------------------------------

    @staticmethod
    def _record(
        event: Event,
        correlation_key: tuple[int, int],
        pending: dict[tuple[int, int], tuple[float, int]],
        hist: Any,
        real_bytes_transferred: int | None = None,
    ) -> None:
        """
        Record a throughput sample.

        Args:
            event: The COMPLETED event, used for timestamp and metadata.
            correlation_key: The key to look up the matching SUBMITTED event.
            pending: The dict mapping correlation keys to (t_start, total_bytes).
            hist: The OTel histogram to record into.
            real_bytes_transferred: If set, use this instead of the submitted
                totay_bytes, When it's 0, drop the sample.
        """
        pending_entry = pending.pop(correlation_key, None)
        if pending_entry is None:
            return  # no matching SUBMITTED event;
        t_start, total_bytes = pending_entry

        effective_bytes = (
            total_bytes if real_bytes_transferred is None else real_bytes_transferred
        )
        if effective_bytes <= 0:
            return

        dt = event.timestamp - t_start
        if dt <= 0:
            return

        l2_name = event.metadata.get("l2_name")
        attrs: dict[str, Any] = {}
        if l2_name is not None:
            attrs["l2_name"] = str(l2_name)

        hist.record(effective_bytes / dt / 1e9, attributes=attrs)
