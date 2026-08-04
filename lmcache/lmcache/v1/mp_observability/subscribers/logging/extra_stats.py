# SPDX-License-Identifier: Apache-2.0

"""Opt-in periodic INFO logging of L0<->L1 transfer stats, grouped by GPU.

Enabled via ``--enable-extra-logging``.  Accumulates per-device window and
cumulative stats from ``MP_STORE_*`` / ``MP_RETRIEVE_*`` events and emits a
summary every ``--extra-logging-interval`` seconds.  The 1 Hz
``L1_EVICTION_LOOP_TICK`` event serves as the flush heartbeat, so no timer
thread is needed.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

_PENDING_MAX_AGE_SECONDS = 60.0


@dataclass
class _TransferStats:
    ops: int = 0
    tokens: int = 0
    total_bytes: int = 0
    # Only END events with a correlated START contribute to timed_bytes /
    # copy_time, so the throughput average never uses a partial timing pair.
    timed_bytes: int = 0
    copy_time: float = 0.0


def _format_side(label: str, stats: _TransferStats) -> str:
    if stats.copy_time > 0:
        throughput = f"{stats.timed_bytes / stats.copy_time / 1e9:.2f}GB/s"
    else:
        throughput = "n/a"
    return (
        f"{label} ops={stats.ops} tokens={stats.tokens} "
        f"size={stats.total_bytes / 1e9:.2f}GB avg_copy={throughput}"
    )


class ExtraStatsLoggingSubscriber(EventSubscriber):
    """Logs per-GPU L0<->L1 store/retrieve stats at a fixed interval.

    Emits two INFO lines per active device on each flush: the last-window
    stats and the cumulative totals since start.

    Args:
        interval: Seconds between flushes.  Must be positive.

    Raises:
        ValueError: If ``interval`` is not positive.
    """

    def __init__(self, interval: float) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        self._interval = interval
        self._pending_store: dict[tuple[str, str], float] = {}
        self._pending_retrieve: dict[tuple[str, str], float] = {}
        self._window_store: dict[str, _TransferStats] = {}
        self._window_retrieve: dict[str, _TransferStats] = {}
        self._cum_store: dict[str, _TransferStats] = {}
        self._cum_retrieve: dict[str, _TransferStats] = {}
        self._last_flush = time.monotonic()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: self._on_retrieve_start,
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
            EventType.L1_EVICTION_LOOP_TICK: self._on_tick,
        }

    def _on_store_start(self, event: Event) -> None:
        key = self._correlation_key(event)
        if key is not None:
            self._pending_store[key] = event.timestamp
        self._maybe_flush()

    def _on_store_end(self, event: Event) -> None:
        self._record_end(
            event, self._pending_store, self._window_store, self._cum_store
        )
        self._maybe_flush()

    def _on_retrieve_start(self, event: Event) -> None:
        key = self._correlation_key(event)
        if key is not None:
            self._pending_retrieve[key] = event.timestamp
        self._maybe_flush()

    def _on_retrieve_end(self, event: Event) -> None:
        self._record_end(
            event, self._pending_retrieve, self._window_retrieve, self._cum_retrieve
        )
        self._maybe_flush()

    def _on_tick(self, event: Event) -> None:
        self._maybe_flush()

    @staticmethod
    def _correlation_key(event: Event) -> tuple[str, str] | None:
        device = event.metadata.get("device")
        if not event.session_id or device is None:
            return None
        return (event.session_id, str(device))

    def _record_end(
        self,
        event: Event,
        pending: dict[tuple[str, str], float],
        window: dict[str, _TransferStats],
        cumulative: dict[str, _TransferStats],
    ) -> None:
        key = self._correlation_key(event)
        if key is None:
            return
        device = key[1]
        tokens = int(event.metadata.get("num_tokens", 0))
        total_bytes = int(event.metadata.get("total_bytes", 0))
        t_start = pending.pop(key, None)
        dt = event.timestamp - t_start if t_start is not None else 0.0
        timed = dt > 0 and total_bytes > 0
        for stats in (
            window.setdefault(device, _TransferStats()),
            cumulative.setdefault(device, _TransferStats()),
        ):
            stats.ops += 1
            stats.tokens += tokens
            stats.total_bytes += total_bytes
            if timed:
                stats.timed_bytes += total_bytes
                stats.copy_time += dt

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_flush
        if elapsed < self._interval:
            return
        self._last_flush = now
        self._log_stats(elapsed)
        self._window_store.clear()
        self._window_retrieve.clear()
        self._prune_pending()

    def _log_stats(self, elapsed: float) -> None:
        for device in sorted(set(self._window_store) | set(self._window_retrieve)):
            logger.info(
                "L0<->L1 stats [%s] last %.1fs: %s | %s",
                device,
                elapsed,
                _format_side("store", self._window_store.get(device, _TransferStats())),
                _format_side(
                    "retrieve", self._window_retrieve.get(device, _TransferStats())
                ),
            )
            logger.info(
                "L0<->L1 stats [%s] cumulative: %s | %s",
                device,
                _format_side("store", self._cum_store.get(device, _TransferStats())),
                _format_side(
                    "retrieve", self._cum_retrieve.get(device, _TransferStats())
                ),
            )

    def _prune_pending(self) -> None:
        deadline = time.time() - _PENDING_MAX_AGE_SECONDS
        for pending in (self._pending_store, self._pending_retrieve):
            stale_keys = [k for k, ts in pending.items() if ts < deadline]
            for k in stale_keys:
                del pending[k]
