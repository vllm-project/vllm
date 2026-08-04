# SPDX-License-Identifier: Apache-2.0

"""StorageManager lifecycle subscriber — workload-level real-reuse histograms.

Tracks the gap between consecutive uses of the same logical chunk
(``(cache_salt, chunk_hash)``) at the StorageManager level — i.e.
caller-driven reads/writes, not internal lock releases from the
store/prefetch controllers.  Two histograms are emitted, both tagged
with ``cache_salt``:

- ``lmcache_mp.real_reuse_gap_seconds`` — wall-clock gap between a
  chunk's last access (read or write) and the next read.  Captures
  storage cost.
- ``lmcache_mp.real_reuse_gap_chunks`` — same gap measured in the
  per-salt access-counter stream (bumped on every read AND write of
  every chunk, regardless of sampling).  Captures storage volume.

Sampling: only chunks that pass a deterministic ``hash % prime``
gate get a tracking entry.  The per-salt counter advances on every
access, sampled or not, so the chunks-gap reflects true volume.
"""

# Future
from __future__ import annotations

# Standard
import random
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

# Cap on the (cache_salt, chunk_hash) reuse-tracking dict.  Random
# eviction on overflow — all entries already passed the sampling gate.
_REUSE_TRACK_CAP = 100_000


class SMLifecycleSubscriber(EventSubscriber):
    """Real-reuse histograms over StorageManager events.

    Histograms (tagged with ``cache_salt``):

    - ``lmcache_mp.real_reuse_gap_seconds`` — gap between a chunk's
      last access (read or write) and the next read.  Storage cost.
      Emitted on read events only.
    - ``lmcache_mp.real_reuse_gap_chunks`` — same gap in the per-salt
      access-counter stream (bumped on every read AND write).  Storage
      volume.  Emitted on read events for sampled chunks.

    Sources are caller-facing StorageManager events
    (``SM_READ_PREFETCHED_FINISHED``, ``SM_WRITE_FINISHED``) so
    internal lock releases from the store and prefetch controllers do
    not pollute the signal.

    Parameters:
        sample_rate: Fraction of chunks to track (0, 1.0].  Default 0.01 (1%).
    """

    def __init__(self, sample_rate: float = 0.01) -> None:
        if not 0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1.0], got {sample_rate}")
        self._sample_rate = sample_rate
        self._sample_prime = 1_000_003
        self._sample_threshold = int(sample_rate * self._sample_prime)
        meter = metrics.get_meter("lmcache.sm")
        self._real_reuse_gap_seconds_hist = meter.create_histogram(
            "lmcache_mp.real_reuse_gap",
            description=(
                "Gap between a chunk's last access (read or write) and "
                "next read.  Storage cost.  Tagged with cache_salt."
            ),
            unit="s",
        )
        self._real_reuse_gap_chunks_hist = meter.create_histogram(
            "lmcache_mp.real_reuse_gap_objects",
            description=(
                "Per-cache_salt access-counter gap between two reads of "
                "the same chunk.  Storage volume.  Tagged with cache_salt."
            ),
            unit="{chunks}",
        )
        # Per-salt access counter; bumped on every read and write.
        # cache_salt is operator-set (tenant identifier), so cardinality
        # is naturally bounded.
        self._salt_chunk_counter: dict[str, int] = {}
        # (cache_salt, chunk_hash) -> (last_access_time, counter_at_last_access).
        self._reuse_track: dict[tuple[str, bytes], tuple[float, int]] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.SM_READ_PREFETCHED_FINISHED: self._on_sm_read_finished,
            EventType.SM_WRITE_FINISHED: self._on_sm_write_finished,
        }

    def _bump_salt_counter(self, salt: str) -> int:
        """Advance and return the per-salt access counter."""
        nxt = self._salt_chunk_counter.get(salt, 0) + 1
        self._salt_chunk_counter[salt] = nxt
        return nxt

    def _admit_track(
        self,
        track_key: tuple[str, bytes],
        when: float,
        counter: int,
    ) -> None:
        """Admit a newly-sampled entry; random-evict on overflow."""
        if (
            len(self._reuse_track) >= _REUSE_TRACK_CAP
            and track_key not in self._reuse_track
        ):
            victim = random.choice(list(self._reuse_track))
            self._reuse_track.pop(victim, None)
        self._reuse_track[track_key] = (when, counter)

    def _should_sample(self, key: tuple[str, bytes]) -> bool:
        return hash(key) % self._sample_prime < self._sample_threshold

    def _on_sm_read_finished(self, event: Event) -> None:
        """Bump counter and emit a real-reuse sample for sampled chunks."""
        now = event.timestamp or time.time()
        # Dedupe TP fanout: one ObjectKey per kv_rank for the same logical
        # chunk should bump the per-salt counter once.
        seen: set[tuple[str, bytes]] = set()
        for key in event.metadata.get("succeeded_keys", []):
            chunk_hash = getattr(key, "chunk_hash", None)
            if chunk_hash is None:
                continue
            cache_salt = getattr(key, "cache_salt", "")
            track_key = (cache_salt, chunk_hash)
            if track_key in seen:
                continue
            seen.add(track_key)
            counter = self._bump_salt_counter(cache_salt)
            prior = self._reuse_track.get(track_key)
            if prior is not None:
                last_time, last_counter = prior
                attrs = {"cache_salt": cache_salt}
                self._real_reuse_gap_seconds_hist.record(now - last_time, attrs)
                self._real_reuse_gap_chunks_hist.record(counter - last_counter, attrs)
                self._reuse_track[track_key] = (now, counter)
            elif self._should_sample(track_key):
                self._admit_track(track_key, now, counter)

    def _on_sm_write_finished(self, event: Event) -> None:
        """Bump counter and update anchor; never emit a sample on write."""
        now = event.timestamp or time.time()
        seen: set[tuple[str, bytes]] = set()
        for key in event.metadata.get("succeeded_keys", []):
            chunk_hash = getattr(key, "chunk_hash", None)
            if chunk_hash is None:
                continue
            cache_salt = getattr(key, "cache_salt", "")
            track_key = (cache_salt, chunk_hash)
            if track_key in seen:
                continue
            seen.add(track_key)
            counter = self._bump_salt_counter(cache_salt)
            if track_key in self._reuse_track:
                self._reuse_track[track_key] = (now, counter)
            elif self._should_sample(track_key):
                self._admit_track(track_key, now, counter)
