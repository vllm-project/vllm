# SPDX-License-Identifier: Apache-2.0

"""L0 (GPU) KV cache metrics subscriber — OTel histograms for block lifecycle.

Subscribes to ``MP_VLLM_BLOCK_ALLOCATION`` and ``MP_VLLM_END_SESSION`` events
and maintains a shadow map of physical GPU block IDs to detect cache hits and
evictions.

State machine per block:

  1. Block first seen in allocation → ACTIVE (owned by req_id)
  2. Same block, same tokens, same req_id → ignore (decode continuation)
  3. Same block, same tokens, different req_id, block ACTIVE →
     prefix sharing, just add req_id as co-owner
  4. Same block, same tokens, block RELEASED → true cache hit (reuse),
     record access, transition to ACTIVE
  5. Same block, different tokens → eviction detected,
     emit metrics, start fresh
  6. END_SESSION(req_id) → remove req_id from all blocks it owns.
     If a block has no remaining owners → RELEASED.

Limitations:
  - ``BlockAllocationRecord`` only reports **new** block IDs per request per
    scheduler step, not all blocks a request uses.  Blocks reused via prefix
    cache are invisible after initial allocation.  This means the subscriber
    only tracks a subset of physical blocks and will undercount evictions
    compared to vLLM's internal ``KVCacheMetricsCollector``.
  - Eviction is detected at **reallocation** time, not at the exact moment
    vLLM frees the block.  Lifetime measurements include the gap between
    eviction and reallocation.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import random
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

# Maximum number of recent access timestamps kept per block (ring buffer).
_MAX_ACCESS_HISTORY = 4


class _BlockStatus(Enum):
    ACTIVE = "active"  # Owned by at least one live request.
    RELEASED = "released"  # All owning requests have ended.


@dataclass
class _L0BlockState:
    """Per-block lifecycle state in the shadow map."""

    instance_id: int
    model_name: str
    token_ids: list[int]
    owners: set[str]  # Set of req_ids currently using this block.
    status: _BlockStatus
    alloc_time: float
    last_access_time: float
    access_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_MAX_ACCESS_HISTORY)
    )


class L0LifecycleSubscriber(EventSubscriber):
    """Tracks GPU (L0) KV cache block lifecycle via shadow monitoring.

    Metrics (all histograms, in seconds):
    - ``lmcache_mp.l0_block_lifetime``
    - ``lmcache_mp.l0_block_idle_before_evict``
    - ``lmcache_mp.l0_block_reuse_gap``

    Parameters:
        sample_rate: Fraction of blocks to track (0, 1.0].  Default 0.01
            (1%) matches vLLM's default.
    """

    def __init__(self, sample_rate: float = 0.01) -> None:
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self._sample_rate = sample_rate

        # Shadow map: (instance_id, block_id) -> lifecycle state.
        self._shadow: dict[tuple[int, int], _L0BlockState] = {}
        # Set of (instance_id, block_id) we decided NOT to sample.
        self._skipped: set[tuple[int, int]] = set()
        # Reverse index: req_id -> set of (instance_id, block_id) owned.
        self._req_blocks: dict[str, set[tuple[int, int]]] = {}

        meter = metrics.get_meter("lmcache.l0")
        self._lifetime_hist = meter.create_histogram(
            "lmcache_mp.l0_block_lifetime",
            description=(
                "Histogram of GPU KV cache block lifetime from "
                "allocation to eviction (seconds)."
            ),
            unit="s",
        )
        self._idle_hist = meter.create_histogram(
            "lmcache_mp.l0_block_idle_before_evict",
            description=(
                "Histogram of idle time before GPU KV cache block eviction (seconds)."
            ),
            unit="s",
        )
        self._reuse_gap_hist = meter.create_histogram(
            "lmcache_mp.l0_block_reuse_gap",
            description=(
                "Histogram of time gaps between consecutive GPU KV "
                "cache block accesses (seconds)."
            ),
            unit="s",
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_VLLM_BLOCK_ALLOCATION: self._on_block_allocation,
            EventType.MP_VLLM_END_SESSION: self._on_end_session,
        }

    # -- Event handlers ----------------------------------------------------

    def _on_block_allocation(self, event: Event) -> None:
        """Process a batch of ``BlockAllocationRecord`` from vLLM."""
        instance_id = event.metadata.get("instance_id", 0)
        model_name = event.metadata.get("model_name", "")
        records = event.metadata.get("records", [])
        now = event.timestamp or time.time()

        for record in records:
            self._process_record(instance_id, model_name, record, now)

    def _on_end_session(self, event: Event) -> None:
        """Handle request completion — release blocks owned by this request."""
        req_id = event.metadata.get("request_id", "")
        if not req_id:
            return

        block_keys = self._req_blocks.pop(req_id, set())
        for block_key in block_keys:
            state = self._shadow.get(block_key)
            if state is None:
                continue
            state.owners.discard(req_id)
            if not state.owners:
                state.status = _BlockStatus.RELEASED

    # -- Record processing -------------------------------------------------

    def _process_record(
        self, instance_id: int, model_name: str, record: object, now: float
    ) -> None:
        """Process a single BlockAllocationRecord."""
        req_id: str = record.req_id  # type: ignore[attr-defined]
        block_ids: list[int] = record.new_block_ids  # type: ignore[attr-defined]
        token_ids: list[int] = record.new_token_ids  # type: ignore[attr-defined]

        if not block_ids:
            return

        num_blocks = len(block_ids)
        block_size = (
            (len(token_ids) + num_blocks - 1) // num_blocks if num_blocks else 0
        )

        for i, block_id in enumerate(block_ids):
            start = i * block_size
            end = min(start + block_size, len(token_ids))
            chunk_tokens = token_ids[start:end]
            if not chunk_tokens:
                continue
            self._process_block(
                instance_id, model_name, block_id, chunk_tokens, req_id, now
            )

    def _process_block(
        self,
        instance_id: int,
        model_name: str,
        block_id: int,
        token_ids: list[int],
        req_id: str,
        now: float,
    ) -> None:
        """Update shadow map for a single physical block."""
        block_key = (instance_id, block_id)
        existing = self._shadow.get(block_key)

        if existing is None:
            # Block not in shadow map.
            if block_key in self._skipped:
                return

            if not self._should_sample():
                self._skipped.add(block_key)
                return

            # New allocation — start tracking.
            self._shadow[block_key] = _L0BlockState(
                instance_id=instance_id,
                model_name=model_name,
                token_ids=token_ids,
                owners={req_id},
                status=_BlockStatus.ACTIVE,
                alloc_time=now,
                last_access_time=now,
            )
            self._req_blocks.setdefault(req_id, set()).add(block_key)
            return

        # Block exists in shadow map.
        if existing.token_ids == token_ids:
            # Same content.
            if req_id in existing.owners:
                # Case 2: Same request reporting same block again (decode
                # continuation or redundant report). Ignore.
                return

            if existing.status == _BlockStatus.ACTIVE:
                # Case 3: Prefix sharing — another request is using this
                # block while original request(s) still active. Not a reuse.
                existing.owners.add(req_id)
                self._req_blocks.setdefault(req_id, set()).add(block_key)
            else:
                # Case 4: Block was RELEASED, now reused with same content.
                # This is a true cache hit.
                existing.owners.add(req_id)
                existing.status = _BlockStatus.ACTIVE
                existing.last_access_time = now
                existing.access_history.append(now)
                self._req_blocks.setdefault(req_id, set()).add(block_key)
        else:
            # Case 5: Different content — eviction detected.
            self._emit_eviction_metrics(existing, now)

            # Clear old owners from reverse index.
            for old_req in existing.owners:
                block_set = self._req_blocks.get(old_req)
                if block_set:
                    block_set.discard(block_key)

            # Start fresh.
            self._shadow[block_key] = _L0BlockState(
                instance_id=instance_id,
                model_name=model_name,
                token_ids=token_ids,
                owners={req_id},
                status=_BlockStatus.ACTIVE,
                alloc_time=now,
                last_access_time=now,
            )
            self._req_blocks.setdefault(req_id, set()).add(block_key)

    # -- Metrics emission --------------------------------------------------

    def _emit_eviction_metrics(self, state: _L0BlockState, now: float) -> None:
        """Record histogram observations for an evicted block."""
        lifetime = now - state.alloc_time
        idle_time = now - state.last_access_time
        attrs = {
            "instance_id": str(state.instance_id),
            "model_name": state.model_name,
        }

        self._lifetime_hist.record(lifetime, attrs)
        self._idle_hist.record(idle_time, attrs)

        # Reuse gaps from access history.
        history = list(state.access_history)
        for i in range(1, len(history)):
            gap = history[i] - history[i - 1]
            self._reuse_gap_hist.record(gap, attrs)

    # -- Sampling ----------------------------------------------------------

    def _should_sample(self) -> bool:
        return random.random() < self._sample_rate
