# SPDX-License-Identifier: Apache-2.0

"""Tests for L0LifecycleSubscriber.

Uses ``InMemoryMetricReader`` to read back actual OTel histogram values
and verifies shadow-map eviction detection logic with END_SESSION tracking.

OTel only allows one MeterProvider per process, so we use a module-scoped
provider and assert on histogram observations.
"""

# Standard
from dataclasses import dataclass
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.metrics.l0_lifecycle import (
    L0LifecycleSubscriber,
    _BlockStatus,
)
from tests.v1.mp_observability.subscribers.metrics.otel_setup import reader as _reader

# Time for the drain thread to process queued events.
_DRAIN_WAIT = 0.15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeBlockAllocationRecord:
    """Mimics BlockAllocationRecord for testing without importing vLLM types."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


def _make_allocation_event(
    records: list[FakeBlockAllocationRecord],
    instance_id: int = 0,
    model_name: str = "test-model",
) -> Event:
    return Event(
        event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
        metadata={
            "instance_id": instance_id,
            "model_name": model_name,
            "records": records,
        },
    )


def _make_end_session_event(request_id: str) -> Event:
    return Event(
        event_type=EventType.MP_VLLM_END_SESSION,
        metadata={"request_id": request_id},
    )


def _read_histograms() -> dict[str, list]:
    """Snapshot all histogram data points."""
    data = _reader.get_metrics_data()
    result: dict[str, list] = {}
    if data is None:
        return result
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                result[metric.name] = list(metric.data.data_points)
    return result


def _get_histogram_count(name: str) -> int:
    histograms = _read_histograms()
    dps = histograms.get(name, [])
    return sum(dp.count for dp in dps)


def _get_histogram_attrs(name: str) -> list[dict]:
    """Return a list of attribute dicts from all data points of a histogram."""
    histograms = _read_histograms()
    dps = histograms.get(name, [])
    return [dict(dp.attributes) for dp in dps if dp.count > 0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def subscriber(bus):
    sub = L0LifecycleSubscriber(sample_rate=1.0)
    bus.register_subscriber(sub)
    return sub


# ---------------------------------------------------------------------------
# Tests: New allocation
# ---------------------------------------------------------------------------


class TestL0NewAllocation:
    def test_new_block_no_metrics_emitted(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l0_block_lifetime")
        bus.start()
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-1", [0, 1, 2], [10, 20, 30])]
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        count_after = _get_histogram_count("lmcache_mp.l0_block_lifetime")
        assert count_after == count_before

    def test_shadow_map_populated(self, bus, subscriber):
        bus.start()
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-1", [10, 11], [100, 200])]
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert (0, 10) in subscriber._shadow
        assert (0, 11) in subscriber._shadow
        assert subscriber._shadow[(0, 10)].status == _BlockStatus.ACTIVE


# ---------------------------------------------------------------------------
# Tests: Prefix sharing (no reuse gap)
# ---------------------------------------------------------------------------


class TestL0PrefixSharing:
    def test_prefix_sharing_no_reuse_gap(self, bus, subscriber):
        """Two requests sharing the same block while both active = no reuse."""
        bus.start()

        # Request A allocates block 5.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-A", [5], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        reuse_before = _get_histogram_count("lmcache_mp.l0_block_reuse_gap")

        # Request B also uses block 5 with same tokens (prefix sharing).
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-B", [5], [42])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        reuse_after = _get_histogram_count("lmcache_mp.l0_block_reuse_gap")
        # No reuse gap should be recorded — this is prefix sharing.
        assert reuse_after == reuse_before

    def test_prefix_sharing_adds_owner(self, bus, subscriber):
        """Prefix sharing should add the new request as co-owner."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-A", [6], [99])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-B", [6], [99])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 6)]
        assert "req-A" in state.owners
        assert "req-B" in state.owners
        assert state.status == _BlockStatus.ACTIVE


# ---------------------------------------------------------------------------
# Tests: END_SESSION and true reuse
# ---------------------------------------------------------------------------


class TestL0EndSessionAndReuse:
    def test_end_session_releases_block(self, bus, subscriber):
        """After END_SESSION, block with no remaining owners is RELEASED."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [7], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert subscriber._shadow[(0, 7)].status == _BlockStatus.RELEASED

    def test_end_session_with_coowner_stays_active(self, bus, subscriber):
        """If another request still owns the block, it stays ACTIVE."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-A", [8], [10])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-B", [8], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-A"))
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 8)]
        assert state.status == _BlockStatus.ACTIVE
        assert "req-A" not in state.owners
        assert "req-B" in state.owners

    def test_true_reuse_after_release(self, bus, subscriber):
        """Block released then reused with same tokens = true cache hit."""
        bus.start()

        # Allocate.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release.
        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)

        # Reuse with same tokens — true cache hit.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release again.
        bus.publish(_make_end_session_event("req-2"))
        time.sleep(_DRAIN_WAIT)

        # Another reuse.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-3", [9], [42])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 9)]
        # Two true reuses → 2 access history entries → 1 reuse gap.
        assert len(state.access_history) == 2


# ---------------------------------------------------------------------------
# Tests: Eviction detection
# ---------------------------------------------------------------------------


class TestL0EvictionDetection:
    def test_different_tokens_triggers_eviction(self, bus, subscriber):
        count_before = _get_histogram_count("lmcache_mp.l0_block_lifetime")
        bus.start()

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [3], [10])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [3], [99])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        count_after = _get_histogram_count("lmcache_mp.l0_block_lifetime")
        assert count_after == count_before + 1

    def test_eviction_records_idle_time(self, bus, subscriber):
        idle_before = _get_histogram_count("lmcache_mp.l0_block_idle_before_evict")
        bus.start()

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [20], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [20], [2])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        idle_after = _get_histogram_count("lmcache_mp.l0_block_idle_before_evict")
        assert idle_after == idle_before + 1

    def test_eviction_clears_old_owners(self, bus, subscriber):
        """Eviction should clear old owner references."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [40], [1])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [40], [2])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 40)]
        assert state.owners == {"req-2"}
        assert state.token_ids == [2]


# ---------------------------------------------------------------------------
# Tests: Reuse gap with proper release cycle
# ---------------------------------------------------------------------------


class TestL0ReuseGaps:
    def test_reuse_gap_after_release_and_reuse(self, bus, subscriber):
        """Proper release→reuse cycle should record reuse gaps on eviction."""
        bus.start()

        # Allocate block 30.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        # Release and reuse twice (true cache hits).
        bus.publish(_make_end_session_event("req-1"))
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-2", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        bus.publish(_make_end_session_event("req-2"))
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-3", [30], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        gap_before = _get_histogram_count("lmcache_mp.l0_block_reuse_gap")

        # Now evict to flush reuse gaps.
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-evict", [30], [999])]
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        gap_after = _get_histogram_count("lmcache_mp.l0_block_reuse_gap")
        # 2 true reuses → 1 reuse gap.
        assert gap_after == gap_before + 1


# ---------------------------------------------------------------------------
# Tests: Sampling
# ---------------------------------------------------------------------------


class TestL0Sampling:
    def test_full_sample_rate_tracks_all(self, bus):
        sub = L0LifecycleSubscriber(sample_rate=1.0)
        bus.register_subscriber(sub)
        bus.start()

        for i in range(10):
            bus.publish(
                _make_allocation_event(
                    [FakeBlockAllocationRecord(f"req-{i}", [2000 + i], [i])]
                )
            )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert len(sub._shadow) == 10


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestL0EdgeCases:
    def test_empty_block_ids(self, bus, subscriber):
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-empty", [], [])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        assert len(subscriber._shadow) == 0

    def test_same_req_same_block_ignored(self, bus, subscriber):
        """Same request reporting same block again should be ignored."""
        bus.start()
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [50], [1])])
        )
        time.sleep(_DRAIN_WAIT)

        # Same request, same block, same tokens — decode continuation.
        bus.publish(
            _make_allocation_event([FakeBlockAllocationRecord("req-1", [50], [1])])
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        state = subscriber._shadow[(0, 50)]
        # No access recorded — it was the same request.
        assert len(state.access_history) == 0

    def test_end_session_unknown_req(self, bus, subscriber):
        """END_SESSION for unknown req_id should not crash."""
        bus.start()
        bus.publish(_make_end_session_event("unknown-req"))
        time.sleep(_DRAIN_WAIT)
        bus.stop()
        # No crash = pass.


# ---------------------------------------------------------------------------
# Tests: OTel attributes (instance_id, model_name)
# ---------------------------------------------------------------------------


class TestL0MetricAttributes:
    def test_eviction_emits_instance_id_and_model_name(self, bus, subscriber):
        """Histogram data points should carry instance_id and model_name."""
        bus.start()
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-1", [60], [10])],
                instance_id=42,
                model_name="llama-7b",
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.publish(
            _make_allocation_event(
                [FakeBlockAllocationRecord("req-2", [60], [99])],
                instance_id=42,
                model_name="llama-7b",
            )
        )
        time.sleep(_DRAIN_WAIT)
        bus.stop()

        attrs_list = _get_histogram_attrs("lmcache_mp.l0_block_lifetime")
        matching = [
            a
            for a in attrs_list
            if a.get("instance_id") == "42" and a.get("model_name") == "llama-7b"
        ]
        assert len(matching) > 0
