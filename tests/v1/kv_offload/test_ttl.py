# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.ttl import OffloadTTLTracker

_EMPTY_REQ_CTX = ReqContext(req_id="")


def to_keys(int_hashes: list[int]) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_hashes]


def test_tracker_expires_after_release():
    tracker = OffloadTTLTracker()
    h = to_keys([1, 2])
    tracker.add(h)

    acquired = tracker.acquire(h, 10.0)
    # In use: never expires, however long it runs.
    assert tracker.pop_expired(now=1e6) == []

    tracker.release(acquired, now=100.0)
    assert tracker.pop_expired(now=109.9) == []
    assert tracker.pop_expired(now=110.0) == h
    # Still tracked until the caller evicts them.
    assert len(tracker) == 2
    tracker.remove(h)
    assert len(tracker) == 0


def test_tracker_requeue_retries_unevicted_blocks():
    tracker = OffloadTTLTracker()
    h = to_keys([1])
    tracker.add(h)
    tracker.release(tracker.acquire(h, 10.0), now=100.0)

    assert tracker.pop_expired(now=110.0) == h
    assert tracker.pop_expired(now=111.0) == []
    tracker.requeue(h, now=111.0)
    assert tracker.pop_expired(now=111.0) == h


def test_tracker_in_use_until_last_user_releases():
    tracker = OffloadTTLTracker()
    h = to_keys([1])
    tracker.add(h)

    a = tracker.acquire(h, 10.0)
    b = tracker.acquire(h, 10.0)
    tracker.release(a, now=0.0)
    assert tracker.pop_expired(now=50.0) == []

    tracker.release(b, now=50.0)
    assert tracker.pop_expired(now=59.0) == []
    assert tracker.pop_expired(now=60.0) == h


def test_tracker_reused_after_release_does_not_expire_while_in_use():
    tracker = OffloadTTLTracker()
    h = to_keys([1])
    tracker.add(h)

    # Released at t=0 (deadline t=10), then reused by a long-running request.
    tracker.release(tracker.acquire(h, 10.0), now=0.0)
    acquired = tracker.acquire(h, 10.0)
    assert tracker.pop_expired(now=50.0) == []

    tracker.release(acquired, now=50.0)
    assert tracker.pop_expired(now=59.0) == []
    assert tracker.pop_expired(now=60.0) == h


def test_tracker_renewal_uses_largest_ttl_and_single_heap_item():
    tracker = OffloadTTLTracker()
    h = to_keys([1])
    tracker.add(h)

    tracker.release(tracker.acquire(h, 30.0), now=0.0)
    # Many short-TTL reuses renew the deadline with the largest TTL (30s)
    # without piling up heap items.
    for t in range(1, 101):
        tracker.release(tracker.acquire(h, 1.0), now=float(t))
    assert len(tracker._heap) == 1

    assert tracker.pop_expired(now=129.0) == []
    assert len(tracker._heap) == 1
    assert tracker.pop_expired(now=130.0) == h
    assert tracker._heap == []


def test_tracker_infinite_ttl_never_expires():
    tracker = OffloadTTLTracker()
    h = to_keys([1])
    tracker.add(h)

    tracker.release(tracker.acquire(h, None), now=0.0)
    assert tracker._heap == []
    assert tracker.pop_expired(now=1e18) == []
    assert len(tracker) == 1


def test_tracker_ignores_untracked_and_removed_blocks():
    tracker = OffloadTTLTracker()
    h1, h2 = to_keys([1, 2])
    tracker.add([h1])

    # h2 was never stored: acquire skips it.
    acquired = tracker.acquire([h1, h2], 10.0)
    assert [bh for bh, _ in acquired] == [h1]

    # h1 is evicted by the manager and re-stored while still acquired: the
    # stale acquisition must not touch the new entry.
    tracker.remove([h1])
    tracker.add([h1])
    new_acquired = tracker.acquire([h1], 10.0)
    tracker.release(acquired, now=0.0)
    assert tracker.pop_expired(now=100.0) == []

    tracker.release(new_acquired, now=100.0)
    assert tracker.pop_expired(now=110.0) == [h1]


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_manager_evict(cache_policy: str):
    manager = CPUOffloadingManager(
        num_blocks=4, cache_policy=cache_policy, enable_events=True
    )

    manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.prepare_load(to_keys([2]), _EMPTY_REQ_CTX)
    list(manager.take_events())
    assert manager._get_num_free_blocks() == 1

    # 1: evicted. 2: being loaded. 3: being stored. 4: not stored.
    assert manager.evict(to_keys([1, 2, 3, 4])) == to_keys([1])
    assert manager._get_num_free_blocks() == 2
    assert manager.lookup(to_keys([1])[0], _EMPTY_REQ_CTX) == LookupResult.MISS
    assert manager.lookup(to_keys([2])[0], _EMPTY_REQ_CTX) == LookupResult.HIT

    events = list(manager.take_events())
    assert len(events) == 1
    assert events[0].removed
    assert events[0].keys == to_keys([1])

    # 2 becomes evictable once its load completes.
    manager.complete_load(to_keys([2]), _EMPTY_REQ_CTX)
    assert manager.evict(to_keys([2])) == to_keys([2])
    assert manager._get_num_free_blocks() == 3


def test_arc_evict_does_not_populate_ghost_lists():
    manager = CPUOffloadingManager(num_blocks=4, cache_policy="arc")

    manager.prepare_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2]), _EMPTY_REQ_CTX)
    manager.touch(to_keys([2]), _EMPTY_REQ_CTX)  # promote 2 to T2

    assert manager.evict(to_keys([1, 2])) == to_keys([1, 2])
    policy = manager._policy
    assert isinstance(policy, ARCCachePolicy)
    assert not policy.t1 and not policy.t2
    assert not policy.b1 and not policy.b2


def test_store_threshold_manager_evict():
    manager = CPUOffloadingManager(num_blocks=4, store_threshold=2)
    h = to_keys([1])
    manager.lookup(h[0], _EMPTY_REQ_CTX)
    manager.lookup(h[0], _EMPTY_REQ_CTX)
    manager.prepare_store(h, _EMPTY_REQ_CTX)
    manager.complete_store(h, _EMPTY_REQ_CTX)

    assert manager.evict(h) == h
    assert manager.lookup(h[0], _EMPTY_REQ_CTX) == LookupResult.MISS


def test_tracker_heap_rebuilt_after_capacity_evictions():
    """Heap items of blocks evicted for capacity do not accumulate."""
    tracker = OffloadTTLTracker()
    for i in range(10_000):
        h = to_keys([i])
        tracker.add(h)
        tracker.release(tracker.acquire(h, 3600.0), now=0.0)
        # Capacity eviction keeps at most 10 blocks tracked.
        tracker.remove(to_keys([i - 10]))

    assert len(tracker) == 10
    assert len(tracker._heap) <= 2 * len(tracker) + 64

    # The surviving blocks still expire.
    assert sorted(tracker.pop_expired(now=3600.0)) == sorted(
        to_keys(list(range(9_990, 10_000)))
    )
