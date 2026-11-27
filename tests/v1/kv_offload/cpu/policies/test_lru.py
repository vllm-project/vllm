# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the exact eviction-order index of LRUCachePolicy.

The randomized test replays identical operation streams on LRUCachePolicy and
on a reference copy of the upstream lazy min-heap implementation, and requires
identical eviction results and policy state after every operation, with and
without retention leases.
"""

import heapq
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)
from vllm.v1.kv_offload.cpu.policies.lru import (
    _SUBLIST_LOAD,
    LRUCachePolicy,
    _SortedEntries,
)

_HeapEntry = tuple[bool, tuple[int, int], OffloadKey]


@dataclass(slots=True)
class _ReferenceRequestRank:
    owner: object
    cache_generation: int
    rank: int


class _HeapLRUReference(CachePolicy):
    """The upstream lazy min-heap LRU policy, including retention ordering."""

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, tuple[int, int]] = {}
        self._evictable: set[OffloadKey] = set()
        self._heap: list[_HeapEntry] = []
        self._next_rank = 0
        self._cache_generation = 0

    def _get_request_rank(self, req_context: ReqContext) -> int:
        state = req_context.get_state(_ReferenceRequestRank)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            self._next_rank += 1
            state = _ReferenceRequestRank(self, self._cache_generation, self._next_rank)
            req_context.set_state(state)
        return state.rank

    def _assign_new_rank(self, key: OffloadKey) -> bool:
        if key not in self.chunks:
            return False
        self._next_rank += 1
        self._ranks[key] = (self._next_rank, 0)
        return key in self._evictable

    def _heap_entry(self, key: OffloadKey) -> _HeapEntry:
        return self.chunks[key].retained, self._ranks[key], key

    def _push_evictable(self, key: OffloadKey) -> None:
        heapq.heappush(self._heap, self._heap_entry(key))

    def _is_current(self, entry: _HeapEntry) -> bool:
        retained, rank, key = entry
        return (
            key in self._evictable
            and self._ranks[key] == rank
            and self.chunks[key].retained == retained
        )

    def _rebuild_heap(self) -> None:
        self._heap = [self._heap_entry(key) for key in self._evictable]
        heapq.heapify(self._heap)

    def _maybe_compact_heap(self) -> None:
        if len(self._heap) >= max(64, 2 * len(self._evictable)):
            self._rebuild_heap()

    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        updated_evictable = [key for key in keys if self._assign_new_rank(key)]
        self._update_heap(updated_evictable)

    def _update_heap(self, updated_evictable: Sequence[OffloadKey]) -> None:
        if not updated_evictable:
            return
        # Rebuilding is linear and substantially cheaper than k heap pushes
        # for a long prefix. Small updates retain the incremental path.
        if len(updated_evictable) >= max(64, len(self._evictable) // 4):
            self._rebuild_heap()
        else:
            for key in updated_evictable:
                self._push_evictable(key)
            self._maybe_compact_heap()

    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    def insert(
        self, key: OffloadKey, chunk: ChunkStatus, req_context: ReqContext
    ) -> None:
        self.chunks[key] = chunk
        position = req_context.get_offload_key_position(key)
        if position is None:
            self._next_rank += 1
            self._ranks[key] = (self._next_rank, 0)
        else:
            self._ranks[key] = (self._get_request_rank(req_context), -position)
        if chunk.ref_cnt == 0:
            self._evictable.add(key)
            self._push_evictable(key)

    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._ranks.pop(key, None)
        self._evictable.discard(key)

    def on_request_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        request_rank = self._get_request_rank(req_context)
        updated_evictable = []
        for key in keys:
            position = req_context.get_offload_key_position(key)
            if position is None or key not in self.chunks:
                continue
            # All batches share one request rank; later positions precede
            # earlier ones without revisiting the already observed prefix.
            rank = (request_rank, -position)
            if self._ranks[key] >= rank:
                continue
            self._ranks[key] = rank
            if key in self._evictable:
                updated_evictable.append(key)
        self._update_heap(updated_evictable)

    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        self._update_recency(reversed(order_request_keys(key_groups, req_context)))

    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._evictable.clear()
        self._heap.clear()
        self._next_rank = 0
        self._cache_generation += 1

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []

        selected: list[tuple[_HeapEntry, ChunkStatus]] = []
        selected_keys: set[OffloadKey] = set()
        deferred: list[_HeapEntry] = []
        while self._heap and len(selected) < n:
            entry = heapq.heappop(self._heap)
            if not self._is_current(entry):
                continue
            key = entry[2]
            # Transfer pinning or retention changes can leave equivalent
            # lazy entries behind. Select each cache key at most once.
            if key in selected_keys:
                continue
            if key in protected:
                deferred.append(entry)
                continue
            chunk = self.chunks[key]
            assert chunk.ref_cnt == 0
            selected.append((entry, chunk))
            selected_keys.add(key)

        if len(selected) < n:
            for entry, _ in selected:
                heapq.heappush(self._heap, entry)
            for entry in deferred:
                heapq.heappush(self._heap, entry)
            return None

        for entry in deferred:
            heapq.heappush(self._heap, entry)

        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for entry, chunk in selected:
            key = entry[2]
            self._evictable.remove(key)
            del self.chunks[key]
            del self._ranks[key]
            candidates.append((key, chunk))
        self._maybe_compact_heap()
        return candidates

    def on_retention_changed(self, key: OffloadKey, retained: bool) -> None:
        if key in self._evictable:
            self._push_evictable(key)
            self._maybe_compact_heap()

    def mark_evictable(self, key: OffloadKey) -> None:
        # chunks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self._evictable.add(key)
        self._push_evictable(key)
        self._maybe_compact_heap()

    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been evictable before it was pinned.
        self._evictable.remove(key)


def _key(i: int) -> OffloadKey:
    return make_offload_key(i.to_bytes(8, "big"), i % 3)


def _assert_index_invariant(policy: LRUCachePolicy) -> None:
    indexed_keys: set[OffloadKey] = set()
    for retained in (False, True):
        index = policy._indexes[retained]
        sorted_entries = list(index.sorted_entries)
        assert len(sorted_entries) == len(index.sorted_entries)
        assert sorted_entries == sorted(
            (rank, key) for key, rank in index.sorted_ranks.items()
        )
        tail_entries = [(rank, key) for key, rank in index.tail.items()]
        assert tail_entries == sorted(tail_entries)
        for rank, key in tail_entries + sorted_entries:
            assert key not in indexed_keys
            indexed_keys.add(key)
            assert policy._ranks[key] == rank
            assert policy._evictable[key] == retained == policy.chunks[key].retained
    assert indexed_keys == set(policy._evictable)


def _state(policy: _HeapLRUReference | LRUCachePolicy):
    return (
        sorted(
            (k, c.chunk_id, c.ref_cnt, c.lease_cnt) for k, c in policy.chunks.items()
        ),
        sorted(policy._evictable),
        sorted(policy._ranks.items()),
        policy._next_rank,
    )


def _evicted(result):
    return None if result is None else [(k, c.chunk_id) for k, c in result]


@pytest.mark.parametrize("seed", range(24))
def test_lru_eviction_matches_heap_reference(seed: int):
    rng = random.Random(seed)
    with_retention = seed % 2 == 0
    capacity = rng.choice((8, 64, 400))
    policies: tuple[_HeapLRUReference, LRUCachePolicy] = (
        _HeapLRUReference(capacity),
        LRUCachePolicy(capacity),
    )
    reference, policy = policies
    keyspace = [_key(i) for i in range(capacity * 3)]
    contexts: list[ReqContext] = []
    next_chunk_id = 0

    for op in range(800):
        if not contexts or rng.random() < 0.1:
            ctx = ReqContext(req_id=f"req-{seed}-{op}")
            for key in rng.sample(keyspace, rng.randrange(1, min(40, len(keyspace)))):
                if rng.random() < 0.9:
                    ctx.set_offload_key_position(key, rng.randrange(4096) * 16)
            contexts.append(ctx)
            if len(contexts) > 6:
                contexts.pop(rng.randrange(len(contexts)))
        ctx = rng.choice(contexts)
        request_keys = list(ctx._offload_key_positions)
        kind = rng.random()

        if kind < 0.25:
            key = rng.choice(
                request_keys if request_keys and rng.random() < 0.7 else keyspace
            )
            if key in reference.chunks:
                continue
            ready = rng.random() < 0.3
            lease_count = rng.choice((0, 0, 0, 1, 2)) if with_retention else 0
            for p in policies:
                chunk = ChunkStatus(next_chunk_id)
                chunk.ref_cnt = 0 if ready else -1
                chunk.lease_cnt = lease_count
                p.insert(key, chunk, ctx)
            next_chunk_id += 1
        elif kind < 0.42:
            pending = sorted(
                k for k, c in reference.chunks.items() if c.ref_cnt in (-1, 1)
            )
            if not pending:
                continue
            key = rng.choice(pending)
            for p in policies:
                p.chunks[key].ref_cnt = 0
                p.mark_evictable(key)
        elif kind < 0.52:
            if not reference._evictable:
                continue
            key = rng.choice(sorted(reference._evictable))
            for p in policies:
                p.mark_non_evictable(key)
                p.chunks[key].ref_cnt = 1
        elif with_retention and kind < 0.59:
            if not reference.chunks:
                continue
            key = rng.choice(sorted(reference.chunks))
            lease_count = rng.randrange(3)
            for p in policies:
                chunk = p.chunks[key]
                previously_retained = chunk.retained
                chunk.lease_cnt = lease_count
                if chunk.retained != previously_retained:
                    p.on_retention_changed(key, chunk.retained)
        elif kind < 0.66:
            if not request_keys:
                continue
            keys = rng.sample(request_keys, rng.randrange(1, len(request_keys) + 1))
            for p in policies:
                p.on_request_access(keys, ctx)
        elif kind < 0.76:
            groups = (tuple(request_keys),)
            for p in policies:
                p.on_request_finished(groups, set(), set(), ctx)
        elif kind < 0.94:
            n = rng.randrange(0, 8)
            protected = set(rng.sample(keyspace, rng.randrange(0, 12)))
            results = [_evicted(p.evict(n, protected)) for p in policies]
            assert results[0] == results[1], f"seed {seed} op {op}"
        elif kind < 0.995:
            not_ready = sorted(k for k, c in reference.chunks.items() if c.ref_cnt < 0)
            if not not_ready:
                continue
            key = rng.choice(not_ready)
            for p in policies:
                p.remove(key)
        else:
            for p in policies:
                p.clear()

        assert _state(reference) == _state(policy), f"seed {seed} op {op}"
        _assert_index_invariant(policy)

    # Drain both policies completely: the full eviction order must match.
    for p in policies:
        for key, chunk in p.chunks.items():
            if chunk.ref_cnt != 0:
                chunk.ref_cnt = 0
                p.mark_evictable(key)
    n = len(reference.chunks)
    assert _evicted(reference.evict(n, set())) == _evicted(policy.evict(n, set()))
    _assert_index_invariant(policy)


def test_lru_finished_request_ranks_are_appended_to_the_tail():
    policy = LRUCachePolicy(16)
    index = policy._indexes[False]
    ctx = ReqContext(req_id="r")
    keys = [_key(i) for i in range(8)]
    for i, key in enumerate(keys):
        ctx.set_offload_key_position(key, (i + 1) * 16)
        policy.insert(key, _ready_chunk(i), ctx)
    # Tails precede heads within the running request: out of tail order.
    assert len(index.sorted_ranks) == len(keys) - 1

    policy.on_request_finished((tuple(keys),), set(), set(), ctx)

    assert list(index.tail) == list(reversed(keys))
    assert not index.sorted_ranks
    _assert_index_invariant(policy)


def test_lru_running_request_chunks_are_evicted_by_provisional_rank():
    policy = LRUCachePolicy(16)
    running = ReqContext(req_id="running")
    running_keys = [_key(i) for i in range(4)]
    for i, key in enumerate(running_keys):
        running.set_offload_key_position(key, (i + 1) * 16)
    # The running request takes its rank first, then a later request finishes.
    policy.insert(running_keys[0], _ready_chunk(0), running)
    finished = ReqContext(req_id="finished")
    finished_key = _key(50)
    policy.insert(finished_key, _ready_chunk(50), finished)
    policy.on_request_finished(((finished_key,),), set(), set(), finished)
    # More chunks of the running request complete after that: they are not
    # newer than the finished chunk, so they go to the sorted part.
    for i, key in enumerate(running_keys[1:], start=1):
        chunk = ChunkStatus(i)
        policy.insert(key, chunk, running)
        chunk.ref_cnt = 0
        policy.mark_evictable(key)
    assert set(running_keys[1:]) <= set(policy._indexes[False].sorted_ranks)
    _assert_index_invariant(policy)

    evicted = policy.evict(5, set())
    assert evicted is not None
    # Tails before heads within the running request, all before the finished
    # request's chunk.
    assert [key for key, _ in evicted] == [*reversed(running_keys), finished_key]
    _assert_index_invariant(policy)


def test_lru_failed_eviction_keeps_the_index_unchanged():
    policy = LRUCachePolicy(16)
    index = policy._indexes[False]
    running = ReqContext(req_id="running")
    keys = [_key(i) for i in range(6)]
    for i, key in enumerate(keys):
        running.set_offload_key_position(key, (i + 1) * 16)
        policy.insert(key, _ready_chunk(i), running)
    policy.on_request_finished((tuple(keys[:3]),), set(), set(), running)
    before = (list(index.tail.items()), list(index.sorted_entries))

    assert policy.evict(4, set(keys[:3])) is None

    assert (list(index.tail.items()), list(index.sorted_entries)) == before
    _assert_index_invariant(policy)


def test_lru_pinning_and_removal_unindex_chunks():
    policy = LRUCachePolicy(16)
    ctx = ReqContext(req_id="r")
    keys = [_key(i) for i in range(5)]
    for i, key in enumerate(keys):
        ctx.set_offload_key_position(key, (i + 1) * 16)
    for i, key in enumerate(keys[:4]):
        policy.insert(key, _ready_chunk(i), ctx)
    # A store that is still in flight, then fails.
    policy.insert(keys[4], ChunkStatus(4), ctx)

    policy.mark_non_evictable(keys[0])
    policy.chunks[keys[0]].ref_cnt = 1
    policy.remove(keys[4])
    _assert_index_invariant(policy)

    assert policy.evict(4, set()) is None
    evicted = policy.evict(3, set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [keys[3], keys[2], keys[1]]
    _assert_index_invariant(policy)


def test_lru_negative_eviction_count_keeps_entries():
    policy = LRUCachePolicy(1)
    key = _key(0)
    chunk = _ready_chunk(0)
    policy.insert(key, chunk, ReqContext(req_id="r"))

    assert policy.evict(-1, set()) == []
    assert policy.evict(1, set()) == [(key, chunk)]


def test_sorted_entries_matches_sorted_list_across_sublist_splits():
    rng = random.Random(0)
    entries = _SortedEntries()
    expected: list[tuple[tuple[int, int], OffloadKey]] = []
    for i in range(6 * _SUBLIST_LOAD):
        entry = ((rng.randrange(1000), -rng.randrange(1000)), _key(i))
        entries.add(entry)
        expected.append(entry)
    expected.sort()
    assert list(entries) == expected
    assert len(entries._lists) > 1

    rng.shuffle(expected)
    removed, kept = expected[: len(expected) // 2], expected[len(expected) // 2 :]
    for entry in removed:
        entries.remove(entry)
    assert list(entries) == sorted(kept)
    assert len(entries) == len(kept)
    with pytest.raises(ValueError):
        entries.remove(removed[0])


def _ready_chunk(chunk_id: int) -> ChunkStatus:
    chunk = ChunkStatus(chunk_id)
    chunk.ref_cnt = 0
    return chunk
