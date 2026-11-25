# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import heapq
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)

_Rank = tuple[int, int]
_IndexEntry = tuple[_Rank, OffloadKey]

# Target sublist length of _SortedEntries; a sublist is split at twice this.
_SUBLIST_LOAD = 512


class _SortedEntries:
    """Sorted list of distinct index entries in sublists of bounded length.

    An insert or remove bisects the per-sublist maxima, then moves at most one
    sublist's worth of elements. Splitting a full sublist or dropping an empty
    one also shifts the per-sublist lists, which is O(number of sublists).
    """

    __slots__ = ("_len", "_lists", "_maxes")

    def __init__(self) -> None:
        self._lists: list[list[_IndexEntry]] = []
        self._maxes: list[_IndexEntry] = []
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[_IndexEntry]:
        for sublist in self._lists:
            yield from sublist

    def clear(self) -> None:
        self._lists.clear()
        self._maxes.clear()
        self._len = 0

    def add(self, entry: _IndexEntry) -> None:
        lists, maxes = self._lists, self._maxes
        if not maxes:
            lists.append([entry])
            maxes.append(entry)
            self._len += 1
            return
        i = bisect.bisect_left(maxes, entry)
        if i == len(maxes):
            i -= 1
            lists[i].append(entry)
            maxes[i] = entry
        else:
            bisect.insort(lists[i], entry)
        sublist = lists[i]
        if len(sublist) > 2 * _SUBLIST_LOAD:
            upper = sublist[_SUBLIST_LOAD:]
            del sublist[_SUBLIST_LOAD:]
            maxes[i] = sublist[-1]
            lists.insert(i + 1, upper)
            maxes.insert(i + 1, upper[-1])
        self._len += 1

    def remove(self, entry: _IndexEntry) -> None:
        lists, maxes = self._lists, self._maxes
        i = bisect.bisect_left(maxes, entry)
        if i == len(maxes):
            raise ValueError(f"{entry!r} not in index")
        sublist = lists[i]
        j = bisect.bisect_left(sublist, entry)
        if j == len(sublist) or sublist[j] != entry:
            raise ValueError(f"{entry!r} not in index")
        del sublist[j]
        self._len -= 1
        if not sublist:
            del lists[i]
            del maxes[i]
        elif j == len(sublist):
            maxes[i] = sublist[-1]


class _EvictionIndex:
    """Exact rank order for one retention class.

    Fresh maximum ranks append to the ordered tail in O(1). Entries that
    cannot append go into bounded sorted sublists. Iteration merges both
    parts, and each indexed key has exactly one entry.
    """

    def __init__(self) -> None:
        self.tail: OrderedDict[OffloadKey, _Rank] = OrderedDict()
        self.sorted_entries = _SortedEntries()
        self.sorted_ranks: dict[OffloadKey, _Rank] = {}

    def discard(self, key: OffloadKey) -> None:
        if self.tail.pop(key, None) is None:
            rank = self.sorted_ranks.pop(key, None)
            if rank is not None:
                self.sorted_entries.remove((rank, key))

    def add(self, key: OffloadKey, rank: _Rank) -> None:
        tail = self.tail
        if tail:
            last_key = next(reversed(tail))
            if (rank, key) > (tail[last_key], last_key):
                tail[key] = rank
                return
        elif not self.sorted_ranks:
            tail[key] = rank
            return
        self.sorted_entries.add((rank, key))
        self.sorted_ranks[key] = rank

    def __iter__(self) -> Iterator[_IndexEntry]:
        tail_entries = ((rank, key) for key, rank in self.tail.items())
        if not self.sorted_ranks:
            return tail_entries
        return heapq.merge(self.sorted_entries, tail_entries)

    def clear(self) -> None:
        self.tail.clear()
        self.sorted_entries.clear()
        self.sorted_ranks.clear()


@dataclass(slots=True)
class _RequestRank:
    owner: object
    cache_generation: int
    rank: int


class LRUCachePolicy(CachePolicy):
    """LRU caching policy with logical recency independent of transfer pinning.

    Eviction takes unretained chunks before retained chunks, in increasing
    (rank, key) order within each class. Recency can be updated while pinned;
    when it later becomes evictable, it is ordered by the rank assigned by the
    request rather than by its transfer-completion time.

    Active requests share a provisional rank across batches, ordered by token
    position so tails precede heads. Finalization commits the request's recency.

    A use is indicated by,
     - First time the key is added (store).
     - A request-scoped access.

    Each evictable chunk is indexed exactly once, under its current retention
    class and rank. Updates avoid full rebuilds of the eviction order.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, _Rank] = {}
        # Record the indexed class so a lease change can remove the old entry.
        self._evictable: dict[OffloadKey, bool] = {}
        self._indexes = (_EvictionIndex(), _EvictionIndex())  # unretained, retained
        self._next_rank = 0
        self._cache_generation = 0

    def _get_request_rank(self, req_context: ReqContext) -> int:
        state = req_context.get_state(_RequestRank)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            self._next_rank += 1
            state = _RequestRank(self, self._cache_generation, self._next_rank)
            req_context.set_state(state)
        return state.rank

    def _unindex(self, key: OffloadKey) -> None:
        retained = self._evictable.pop(key, None)
        if retained is not None:
            self._indexes[retained].discard(key)

    def _push_evictable(self, key: OffloadKey) -> None:
        """Index an evictable key using its current lease state."""
        self._unindex(key)
        retained = self.chunks[key].retained
        self._evictable[key] = retained
        self._indexes[retained].add(key, self._ranks[key])

    def _update_recency(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key not in self.chunks:
                continue
            self._next_rank += 1
            rank = (self._next_rank, 0)
            self._ranks[key] = rank
            retained = self._evictable.get(key)
            if retained is not None:
                index = self._indexes[retained]
                index.discard(key)
                # A fresh rank sorts after every indexed rank.
                index.tail[key] = rank

    def _iter_evictable(self) -> Iterator[_IndexEntry]:
        return chain.from_iterable(self._indexes)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
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
            self._push_evictable(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._ranks.pop(key, None)
        self._unindex(key)

    @override
    def on_request_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        request_rank = self._get_request_rank(req_context)
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
            retained = self._evictable.get(key)
            if retained is not None:
                index = self._indexes[retained]
                index.discard(key)
                index.add(key, rank)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        self._update_recency(reversed(order_request_keys(key_groups, req_context)))

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._evictable.clear()
        for index in self._indexes:
            index.clear()
        self._next_rank = 0
        self._cache_generation += 1

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []

        selected: list[OffloadKey] = []
        for _, key in self._iter_evictable():
            if key in protected:
                continue
            assert self.chunks[key].ref_cnt == 0
            selected.append(key)
            if len(selected) == n:
                break
        if len(selected) < n:
            return None

        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for key in selected:
            chunk = self.chunks[key]
            self._unindex(key)
            del self.chunks[key]
            del self._ranks[key]
            candidates.append((key, chunk))
        return candidates

    @override
    def on_retention_changed(self, key: OffloadKey, retained: bool) -> None:
        previous = self._evictable.get(key)
        if previous is None or previous == retained:
            return
        self._indexes[previous].discard(key)
        self._evictable[key] = retained
        self._indexes[retained].add(key, self._ranks[key])

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        # chunks can become evictable when,
        # store completes - i.e. ref_cnt -1 -> 0 # not in evictable list
        # all loads complete - i.e ref_cnt 1 -> 0  # not in evictable list
        self._push_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        # key must have been evictable before it was pinned.
        retained = self._evictable.pop(key)
        self._indexes[retained].discard(key)
