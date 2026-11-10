# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GreedyDual-Size eviction of recovery units (`prefix_cache_eviction="gds"`).

The free queue stays the only source of truth for which block an allocation
takes; before an allocation would take a cached block, this policy moves the
blocks it chooses to the front.

A unit is a recovery unit (`units.RecoveryUnit`) with the pages only it
needs: the suffix of its path (the attention page keys of the full blocks
before it) that no other unit's path holds. A cached free block that is
neither a unit's state block nor a page on a unit's path is an orphan and
serves no hit.

`gain` is the prefill seconds of `[fallback, num_tokens)`, the fallback
being the deepest unit before it on its path (an aligned unit among its page
keys, or its graph parent), or position 0; a unit that can no longer be hit
(a page on its path was lost) has gain 0. `size` is the number of blocks
freed if it goes: its state blocks (every group for a frontier, the Mamba
groups for a block-boundary unit) and its exclusive pages. A unit is used
when it is cached, resumed at or its state blocks are released by a request,
and `stamp` is the clock `L` then. The score is `stamp + gain / size`,
lowest first; `L` is the score of the last evicted unit, and an orphan
scores `L`.

Before an allocation takes a cached block, the keyless free blocks at the
queue front go first; then the free unit with the lowest score goes whole
(its state blocks, then its pages, deepest first): its keys are dropped and
its blocks go to the front. A popped entry whose score grew is pushed again
at its score; a change that lowers a score pushes the unit at once. Units a
waiting request resumes at and units a running request holds are parked
until free and unpinned. With no free unit left, the pool's order stands,
pinned blocks last, those of the waiting queue's back first: its head, the
next request admitted, keeps its hit longest. A new unit whose score is below
every other unit's when its blocks are first free is released at once while
the pool has no keyless free block.
"""

import heapq
import itertools
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierIndex
from vllm.v1.core.hybrid_checkpoint.units import CheckpointGraph, RecoveryUnit
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.request import Request

# Removed units whose fallback is kept for the units that pointed at them,
# beyond twice the live units.
_GONE = 1024
# The heap is rebuilt from its live entries past this many per live entry.
_HEAP_FACTOR = 4
_HEAP_SLACK = 1024
# Waiting requests whose prefix hits pin units, the hash units their lookups
# may walk in all (the queue's front first), and the least time between
# refreshes of that set.
_PIN_WINDOW = 256
_PIN_UNITS = 1 << 18
_PIN_REFRESH_S = 0.5
# Leaf-last tiers, lowest first.
_DEAD, _INNER, _LEAF = 0, 1, 2

# (score, sequence, unit key or orphan block); the score is (score,), or
# (tier, score) with leaf-last tiers. A unit's entry is current while the
# unit's version is the entry's sequence.
_Score = tuple[float, ...]
_Entry = tuple[_Score, int, BlockHash | KVCacheBlock]


class GreedyDualSizeEviction:
    """The `CachedBlockEvictionPolicy` of the coordinator's block pool and
    the observer of the checkpoint graph's units.

    Args:
        coordinator: The KV cache coordinator; every group's blocks come from
            its block pool.
        graph: The recovery units this policy ranks.
        frontiers: The frontier checkpoints, for waiting requests' hits.
        cost: The prefill cost model behind each unit's gain.
        waiting: The waiting requests, first to be scheduled first.
        leaf_last: Evict every inner unit before any leaf. A unit with a
            deeper live unit on its path (among the units holding its key as a
            page, or its graph children) is inner unless it is a branch point;
            the score is (tier, score) with dead 0, inner 1 and leaf 2, and
            `L` rises only when a leaf goes.
        leaf_idle_s: With `leaf_last`, a leaf (not a branch point) unused
            for this many seconds ranks as inner.

    """

    def __init__(
        self,
        coordinator: "KVCacheCoordinator",
        graph: CheckpointGraph,
        frontiers: FrontierIndex,
        cost: PrefillCostModel,
        waiting: Callable[[], Iterable["Request"]],
        *,
        leaf_last: bool = False,
        leaf_idle_s: float | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.leaf_last = leaf_last
        self._demote_s = leaf_idle_s if leaf_last else None
        # leaf_idle_s: (time of a use, unit key), oldest first.
        self._recent: deque[tuple[float, BlockHash]] = deque()
        self.pool = coordinator.block_pool
        assert all(
            m.block_pool is self.pool for m in coordinator.single_type_managers
        ), "GreedyDual-Size eviction orders one block pool"
        self.frontiers = frontiers
        self.cost = cost
        self.waiting = waiting
        self.block_size = coordinator.scheduler_block_size
        self.hashes_per_block = self.block_size // self.pool.hash_block_size
        groups = coordinator.kv_cache_config.kv_cache_groups
        self.mamba = tuple(
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        )
        self.every = tuple(range(len(groups)))
        self.page_groups = tuple(i for i in self.every if i not in self.mamba)
        self.clock = 0.0
        # Unit key -> the page keys of the full blocks before it.
        self.paths: dict[BlockHash, tuple[BlockHash, ...]] = {}
        self.units: dict[BlockHash, RecoveryUnit] = {}
        # Page key -> keys of the units whose path holds it (removed keys
        # linger until settle).
        self.on_path: dict[BlockHash, set[BlockHash]] = {}
        self.heap: list[_Entry] = []
        self._seq = itertools.count()
        self._gone: dict[BlockHash, BlockHash | None] = {}
        self._removed_units: list[RecoveryUnit] = []
        self._parked: dict[BlockHash, RecoveryUnit] = {}
        # Units not yet admitted: blocks not free, or path unknown.
        self._fresh: set[BlockHash] = set()
        # Ids of the blocks with a live orphan entry.
        self._orphans: set[int] = set()
        # Block id -> rank of the first waiting request whose hit reads it.
        self._pinned_blocks: dict[int, int] = {}
        self._pinned_at = float("-inf")
        self.num_evicted_units = 0
        self.num_evicted_blocks = 0
        graph.observer = self

    # Graph events.
    def unit_added(self, unit: RecoveryUnit) -> None:
        self._use(unit)
        self.units[unit.key] = unit
        self._fresh.add(unit.key)
        path = self.paths.get(unit.key)
        if path:
            self._index(unit, path)
        self._push(unit)

    def unit_recached(self, unit: RecoveryUnit) -> None:
        """Another copy of a cached unit: more blocks free when it goes."""
        if unit.key in self.units:
            unit.dirty = True
            self._push(unit)

    def unit_removed(self, unit: RecoveryUnit) -> None:
        if self.units.get(unit.key) is unit:
            self._drop(unit)
        if self.leaf_last and unit.parent is not None:
            # Its children moved up to its parent, which may be a branch point
            # no longer.
            self._recheck(self.units.get(unit.parent))

    def unit_used(self, unit: RecoveryUnit) -> None:
        if unit.key in self.units:
            self._use(unit)
            self._push(unit)

    def unit_linked(self, unit: RecoveryUnit) -> None:
        """Its graph parent may be set now: its fallback."""
        if unit.key in self.units:
            fallback = unit.fallback
            if self._fallback(unit) != fallback:
                self._push(unit)
            if self.leaf_last and unit.parent is not None:
                # The parent has a deeper child now: it may be a leaf no longer.
                self._recheck(self.units.get(unit.parent))

    def record(self, key: BlockHash, num_tokens: int, request: "Request") -> None:
        """`request` cached the unit `key` at `num_tokens`: its path."""
        if key in self.paths:
            return
        hashes = request.block_hashes
        n = min(num_tokens // self.block_size, len(hashes) // self.hashes_per_block)
        path = tuple(hashes[(j + 1) * self.hashes_per_block - 1] for j in range(n))
        self.paths[key] = path
        unit = self.units.get(key)
        if unit is not None and not unit.path:
            self._index(unit, path)
            self._push(unit)
            if key in self._fresh and self._free(unit):
                self._admit(unit)

    # Block pool events.
    def keys_removed(self, keys: list[BlockHashWithGroupId]) -> None:
        """A unit whose path lost a page frees one block less when it goes;
        with no copy of the page left, it cannot be hit. A state key that
        never became a unit no longer needs its path."""
        cached = self.pool.cached_block_hash_to_block
        for grouped in keys:
            if get_group_id(grouped) not in self.page_groups:
                key = get_block_hash(grouped)
                if key not in self.units:
                    self.paths.pop(key, None)
                continue
            holders = self.on_path.get(get_block_hash(grouped))
            if not holders:
                continue
            lost = cached.get_one_block(grouped) is None
            for key in list(holders):
                unit = self.units.get(key)
                if unit is None:
                    continue
                unit.dirty = True
                if lost and not unit.dead:
                    unit.dead = True
                    self._push(unit)

    def on_blocks_freed(self, blocks: list[KVCacheBlock]) -> None:
        """`blocks` (cached) went back to the free queue. A unit whose state
        block a request released was used now."""
        released: dict[BlockHash, RecoveryUnit] = {}
        for block in blocks:
            needed = False
            for grouped in self._keys(block):
                key = get_block_hash(grouped)
                unit = self.units.get(key)
                if unit is not None:
                    needed = True
                    if unit.frontier or get_group_id(grouped) not in self.page_groups:
                        released[key] = unit
                elif get_group_id(grouped) in self.page_groups and any(
                    k in self.units for k in self.on_path.get(key, ())
                ):
                    needed = True
            if not needed and self._orphan(block):
                self._push_orphan(block)
        for key, unit in released.items():
            self._use(unit)
            if not self._free(unit):
                continue
            self._parked.pop(key, None)
            self._push(unit)
            if key in self._fresh and unit.path:
                self._admit(unit)

    def prepare_eviction(self, num_blocks: int) -> None:
        """Before `num_blocks` blocks leave the front of the free queue."""
        queue = self.pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        for _ in range(num_blocks):
            if block is None or block is queue.fake_free_list_tail:
                return
            if self._keyed(block):
                break
            block = block.next_free_block
        else:
            return
        self._reorder(num_blocks)

    # Units.
    def _use(self, unit: RecoveryUnit) -> None:
        unit.stamp = self.clock
        if self._demote_s is not None:
            unit.used_at = time.monotonic()
            self._recent.append((unit.used_at, unit.key))

    def _index(self, unit: RecoveryUnit, path: tuple[BlockHash, ...]) -> None:
        """Put `unit` on its path's pages. A page it shares ends the chain of
        the one unit that held it alone (that score grows: lazy); a unit
        cached under deeper units becomes their fallback (their score drops:
        pushed now)."""
        unit.path = path
        key, units, on_path = unit.key, self.units, self.on_path
        for p in path:
            holders = on_path.get(p)
            if holders is None:
                on_path[p] = {key}
                continue
            sole = self._sole(holders, key)
            if sole is not None:
                sole.dirty = True
            holders.add(key)
        unit.dirty = True
        self._fallback(unit)
        if self.leaf_last:
            # The units on its path have a deeper unit now: a leaf among them
            # becomes inner.
            for p in path:
                if p != key:
                    self._recheck(units.get(p))
        if unit.frontier:
            return
        for k in list(on_path.get(key, ())):
            deeper = units.get(k)
            if deeper is None or deeper is unit or deeper.num_tokens <= unit.num_tokens:
                continue
            fallback = self._resolve(deeper)
            if fallback is None or units[fallback].num_tokens < unit.num_tokens:
                deeper.fallback = key
                deeper.dirty = True
                self._push(deeper)

    def _drop(self, unit: RecoveryUnit, orphans: bool = True) -> None:
        """`unit` left. Its pages no other unit holds are orphans (if still
        cached and `orphans`: an eviction drops their keys itself); the one
        other unit on its deepest shared page takes that page and the ones
        below it that only the two held (its score drops)."""
        key = unit.key
        if self.units.get(key) is not unit:
            return
        del self.units[key]
        self._gone[key] = self._resolve(unit)
        unit.version = -1
        self._parked.pop(key, None)
        self._fresh.discard(key)
        self.paths.pop(key, None)
        self._removed_units.append(unit)
        units, cached = self.units, self.pool.cached_block_hash_to_block
        for p in reversed(unit.path):
            holders = self.on_path.get(p)
            if holders is not None and any(k in units for k in holders):
                owner = self._sole(holders, None)
                if owner is not None:
                    owner.dirty = True
                    self._push(owner)
                break
            if not orphans:
                continue
            for group_id in self.page_groups:
                grouped = make_block_hash_with_group_id(p, group_id)
                for b in cached.get_all_blocks(grouped):
                    if self._orphan(b):
                        self._push_orphan(b)
        if orphans:
            for b in self._state_blocks(unit):
                if self._orphan(b):
                    self._push_orphan(b)
        if self.leaf_last:
            # A branch under the units on its path went, and its fallback lost
            # a deeper unit: a branch point among them may be inner now.
            for p in unit.path:
                self._recheck(units.get(p))
            fallback = self._gone.get(key)
            if fallback is not None:
                self._recheck(units.get(fallback))

    def _sole(
        self, holders: set[BlockHash], exclude: BlockHash | None
    ) -> RecoveryUnit | None:
        """The one live unit among `holders` other than `exclude`."""
        found = None
        for k in holders:
            if k != exclude:
                unit = self.units.get(k)
                if unit is not None:
                    if found is not None:
                        return None
                    found = unit
        return found

    def _resolve(self, unit: RecoveryUnit) -> BlockHash | None:
        fallback = unit.fallback
        while fallback is not None and fallback not in self.units:
            fallback = self._gone.get(fallback)
        unit.fallback = fallback
        return fallback

    def _fallback(self, unit: RecoveryUnit) -> BlockHash | None:
        """The deepest unit before `unit` on its path: its graph parent or
        an aligned unit among its page keys."""
        units = self.units
        best, best_tokens = None, -1
        parent = units.get(unit.parent) if unit.parent is not None else None
        if (
            parent is not None
            and parent is not unit
            and parent.num_tokens < unit.num_tokens
        ):
            best, best_tokens = parent.key, parent.num_tokens
        path = unit.path
        for j in range(len(path) - 1, -1, -1):
            found = units.get(path[j])
            if (
                found is None
                or found is unit
                or found.frontier
                or found.num_tokens >= unit.num_tokens
            ):
                continue
            if found.num_tokens > best_tokens:
                best = found.key
            break
        unit.fallback = best
        return best

    def _state_blocks(self, unit: RecoveryUnit) -> list[KVCacheBlock]:
        cached = self.pool.cached_block_hash_to_block
        return [
            b
            for group_id in (self.every if unit.frontier else self.mamba)
            for b in cached.get_all_blocks(
                make_block_hash_with_group_id(unit.key, group_id)
            )
        ]

    def _exclusive(self, unit: RecoveryUnit) -> list[KVCacheBlock]:
        """The page blocks of `unit`'s exclusive chain, deepest first."""
        units, cached = self.units, self.pool.cached_block_hash_to_block
        out: list[KVCacheBlock] = []
        for p in reversed(unit.path):
            holders = self.on_path.get(p)
            if holders is None or any(k in units and k != unit.key for k in holders):
                break
            for group_id in self.page_groups:
                grouped = make_block_hash_with_group_id(p, group_id)
                out.extend(cached.get_all_blocks(grouped))
        return out

    def _shared_pages(self, unit: RecoveryUnit) -> list[KVCacheBlock]:
        """A frontier's pages that are also cached under a key of their own
        (a hash unit finer than the block): evicting the frontier only removes
        its key from them."""
        if not unit.frontier:
            return []
        cached = self.pool.cached_block_hash_to_block
        return [
            b
            for group_id in self.page_groups
            for grouped in (make_block_hash_with_group_id(unit.key, group_id),)
            for b in cached.get_all_blocks(grouped)
            if b.block_hash != grouped
        ]

    def _size(self, unit: RecoveryUnit) -> int:
        if unit.dirty:
            unit.size = max(
                1,
                len(self._state_blocks(unit))
                - len(self._shared_pages(unit))
                + len(self._exclusive(unit)),
            )
            unit.dirty = False
        return unit.size

    def _gain(self, unit: RecoveryUnit) -> float:
        if unit.dead:
            return 0.0
        fallback = self._resolve(unit)
        base = self.units[fallback].num_tokens if fallback is not None else 0
        if unit.num_tokens <= base:
            return 0.0
        return max(0.0, self.cost.seconds(base, unit.num_tokens))

    def _score(self, unit: RecoveryUnit) -> _Score:
        score = unit.stamp + self._gain(unit) / self._size(unit)
        if not self.leaf_last:
            return (score,)
        unit.tier = self._tier(unit)
        return (unit.tier, score)

    def _orphan_score(self) -> _Score:
        return (_DEAD, self.clock) if self.leaf_last else (self.clock,)

    def _tier(self, unit: RecoveryUnit) -> int:
        """Dead (it cannot be hit), inner (a deeper live unit on its path, and
        not a branch point) or leaf."""
        if unit.dead:
            return _DEAD
        if not self._deeper(unit):
            if (
                self._demote_s is not None
                and time.monotonic() - unit.used_at > self._demote_s
            ):
                return _INNER
            return _LEAF
        if unit.is_branch_point or self._forks(unit):
            return _LEAF
        return _INNER

    def _deeper(self, unit: RecoveryUnit) -> bool:
        """A live unit deeper than `unit` holds its key as a page, or is its
        child in the graph."""
        units = self.units
        for k in (*self.on_path.get(unit.key, ()), *unit.children):
            other = units.get(k)
            if (
                other is not None
                and other is not unit
                and other.num_tokens > unit.num_tokens
            ):
                return True
        return False

    def _forks(self, unit: RecoveryUnit) -> bool:
        """Two live units deeper than `unit` whose paths part in the full
        block right after it."""
        j = unit.num_tokens // self.block_size
        first = None
        for k in self.on_path.get(unit.key, ()):
            other = self.units.get(k)
            if (
                other is None
                or other is unit
                or other.num_tokens <= unit.num_tokens
                or len(other.path) <= j
            ):
                continue
            if first is None:
                first = other.path[j]
            elif other.path[j] != first:
                return True
        return False

    def _recheck(self, unit: RecoveryUnit | None) -> None:
        """`unit`'s tier may have fallen: a unit last scored as a leaf is
        pushed again at its score now."""
        if unit is not None and unit.tier == _LEAF:
            self._push(unit)

    def _push(self, unit: RecoveryUnit) -> None:
        if self.units.get(unit.key) is not unit:
            return
        unit.version = next(self._seq)
        heapq.heappush(self.heap, (self._score(unit), unit.version, unit.key))

    def _push_orphan(self, block: KVCacheBlock) -> None:
        if block.block_id in self._orphans:
            return
        self._orphans.add(block.block_id)
        heapq.heappush(self.heap, (self._orphan_score(), next(self._seq), block))

    def _live(self, entry: _Entry) -> bool:
        _score, seq, ref = entry
        if isinstance(ref, KVCacheBlock):
            return ref.block_id in self._orphans
        unit = self.units.get(ref)
        return unit is not None and unit.version == seq

    def _keys(self, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        """Every prefix-cache key `block` holds, its own hash last."""
        keys = list(self.pool.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            keys.append(block.block_hash)
        return keys

    def _keyed(self, block: KVCacheBlock) -> bool:
        return (
            block.block_hash is not None
            or block.block_id in self.pool.cached_block_hashes_by_block
        )

    def _orphan(self, block: KVCacheBlock) -> bool:
        """A free cached block no unit needs."""
        if block.is_null or block.ref_cnt or block.prev_free_block is None:
            return False
        keys = self._keys(block)
        if not keys:
            return False
        for grouped in keys:
            key = get_block_hash(grouped)
            if key in self.units:
                return False
            if get_group_id(grouped) in self.page_groups and any(
                k in self.units for k in self.on_path.get(key, ())
            ):
                return False
        return True

    def _free(self, unit: RecoveryUnit) -> bool:
        blocks = self._state_blocks(unit)
        return bool(blocks) and all(b.ref_cnt == 0 for b in blocks)

    def pinned(self, unit: RecoveryUnit) -> bool:
        """Whether a waiting request's hit holds `unit`'s state blocks."""
        pinned = self._pinned_blocks
        return bool(pinned) and any(
            b.block_id in pinned for b in self._state_blocks(unit)
        )

    def _keyless_front(self) -> bool:
        queue = self.pool.free_block_queue
        node = queue.fake_free_list_head.next_free_block
        return (
            node is not None
            and node is not queue.fake_free_list_tail
            and not self._keyed(node)
        )

    def _admit(self, unit: RecoveryUnit) -> None:
        """`unit`'s blocks are free. Under pressure (no keyless free block) a
        unit whose score is below every other's goes at once."""
        self._fresh.discard(unit.key)
        if self._keyless_front():
            return
        low = self._lowest(unit.key)
        if low is not None and self._score(unit) < low:
            self._to_front(self._evict(unit))

    def _lowest(self, exclude: BlockHash) -> _Score | None:
        """The lowest score of a live entry other than `exclude`'s; stale
        entries on the way out of the heap are dropped."""
        heap, stash = self.heap, []
        low = None
        while heap:
            entry = heap[0]
            if not self._live(entry):
                heapq.heappop(heap)
                continue
            ref = entry[2]
            if not isinstance(ref, KVCacheBlock) and ref == exclude:
                stash.append(heapq.heappop(heap))
                continue
            low = entry[0]
            break
        for entry in stash:
            heapq.heappush(heap, entry)
        return low

    # Eviction.
    def _reorder(self, num_blocks: int) -> None:
        queue = self.pool.free_block_queue
        end = queue.fake_free_list_tail
        node = queue.fake_free_list_head.next_free_block
        ready = 0
        while (
            ready < num_blocks
            and node is not None
            and node is not end
            and not self._keyed(node)
        ):
            ready += 1
            node = node.next_free_block
        need = num_blocks - ready
        out: list[KVCacheBlock] = []
        while len(out) < need:
            blocks = self._pop()
            if blocks is None:
                break
            out.extend(blocks)
        self._to_front(out)
        if len(out) < need:
            self._pool_order(need - len(out))

    def _pop(self) -> list[KVCacheBlock] | None:
        """Evict the free unit or orphan with the lowest score; its freed
        blocks, or None when none is left."""
        heap = self.heap
        while heap:
            score, seq, ref = heap[0]
            if isinstance(ref, KVCacheBlock):
                heapq.heappop(heap)
                self._orphans.discard(ref.block_id)
                if self._orphan(ref):
                    self.pool._maybe_evict_cached_block(ref)
                    return [ref]
                continue
            unit = self.units.get(ref)
            if unit is None or unit.version != seq:
                heapq.heappop(heap)
                continue
            now = self._score(unit)
            if now > score:
                unit.version = next(self._seq)
                heapq.heapreplace(heap, (now, unit.version, ref))
                continue
            heapq.heappop(heap)
            if not self._free(unit) or self.pinned(unit):
                self._parked[ref] = unit
                continue
            if not self.leaf_last or unit.tier == _LEAF:
                self.clock = max(self.clock, now[-1])
            blocks = self._evict(unit)
            self.num_evicted_units += 1
            self.num_evicted_blocks += len(blocks)
            return blocks
        return None

    def _evict(self, unit: RecoveryUnit) -> list[KVCacheBlock]:
        """Drop `unit`'s keys: its state blocks, then its exclusive pages,
        deepest first. Returns the freed blocks. A frontier's page cached under
        a key of its own as well keeps that key and is not freed."""
        shared = self._shared_pages(unit)
        kept = {b.block_id for b in shared}
        blocks = [
            b
            for b in self._state_blocks(unit) + self._exclusive(unit)
            if b.ref_cnt == 0 and b.block_id not in kept
        ]
        self._drop(unit, orphans=False)
        for b in shared:
            assert b.block_hash is not None
            group_id = get_group_id(b.block_hash)
            self.pool.uncache_key(b, make_block_hash_with_group_id(unit.key, group_id))
        for b in blocks:
            self.pool._maybe_evict_cached_block(b)
        return blocks

    def _to_front(self, blocks: list[KVCacheBlock]) -> None:
        queue = self.pool.free_block_queue
        moved = [b for b in blocks if b.prev_free_block is not None]
        for b in moved:
            queue.remove(b)
        queue.prepend_n(moved)

    def _pool_order(self, n: int) -> None:
        """No free unit is left: the next `n` keyed free blocks in the pool's
        order, pinned ones last, the waiting queue's back first."""
        queue = self.pool.free_block_queue
        end = queue.fake_free_list_tail
        node = queue.fake_free_list_head.next_free_block
        picked: list[KVCacheBlock] = []
        pinned: list[KVCacheBlock] = []
        while node is not None and node is not end and len(picked) < n:
            if self._keyed(node):
                (pinned if node.block_id in self._pinned_blocks else picked).append(
                    node
                )
            node = node.next_free_block
        if len(picked) < n:
            pinned.sort(key=lambda block: -self._pinned_blocks[block.block_id])
            picked += pinned[: n - len(picked)]
        self._to_front(picked)

    # Per step.
    def new_step(self) -> None:
        self._refresh_pins()
        self.settle()

    def _refresh_pins(self) -> None:
        """Pin the blocks the waiting requests' prefix hits read."""
        now = time.monotonic()
        if now - self._pinned_at < _PIN_REFRESH_S:
            return
        self._pinned_at = now
        pinned: dict[int, int] = {}
        budget = _PIN_UNITS
        for rank, request in enumerate(itertools.islice(self.waiting(), _PIN_WINDOW)):
            budget -= request.num_tokens // self.pool.hash_block_size + 1
            if budget < 0:
                break
            max_length = max(request.num_tokens - 1, 0)
            blocks, hit, _ = self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_length
            )
            blocks, hit = self.frontiers.peek_blocks(request, blocks, hit, max_length)
            if not hit:
                continue
            for group in blocks:
                for block in group:
                    if not block.is_null:
                        pinned.setdefault(block.block_id, rank)
        self._pinned_blocks = pinned

    def settle(self) -> None:
        """Bookkeeping outside allocations: removed units leave their pages'
        sets; parked units that are free and unpinned, and fresh ones, are
        pushed or admitted."""
        units = self.units
        for unit in self._removed_units:
            if unit.key in units:
                continue
            for p in unit.path:
                holders = self.on_path.get(p)
                if holders is not None:
                    holders.discard(unit.key)
                    if not holders:
                        del self.on_path[p]
        self._removed_units.clear()
        for key, unit in list(self._parked.items()):
            if units.get(key) is not unit:
                del self._parked[key]
            elif self._free(unit) and not self.pinned(unit):
                del self._parked[key]
                self._push(unit)
        for key in list(self._fresh):
            fresh = units.get(key)
            if fresh is None:
                self._fresh.discard(key)
            elif self._free(fresh):
                self._admit(fresh)
        if self._demote_s is not None:
            # Leaves unused for leaf_idle_s rank as inner from now: pushed again
            # at their lower score.
            recent, cut = self._recent, time.monotonic() - self._demote_s
            while recent and recent[0][0] < cut:
                at, key = recent.popleft()
                idle = units.get(key)
                if idle is not None and idle.used_at == at and idle.tier == _LEAF:
                    self._push(idle)
        if len(self._gone) > 2 * len(units) + _GONE:
            for unit in units.values():
                self._resolve(unit)
            self._gone.clear()
        if (
            len(self.heap)
            > _HEAP_FACTOR * (len(units) + len(self._orphans)) + _HEAP_SLACK
        ):
            self.heap = [entry for entry in self.heap if self._live(entry)]
            heapq.heapify(self.heap)

    def reset(self) -> None:
        """The prefix cache was reset."""
        self.units.clear()
        self.on_path.clear()
        self.heap.clear()
        self.paths.clear()
        self._gone.clear()
        self._removed_units.clear()
        self._parked.clear()
        self._fresh.clear()
        self._orphans.clear()
        self._recent.clear()
        self._pinned_blocks.clear()
        self._pinned_at = float("-inf")
