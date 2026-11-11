# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontier checkpoints: a finished request's state at its exact end.

For a freed request that processed `L` tokens, `L` inside block `k`, a
frontier checkpoint is the attention page holding `[kB, L)` in every
full-attention group, the running state block in every Mamba group and, when
`L` is inside an indexer compression pool, the indexer's ring block. Each
block is cached under `(key, group)` with the frontier key of `L`
(`keys.frontier_key`). A later request whose tokens extend those `L`
tokens resumes at `L`: upstream's partial-hit copy-on-write gives it private
copies of the page and the state, and the ring is copied when it is claimed.
The checkpoint is usable only while every one of its blocks is cached; when
any of them is evicted the others lose the key too.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.utils.math_utils import cdiv
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.request import Request

# Tokens before the frontier compared ahead of the key, so that a lookup
# hashes only candidates that already end like the request.
_SUFFIX = 8

HitBlocks = tuple[list[KVCacheBlock], ...]


@dataclass(slots=True)
class _Frontier:
    key: BlockHash
    anchor: BlockHash | None
    num_tokens: int
    suffix: tuple[int, ...]
    # Group id -> the block cached under (key, group).
    blocks: dict[int, KVCacheBlock]


@dataclass(slots=True)
class _Lookup:
    hit: int
    num_tokens: int
    key: BlockHash | None
    # Live frontiers equal to the request's tokens but for their last token.
    stale: tuple[BlockHash, ...]


class FrontierIndex:
    def __init__(
        self,
        coordinator: "KVCacheCoordinator",
        hash_fn: Callable[[Any], bytes],
        pages: list[int],
        states: list[int],
        rings: list[int],
    ) -> None:
        self.pool = coordinator.block_pool
        self.managers = coordinator.single_type_managers
        self.block_size = coordinator.scheduler_block_size
        self.hash_block_size = self.pool.hash_block_size
        self.hash_fn = hash_fn
        self.pages = pages
        self.states = states
        self.rings = rings
        group_sizes = []
        for group_id in rings:
            ring = self.managers[group_id]
            assert isinstance(ring, CircularBufferManager)
            assert isinstance(ring.kv_cache_spec, CircularBufferSpec)
            ring.takes_frontier_hits = True
            group_sizes.append(ring.kv_cache_spec.group_size)
        # A frontier inside this many tokens' pool also keeps the ring.
        self.pool_size = math.lcm(1, *group_sizes)
        # Free blocks that lose their keys go to the front of the free queue.
        self.reclaim = False
        self.entries: dict[BlockHash, _Frontier] = {}
        # anchor -> num_tokens -> suffix -> keys
        self._index: dict[
            BlockHash | None, dict[int, dict[tuple[int, ...], list[BlockHash]]]
        ] = {}
        self._lookups: dict[str, _Lookup] = {}
        # Admitted request -> stale frontiers its lookup found, which its
        # finish may release.
        self.superseded: dict[str, tuple[BlockHash, ...]] = {}
        self.num_published = 0
        self.num_hits = 0
        self.num_hit_tokens = 0
        self.num_removed = 0

    def publish(self, request: "Request") -> BlockHash | None:
        """Cache the freed request's state at its processed frontier in every
        group, or in none. Returns the key."""
        self._lookups.pop(request.request_id, None)
        num_tokens = request.num_computed_tokens
        if (
            num_tokens == 0
            or num_tokens % self.block_size == 0
            or request.resumable
            or request.status == RequestStatus.FINISHED_ABORTED
            # An input token of a dispatched step is not known yet.
            or request.num_tokens < num_tokens
        ):
            return None
        blocks = self._frontier_blocks(request, num_tokens)
        if blocks is None:
            return None
        key = frontier_key(self.hash_fn, request, num_tokens, self.hash_block_size)
        if key in self.entries:
            return None
        h = num_tokens // self.hash_block_size
        anchor = request.block_hashes[h - 1] if h else None
        for group_id, block in blocks.items():
            self.pool.cache_block_under_key(
                block,
                make_block_hash_with_group_id(key, group_id),
                num_tokens,
                request,
                anchor,
            )
        start = max(h * self.hash_block_size, num_tokens - _SUFFIX)
        suffix = tuple(request.all_token_ids[start:num_tokens])
        self.entries[key] = _Frontier(key, anchor, num_tokens, suffix, blocks)
        by_length = self._index.setdefault(anchor, {}).setdefault(num_tokens, {})
        by_length.setdefault(suffix, []).append(key)
        self.num_published += 1
        return key

    def _frontier_blocks(
        self, request: "Request", num_tokens: int
    ) -> dict[int, KVCacheBlock] | None:
        """The block to cache in every group."""
        request_id = request.request_id
        k = num_tokens // self.block_size
        blocks: dict[int, KVCacheBlock] = {}
        for group_id in self.pages:
            table = self.managers[group_id].req_to_blocks.get(request_id) or []
            if len(table) <= k:
                return None
            blocks[group_id] = table[k]
        running = cdiv(num_tokens, self.block_size) - 1
        for group_id in self.states:
            table = self.managers[group_id].req_to_blocks.get(request_id) or []
            # Align mode keeps the running state in the last block.
            if len(table) != running + 1:
                return None
            blocks[group_id] = table[running]
        if num_tokens % self.pool_size:
            for group_id in self.rings:
                table = self.managers[group_id].req_to_blocks.get(request_id) or []
                if len(table) != 1:
                    return None
                blocks[group_id] = table[0]
        keyed = self.pool.cached_block_hashes_by_block
        units_per_block = self.block_size // self.hash_block_size
        for group_id, block in blocks.items():
            if block.is_null:
                return None
            # With a hash unit finer than the block, the page may already be
            # cached under the request's own key at a hash-unit boundary inside
            # it; the frontier key joins that key.
            if (block.block_hash is not None or block.block_id in keyed) and (
                group_id not in self.pages or units_per_block == 1
            ):
                return None
        return blocks

    def extend(
        self,
        request: "Request",
        blocks: HitBlocks,
        hit: int,
        max_length: int,
    ) -> tuple[HitBlocks, int]:
        """`(blocks, hit)` resumed at the longest frontier past `hit` that
        `request`'s tokens extend, or the arguments unchanged."""
        stale: list[BlockHash] = []
        found = self._lookup(request, blocks, hit, max_length, stale=stale)
        if found is None:
            self._lookups[request.request_id] = _Lookup(hit, 0, None, tuple(stale))
            return blocks, hit
        entry, k, pages = found
        self._lookups[request.request_id] = _Lookup(
            hit, entry.num_tokens, entry.key, tuple(stale)
        )
        return self._hit_blocks(blocks, pages, entry, k), entry.num_tokens

    def peek_blocks(
        self,
        request: "Request",
        blocks: HitBlocks,
        hit: int,
        max_length: int,
    ) -> tuple[HitBlocks, int]:
        """What `extend` would return, recording and removing nothing."""
        found = self._lookup(request, blocks, hit, max_length, stale=None)
        if found is None:
            return blocks, hit
        entry, k, pages = found
        return self._hit_blocks(blocks, pages, entry, k), entry.num_tokens

    def resumed_at(self, request_id: str, num_computed_tokens: int) -> BlockHash | None:
        """The frontier the request's latest lookup resumes it at, when it is
        admitted at that frontier's length."""
        lookup = self._lookups.get(request_id)
        if lookup is None or lookup.num_tokens != num_computed_tokens:
            return None
        return lookup.key

    def _lookup(
        self,
        request: "Request",
        blocks: HitBlocks,
        hit: int,
        max_length: int,
        *,
        stale: list[BlockHash] | None,
    ) -> tuple[_Frontier, int, dict[int, list[KVCacheBlock]]] | None:
        """`stale` (None for a read-only lookup) collects the live frontiers
        found equal to the request's tokens but for their last token."""
        unit = self.hash_block_size
        tokens = request.all_token_ids
        for h in range(max_length // unit, hit // unit - 1, -1):
            anchor = request.block_hashes[h - 1] if h else None
            lengths = self._index.get(anchor)
            if not lengths:
                continue
            for num_tokens in sorted(lengths, reverse=True):
                if num_tokens > max_length:
                    continue
                if num_tokens <= hit:
                    break
                start = max(h * unit, num_tokens - _SUFFIX)
                suffix = tuple(tokens[start:num_tokens])
                keys = lengths[num_tokens].get(suffix, ())
                if not keys:
                    if stale is not None:
                        last = self._mismatch_last(
                            request, num_tokens, suffix, lengths[num_tokens]
                        )
                        if last is not None:
                            stale.append(last)
                    continue
                expected = frontier_key(self.hash_fn, request, num_tokens, unit)
                if expected not in keys:
                    continue
                entry = self.entries[expected]
                if not self._still_cached(entry):
                    if stale is not None:
                        self._remove(entry)
                    continue
                k = num_tokens // self.block_size
                pages = self._prefix_pages(request, blocks, hit, k)
                if pages is None:
                    continue
                return entry, k, pages
        return None

    def _mismatch_last(
        self,
        request: "Request",
        num_tokens: int,
        suffix: tuple[int, ...],
        by_suffix: dict[tuple[int, ...], list[BlockHash]],
    ) -> BlockHash | None:
        """The frontier of `num_tokens` tokens equal to the request's first
        `num_tokens` tokens with only the last one different (an async stop
        token the client replaced, say)."""
        for other, keys in by_suffix.items():
            if other[:-1] != suffix[:-1] or other[-1] == suffix[-1]:
                continue
            key = frontier_key(
                self.hash_fn, request, num_tokens, self.hash_block_size, other[-1]
            )
            if key in keys:
                return key
        return None

    def _still_cached(self, entry: _Frontier) -> bool:
        cached = self.pool.cached_block_hash_to_block
        return all(
            cached.contain(make_block_hash_with_group_id(entry.key, g), b.block_id)
            for g, b in entry.blocks.items()
        )

    def _prefix_pages(
        self,
        request: "Request",
        blocks: HitBlocks,
        hit: int,
        k: int,
    ) -> dict[int, list[KVCacheBlock]] | None:
        """Full-attention blocks `[0, k)`: the block-aligned hit's, then the
        blocks cached under the request's own block hashes."""
        num_hit_blocks = hit // self.block_size
        pages = {g: list(blocks[g][:num_hit_blocks]) for g in self.pages}
        per_block = self.block_size // self.hash_block_size
        for i in range(num_hit_blocks, k):
            found = self.pool.get_cached_block(
                request.block_hashes[(i + 1) * per_block - 1], self.pages
            )
            if found is None:
                return None
            for group_id, block in zip(self.pages, found):
                pages[group_id].append(block)
        return pages

    def _hit_blocks(
        self,
        blocks: HitBlocks,
        pages: dict[int, list[KVCacheBlock]],
        entry: _Frontier,
        k: int,
    ) -> HitBlocks:
        hit = list(blocks)
        for group_id in self.pages:
            hit[group_id] = pages[group_id] + [entry.blocks[group_id]]
        for group_id in self.states:
            hit[group_id] = [self.pool.null_block] * k + [entry.blocks[group_id]]
        for group_id in self.rings:
            hit[group_id] = [entry.blocks[group_id]] if group_id in entry.blocks else []
        return tuple(hit)

    def hit_tokens(self, request_id: str, num_hits: int) -> int:
        """The tokens past the block-aligned hit that a frontier serves the
        request if it is admitted `num_hits` tokens in, else 0."""
        lookup = self._lookups.get(request_id)
        if lookup is None or lookup.key is None or num_hits != lookup.num_tokens:
            return 0
        return lookup.num_tokens - lookup.hit

    def admitted(self, request: "Request", num_hits: int) -> None:
        """The request was admitted `num_hits` tokens in."""
        hit_tokens = self.hit_tokens(request.request_id, num_hits)
        lookup = self._lookups.pop(request.request_id, None)
        if lookup is None:
            return
        if lookup.stale:
            self.superseded[request.request_id] = lookup.stale
        else:
            self.superseded.pop(request.request_id, None)
        if hit_tokens:
            self.num_hits += 1
            self.num_hit_tokens += hit_tokens

    def removed(self, keys: list[BlockHashWithGroupId]) -> None:
        """A frontier missing any block cannot be hit; its other blocks lose
        the key too."""
        for grouped in keys:
            entry = self.entries.get(get_block_hash(grouped))
            if entry is not None:
                self._remove(entry)

    def drop(self, key: BlockHash) -> bool:
        """Drop the frontier `key` in every group. False if it is none."""
        entry = self.entries.get(key)
        if entry is None:
            return False
        self._remove(entry)
        return True

    def _remove(self, entry: _Frontier) -> None:
        del self.entries[entry.key]
        lengths = self._index[entry.anchor]
        suffixes = lengths[entry.num_tokens]
        keys = suffixes[entry.suffix]
        keys.remove(entry.key)
        if not keys:
            del suffixes[entry.suffix]
            if not suffixes:
                del lengths[entry.num_tokens]
                if not lengths:
                    del self._index[entry.anchor]
        cached = self.pool.cached_block_hash_to_block
        with self.pool.removing("frontier"):
            for group_id, block in entry.blocks.items():
                key = make_block_hash_with_group_id(entry.key, group_id)
                if not cached.contain(key, block.block_id):
                    continue
                if block.block_hash != key:
                    # A page that stays cached under the request's own key.
                    self.pool.uncache_key(block, key)
                else:
                    self.pool._maybe_evict_cached_block(block)
                    self.to_front(block)
        self.num_removed += 1

    def to_front(self, block: KVCacheBlock) -> None:
        """With `reclaim`: a free block that just lost its keys is reused
        before any cached block."""
        if not self.reclaim or block.ref_cnt or block.prev_free_block is None:
            return
        queue = self.pool.free_block_queue
        queue.remove(block)
        queue.prepend_n([block])

    def forget(self, request_id: str) -> None:
        """The request finished: its lookup goes."""
        self._lookups.pop(request_id, None)

    def reset(self) -> None:
        self.entries.clear()
        self._index.clear()
        self._lookups.clear()
        self.superseded.clear()
