# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-scoped lookup acceleration for the prefix cache.

A session is the latest request seen for a `session_id`. Its full-attention
block lists let the next turn's prefix lookup skip the blocks the two turns
share. Eviction order is separate: see `KVCacheBlock.retain_until`.
"""

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
    resolve_block_hashes,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(slots=True)
class _Session:
    """Per lookup group, the leading blocks of a session's latest request that
    were cached under the request's hash for their position, and that hash."""

    blocks: dict[int, list[KVCacheBlock]]
    keys: dict[int, list[BlockHashWithGroupId]]
    expires_at: float = 0.0


def _num_cached(
    blocks: Sequence[KVCacheBlock],
    keys: Sequence[BlockHashWithGroupId],
    end: int | None = None,
) -> int:
    """Count the leading blocks, up to `end`, that were not evicted since
    their hash was recorded in `keys`."""
    num_cached = 0
    for block, key in islice(zip(blocks, keys), end):
        if block.block_hash is not key:
            break
        num_cached += 1
    return num_cached


class KVCacheSessions:
    """Tracks the cached blocks of each live session.

    Args:
        ttl: Seconds a session outlives its latest request.
        hash_block_size: Token granularity of request block hashes.
        lookup_block_sizes: Block size of each KV cache group whose lookup
            scans from the first block and so can start from a session's blocks.
        max_sessions: Sessions tracked before the oldest is dropped.

    """

    def __init__(
        self,
        ttl: float,
        hash_block_size: int,
        lookup_block_sizes: dict[int, int],
        max_sessions: int = 65536,
    ) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.hash_block_size = hash_block_size
        self.lookup_block_sizes = lookup_block_sizes
        # Ordered by expiry. Keyed by session id and the request's first block
        # hash, so requests that share an id but not a prefix (an agent's side
        # calls) do not displace each other.
        self._sessions: OrderedDict[tuple[str, bytes], _Session] = OrderedDict()

    def register(
        self, request: "Request", blocks_by_group: Sequence[Sequence[KVCacheBlock]]
    ) -> None:
        """Record a finished request's blocks as its session's prefix."""
        if not request.session_id or not request.block_hashes or not self.ttl:
            return
        session_key = (request.session_id, request.block_hashes[0])
        session = self._sessions.pop(session_key, None) or _Session({}, {})
        session.expires_at = time.monotonic() + self.ttl
        for group_id, block_size in self.lookup_block_sizes.items():
            blocks = blocks_by_group[group_id]
            kept = session.blocks.setdefault(group_id, [])
            keys = session.keys.setdefault(group_id, [])
            # Keep what the previous turn recorded if this one extends it.
            num_kept = min(len(kept), len(blocks))
            if not num_kept or kept[num_kept - 1] is not blocks[num_kept - 1]:
                num_kept = 0
            del kept[num_kept:], keys[num_kept:]
            hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, block_size
            )
            # A block can also be cached under a partial-prefix hash.
            for block, block_hash in zip(blocks[num_kept:], hashes[num_kept:]):
                key = block.block_hash
                if key != make_block_hash_with_group_id(block_hash, group_id):
                    break
                kept.append(block)
                keys.append(key)

        self._sessions[session_key] = session
        if len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def get_blocks(self, request: "Request") -> dict[int, list[KVCacheBlock]] | None:
        """Return, per KV cache group, the cached blocks of the request's
        session that hold the start of the request, or None if it does not
        continue a session."""
        if not request.session_id or not request.block_hashes:
            return None
        session = self._sessions.get((request.session_id, request.block_hashes[0]))
        if session is None:
            return None
        found = {}
        for group_id, block_size in self.lookup_block_sizes.items():
            blocks, keys = session.blocks[group_id], session.keys[group_id]
            hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, block_size
            )
            num_blocks = min(len(blocks), len(hashes))
            if num_blocks == 0:
                continue
            # Hashes are chained: a match at the last block implies a match
            # at every block before it.
            last_key = make_block_hash_with_group_id(hashes[num_blocks - 1], group_id)
            if keys[num_blocks - 1] == last_key:
                num_cached = _num_cached(blocks, keys, num_blocks)
                if num_cached < num_blocks:
                    # Evicted since; `register` rebuilds from here.
                    del blocks[num_cached:], keys[num_cached:]
                found[group_id] = blocks[:num_cached]
        return found or None

    def close(self, session_id: str) -> None:
        """Forget a session and stop retaining the blocks it recorded."""
        for session_key in [key for key in self._sessions if key[0] == session_id]:
            session = self._sessions.pop(session_key)
            for group_id, blocks in session.blocks.items():
                # Tail first, so the blocks keep the tail-first eviction order.
                for block, key in zip(
                    reversed(blocks), reversed(session.keys[group_id])
                ):
                    if block.block_hash is not key or not block.retain_until:
                        continue
                    block.retain_until = 0.0
                    pool = block.pool() if block.pool is not None else None
                    if (
                        pool is not None
                        and not block.ref_cnt
                        and block.prev_free_block is not None
                    ):
                        pool.free_block_queue.remove(block)
                        pool.free_block_queue.append(block)

    def expire(self) -> None:
        """Drop sessions past their TTL."""
        now = time.monotonic()
        sessions = self._sessions
        while sessions and next(iter(sessions.values())).expires_at <= now:
            sessions.popitem(last=False)

    def clear(self) -> None:
        """Forget every session after the prefix cache was reset."""
        self._sessions.clear()
