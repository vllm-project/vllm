# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-scoped retention and lookup acceleration for the prefix cache.

A session is the latest request seen for a `session_id`. Its cached blocks
carry a soft reference that orders them last for eviction, and its
full-attention block lists let the next turn's prefix lookup skip the blocks
the two turns share.
"""

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from operator import is_
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
    """The cached blocks of a session's latest request, per KV cache group."""

    # The blocks holding one `session_ref` for this session, and the hash
    # each had when the reference was taken.
    blocks: list[list[KVCacheBlock]]
    keys: list[list[BlockHashWithGroupId]]
    # How many leading `blocks` were cached under the request's hash for
    # their position; 0 for groups without a session lookup.
    num_lookup_blocks: list[int]
    expires_at: float


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
        # Ordered by expiry.
        self._sessions: OrderedDict[str, _Session] = OrderedDict()

    def register(
        self, request: "Request", blocks_by_group: Sequence[Sequence[KVCacheBlock]]
    ) -> None:
        """Make a request's blocks its session's retained prefix. Must be
        called before the blocks are freed."""
        session_id = request.session_id
        if not session_id or not self.ttl:
            return
        old = self._sessions.pop(session_id, None)
        session = _Session([], [], [], time.monotonic() + self.ttl)
        for group_id, blocks in enumerate(blocks_by_group):
            kept: list[KVCacheBlock] = []
            keys: list[BlockHashWithGroupId] = []
            num_lookup_blocks = 0
            if old is not None:
                old_blocks, old_keys = old.blocks[group_id], old.keys[group_id]
                if (
                    len(old_blocks) <= len(blocks)
                    and all(map(is_, old_blocks, blocks))
                    and _num_cached(old_blocks, old_keys) == len(old_blocks)
                ):
                    # The turn extends the previous one: keep its references
                    # and only take new ones for the tail.
                    kept, keys = old_blocks, old_keys
                    num_lookup_blocks = old.num_lookup_blocks[group_id]
                    old.blocks[group_id], old.keys[group_id] = [], []
            for block in blocks[len(kept) :]:
                if block.block_hash is not None:
                    block.session_ref += 1
                    kept.append(block)
                    keys.append(block.block_hash)

            block_size = self.lookup_block_sizes.get(group_id)
            if block_size is not None:
                # A block can also be cached under a partial-prefix hash, so
                # count only blocks cached under the request's hash for
                # their position.
                hashes = resolve_block_hashes(
                    request.block_hashes, self.hash_block_size, block_size
                )
                for key, block_hash in zip(
                    keys[num_lookup_blocks:], hashes[num_lookup_blocks:]
                ):
                    if key != make_block_hash_with_group_id(block_hash, group_id):
                        break
                    num_lookup_blocks += 1

            session.blocks.append(kept)
            session.keys.append(keys)
            session.num_lookup_blocks.append(num_lookup_blocks)

        if old is not None:
            self._release(old)
        self._sessions[session_id] = session
        if len(self._sessions) > self.max_sessions:
            self._release(self._sessions.popitem(last=False)[1])

    def get_blocks(self, request: "Request") -> dict[int, list[KVCacheBlock]] | None:
        """Return, per KV cache group, the cached blocks of the request's
        session that hold the start of the request, or None if it does not
        continue a session."""
        session = self._sessions.get(request.session_id or "")
        if session is None:
            return None
        found = {}
        for group_id, block_size in self.lookup_block_sizes.items():
            blocks, keys = session.blocks[group_id], session.keys[group_id]
            hashes = resolve_block_hashes(
                request.block_hashes, self.hash_block_size, block_size
            )
            num_blocks = min(session.num_lookup_blocks[group_id], len(hashes))
            if num_blocks == 0:
                continue
            # Hashes are chained: a match at the last block implies a match
            # at every block before it.
            last_key = make_block_hash_with_group_id(hashes[num_blocks - 1], group_id)
            if keys[num_blocks - 1] == last_key:
                num_blocks = _num_cached(blocks, keys, num_blocks)
                found[group_id] = blocks[:num_blocks]
        return found or None

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._release(session)

    def expire(self) -> None:
        """Drop sessions past their TTL."""
        now = time.monotonic()
        sessions = self._sessions
        while sessions and next(iter(sessions.values())).expires_at <= now:
            self._release(sessions.popitem(last=False)[1])

    def clear(self) -> None:
        """Forget every session after the prefix cache was reset, which
        dropped their references."""
        self._sessions.clear()

    @staticmethod
    def _release(session: _Session) -> None:
        for blocks, keys in zip(session.blocks, session.keys):
            # Tail first, so demoted blocks keep the tail-first eviction order.
            for block, key in zip(reversed(blocks), reversed(keys)):
                if block.block_hash is not key:
                    continue  # Evicted, which dropped every session's reference.
                block.session_ref -= 1
                if block.session_ref or block.ref_cnt:
                    continue
                # Free and no longer referenced: move behind unreferenced blocks.
                pool = block.pool() if block.pool is not None else None
                if pool is not None and block.prev_free_block is not None:
                    pool.free_block_queue.remove(block)
                    pool.free_block_queue.append(block)
