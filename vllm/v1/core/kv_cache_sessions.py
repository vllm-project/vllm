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
from operator import is_

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
    resolve_block_hashes,
)


@dataclass(slots=True)
class SessionPrefix:
    """The cached blocks of a session's latest request. Read-only outside
    `KVCacheSessions`."""

    # Per KV cache group, the blocks holding one `session_ref` for this
    # session and the hash each had when the reference was taken.
    blocks: list[list[KVCacheBlock]]
    keys: list[list[BlockHashWithGroupId]]
    # Per lookup group, how many leading `blocks` are still cached under
    # the request's hash for their position.
    num_lookup_blocks: dict[int, int]
    # `KVCacheSessions.epoch` at registration and at the last check of
    # `num_lookup_blocks`.
    epoch: int
    lookup_epoch: int
    expires_at: float

    def lookup(
        self,
        computed_blocks: Sequence[list[KVCacheBlock]],
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: Sequence[int],
        max_num_blocks: int,
    ) -> int:
        """If `block_hashes` continues the session, extend each group's
        `computed_blocks` with the session's blocks and return how many."""
        num_blocks = min(
            max_num_blocks,
            len(block_hashes),
            *(self.num_lookup_blocks.get(g, 0) for g in kv_cache_group_ids),
        )
        if num_blocks == 0:
            return 0
        group_id = kv_cache_group_ids[0]
        last_key = make_block_hash_with_group_id(block_hashes[num_blocks - 1], group_id)
        # Hashes are chained: a match at the last block implies a match at
        # every block before it.
        if self.keys[group_id][num_blocks - 1] != last_key:
            return 0
        for computed, group_id in zip(computed_blocks, kv_cache_group_ids):
            computed.extend(self.blocks[group_id][:num_blocks])
        return num_blocks


class KVCacheSessions:
    """Tracks the cached blocks of each live session.

    Args:
        ttl: Seconds a session outlives its latest request.
        max_sessions: Sessions tracked before the oldest is dropped.
        hash_block_size: Token granularity of request block hashes.
        lookup_block_sizes: Block size of each KV cache group whose lookup
            scans from the first block and so can start from a session's blocks.

    """

    def __init__(
        self,
        ttl: float,
        max_sessions: int,
        hash_block_size: int,
        lookup_block_sizes: dict[int, int],
    ) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.hash_block_size = hash_block_size
        self.lookup_block_sizes = lookup_block_sizes
        # Ordered by expiry.
        self._sessions: OrderedDict[str, SessionPrefix] = OrderedDict()
        # Bumped by BlockPool whenever a session-referenced block loses its
        # hash; a session registered at the current epoch is intact.
        self.epoch = 0

    def register(
        self,
        session_id: str,
        block_hashes: Sequence[BlockHash],
        blocks_by_group: Sequence[Sequence[KVCacheBlock]],
    ) -> None:
        """Make a request's blocks the session's retained prefix. Must be
        called before the blocks are freed."""
        old = self._sessions.pop(session_id, None)
        intact = old is not None and old.epoch == self.epoch
        session = SessionPrefix(
            blocks=[],
            keys=[],
            num_lookup_blocks={},
            epoch=self.epoch,
            lookup_epoch=self.epoch,
            expires_at=time.monotonic() + self.ttl,
        )
        for group_id, blocks in enumerate(blocks_by_group):
            kept: list[KVCacheBlock] = []
            keys: list[BlockHashWithGroupId] = []
            num_lookup_blocks = 0
            if intact:
                assert old is not None
                old_blocks = old.blocks[group_id]
                if len(old_blocks) <= len(blocks) and all(map(is_, old_blocks, blocks)):
                    # The turn extends the previous one: keep its references
                    # and only take new ones for the tail.
                    kept, keys = old_blocks, old.keys[group_id]
                    num_lookup_blocks = old.num_lookup_blocks.get(group_id, 0)
                    old.blocks[group_id] = old.keys[group_id] = []
            num_kept = len(kept)
            for block in blocks[num_kept:]:
                if block.block_hash is not None:
                    block.session_ref += 1
                    kept.append(block)
                    keys.append(block.block_hash)
            session.blocks.append(kept)
            session.keys.append(keys)

            block_size = self.lookup_block_sizes.get(group_id)
            if block_size is None:
                continue
            if num_lookup_blocks == num_kept:
                # A block may also be cached under secondary hashes, so check
                # that its own hash is the request's hash for its position.
                hashes = resolve_block_hashes(
                    block_hashes, self.hash_block_size, block_size
                )
                end = min(len(kept), len(hashes))
                while num_lookup_blocks < end and keys[
                    num_lookup_blocks
                ] == make_block_hash_with_group_id(hashes[num_lookup_blocks], group_id):
                    num_lookup_blocks += 1
            session.num_lookup_blocks[group_id] = num_lookup_blocks

        if old is not None:
            self._release(old)
        self._sessions[session_id] = session
        if len(self._sessions) > self.max_sessions:
            self._release(self._sessions.popitem(last=False)[1])

    def get_prefix(self, session_id: str) -> SessionPrefix | None:
        """Return the session's latest request's blocks, with `num_lookup_blocks`
        covering only blocks that are still cached."""
        session = self._sessions.get(session_id)
        if session is not None and session.lookup_epoch != self.epoch:
            for group_id, num_lookup_blocks in session.num_lookup_blocks.items():
                blocks, keys = session.blocks[group_id], session.keys[group_id]
                for i in range(num_lookup_blocks):
                    if blocks[i].block_hash is not keys[i]:
                        session.num_lookup_blocks[group_id] = i
                        break
            session.lookup_epoch = self.epoch
        return session

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
    def _release(session: SessionPrefix) -> None:
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
