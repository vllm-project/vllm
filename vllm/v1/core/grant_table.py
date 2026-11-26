# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy GrantTable: the sole record of KV block ownership.

One `GrantTable` governs one `BlockPool`. Every ownership change of a block
(grant, share, publish, release, revoke, drain, quarantine) goes through the
methods of this class; no other code writes its arrays. Every reference to a
block carries a generation (`gen`); a revoke bumps it before the revoke is
sent, so stale references are refused downstream.

Null block (block 0): it is constructed in state `NULL` with gen `NULL_GEN`,
a value no reference ever carries. Ownership and write APIs (`grant`,
`check_writes`, `publish`, `revoke`, `quarantine`, `lease`) refuse it.
Read-side and release APIs (`share`, `release`, `note_use`, `ack`,
`confirm_valid`) skip it silently, since rows legitimately contain null
placeholders (e.g. sliding-window columns).
"""

from collections.abc import Iterable, Sequence
from enum import IntEnum

import numpy as np

NULL_BLOCK_ID = 0
NULL_GEN = 0xFFFFFFFF
NO_OWNER = -1
MAX_READERS = np.iinfo(np.uint16).max
_MAX_GROUPS = 256
_MAX_OWNER = np.iinfo(np.int32).max

BlockIds = Sequence[int] | np.ndarray


class BlockState(IntEnum):
    FREE = 0
    EXCLUSIVE = 1
    SHARED = 2
    CACHED_IDLE = 3
    DRAINING = 4
    QUARANTINED = 5
    NULL = 6


class GrantError(RuntimeError):
    """An ownership operation was refused; no state was changed."""


class WriteGrantRefused(GrantError):
    """A planned step write targets blocks its request may not write (I2).

    The scheduler skips the request for the step; the write is never sent.
    """

    def __init__(self, request_id: str, block_ids: Sequence[int]) -> None:
        self.request_id = request_id
        self.block_ids = list(block_ids)
        super().__init__(
            f"refused write of request {request_id} into blocks {self.block_ids}"
        )


_M64 = (1 << 64) - 1
_SM_GAMMA = 0x9E3779B97F4A7C15
_SM_MUL1 = 0xBF58476D1CE4E5B9
_SM_MUL2 = 0x94D049BB133111EB


def _splitmix64(x: int) -> int:
    z = (x + _SM_GAMMA) & _M64
    z = ((z ^ (z >> 30)) * _SM_MUL1) & _M64
    z = ((z ^ (z >> 27)) * _SM_MUL2) & _M64
    return z ^ (z >> 31)


def row_digest_term(col: int, block_id: int, gen: int) -> int:
    """Digest contribution of one row entry: chained splitmix64, mod 2^64.

    A row digest is the sum of these terms over all columns, mod 2^64, so it
    can be maintained incrementally on appends and revokes.
    """
    h = _splitmix64(col & _M64)
    h = _splitmix64(h ^ (block_id & _M64))
    return _splitmix64(h ^ (gen & 0xFFFFFFFF))


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(_SM_GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_SM_MUL1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_SM_MUL2)
    return z ^ (z >> np.uint64(31))


def row_digest_terms(cols, block_ids, gens) -> np.ndarray:
    """Vectorised `row_digest_term`; returns a uint64 array."""
    c = np.asarray(cols, dtype=np.int64).astype(np.uint64)
    b = np.asarray(block_ids, dtype=np.int64).astype(np.uint64)
    # Accept uint32 gens or their int32 bit pattern (e.g. a GPU mirror).
    g = np.asarray(gens).astype(np.int64).astype(np.uint32).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = _splitmix64_np(c)
        h = _splitmix64_np(h ^ b)
        return _splitmix64_np(h ^ g)


def row_digest(cols, block_ids, gens) -> int:
    """Sum of `row_digest_terms` mod 2^64, as a Python int."""
    terms = row_digest_terms(cols, block_ids, gens)
    with np.errstate(over="ignore"):
        return int(np.sum(terms, dtype=np.uint64))


def normalize_gen_updates(
    block_ids, gens, num_blocks: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Worker-side form of one message's `gen_updates` (from
    `take_gen_updates`), shared by every generation mirror on a worker.

    Drops entries for the null block (its gen is reserved and never changes)
    and out-of-range ids, and keeps only the last gen of a repeated id, so
    each mirror applies the same deterministic update (a device scatter with
    duplicate indices is not deterministic). Accepts u32 gens or their int32
    bit pattern.

    Returns:
        (block_ids as int64, gens as uint32, number of entries dropped).

    """
    ids = np.asarray(block_ids, dtype=np.int64).reshape(-1)
    g = np.asarray(gens).astype(np.int64).astype(np.uint32).reshape(-1)
    if ids.size != g.size:
        raise ValueError("gen_updates: block_ids and gens differ in length")
    keep = (ids != NULL_BLOCK_ID) & (ids >= 0) & (ids < num_blocks)
    dropped = int(ids.size - np.count_nonzero(keep))
    ids, g = ids[keep], g[keep]
    if ids.size > 1:
        _, idx = np.unique(ids[::-1], return_index=True)
        last = np.sort(ids.size - 1 - idx)
        ids, g = ids[last], g[last]
    return ids, g, dropped


class GrantTable:
    """Per-block ownership state for one BlockPool, as parallel numpy arrays.

    Args:
        num_blocks: Number of blocks in the governed pool.
        group_block_sizes: Tokens per block for each KV cache group. A table
            with a single entry applies that size to every group.
        enable_caching: Whether blocks may be published to the prefix cache.

    """

    def __init__(
        self,
        num_blocks: int,
        group_block_sizes: Sequence[int],
        enable_caching: bool = True,
    ) -> None:
        assert num_blocks > 0
        assert 0 < len(group_block_sizes) <= _MAX_GROUPS
        assert all(s > 0 for s in group_block_sizes)
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        # Indexed by group id. A single size applies to every group until a
        # manager registers its own (`set_group_block_size`).
        self._block_sizes = np.zeros(_MAX_GROUPS, dtype=np.uint32)
        if len(group_block_sizes) == 1:
            self._block_sizes[:] = group_block_sizes[0]
        else:
            self._block_sizes[: len(group_block_sizes)] = group_block_sizes

        self._gen = np.ones(num_blocks, dtype=np.uint32)
        self._state = np.full(num_blocks, BlockState.FREE, dtype=np.uint8)
        self._group = np.zeros(num_blocks, dtype=np.uint8)
        self._domain = np.zeros(num_blocks, dtype=np.uint32)
        self._owner = np.full(num_blocks, NO_OWNER, dtype=np.int32)
        self._readers = np.zeros(num_blocks, dtype=np.uint16)
        self._valid = np.zeros(num_blocks, dtype=np.uint32)
        self._last_use = np.zeros(num_blocks, dtype=np.uint64)
        self._pending_write_acks = np.zeros(num_blocks, dtype=np.uint16)
        self._pending_read_acks = np.zeros(num_blocks, dtype=np.uint16)

        self._gen[NULL_BLOCK_ID] = NULL_GEN
        self._state[NULL_BLOCK_ID] = BlockState.NULL

        # Insertion-ordered set of DRAINING blocks, so drain() is O(draining).
        self._draining: dict[int, None] = {}
        # Gen changes since the last take_gen_updates(), last value wins.
        self._gen_updates: dict[int, int] = {}
        self._sched_seq = 0
        self._processed_seq = 0
        self._owner_ids: dict[str, int] = {}
        self._domains: dict[str, int] = {}
        self._next_owner = 0

    # ------------------------------------------------------------------ utils
    def _ids(self, block_ids: BlockIds) -> np.ndarray:
        ids = np.asarray(block_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_blocks):
            raise ValueError(f"block id out of range [0, {self.num_blocks})")
        return ids

    def _non_null(self, block_ids: BlockIds) -> np.ndarray:
        ids = self._ids(block_ids)
        return ids[ids != NULL_BLOCK_ID]

    def _refuse_null(self, ids: np.ndarray, op: str) -> None:
        if np.any(ids == NULL_BLOCK_ID):
            raise GrantError(f"{op}: the null block is never granted")

    def _sizes_of(self, groups: np.ndarray) -> np.ndarray:
        return self._block_sizes[groups]

    def block_size_of(self, group: int) -> int:
        return int(self._block_sizes[group])

    def set_group_block_size(self, group: int, block_size: int) -> None:
        """Register the tokens per block of a group (its managers' size)."""
        if not 0 <= group < _MAX_GROUPS or block_size <= 0:
            raise ValueError(f"invalid group {group} or block size {block_size}")
        self._block_sizes[group] = block_size

    def _bump_gen(self, ids: np.ndarray) -> None:
        new = self._gen[ids].astype(np.uint64) + np.uint64(1)
        new[new >= np.uint64(NULL_GEN)] = 1
        self._gen[ids] = new.astype(np.uint32)
        for block_id, gen in zip(ids.tolist(), self._gen[ids].tolist()):
            self._gen_updates[block_id] = gen

    def _stamp(self, ids: np.ndarray, step_seq: int) -> None:
        if ids.size:
            self._last_use[ids] = np.maximum(self._last_use[ids], np.uint64(step_seq))

    def _enter_draining(self, ids: np.ndarray, step_seq: int) -> None:
        """Revoke: bump gen FIRST, then enter DRAINING (I4)."""
        self._bump_gen(ids)
        self._state[ids] = BlockState.DRAINING
        self._owner[ids] = NO_OWNER
        self._valid[ids] = 0
        self._stamp(ids, step_seq)
        self._draining.update(dict.fromkeys(ids.tolist()))

    # ---------------------------------------------------------------- owners
    def owner_for(self, request_id: str) -> int:
        """Return the owner slot of a request, interning a new one if needed.

        Owner ids are monotonic and never recycled, so a stale owner id can
        never pass `check_writes` on a block granted to a later request.
        """
        owner = self._owner_ids.get(request_id)
        if owner is None:
            if self._next_owner > _MAX_OWNER:
                raise GrantError("owner id space exhausted")
            owner = self._next_owner
            self._next_owner += 1
            self._owner_ids[request_id] = owner
        return owner

    def forget_owner(self, request_id: str) -> None:
        self._owner_ids.pop(request_id, None)
        self._domains.pop(request_id, None)

    def set_domain(self, request_id: str, domain: int) -> None:
        """Record the sharing domain (I9) of a request's blocks."""
        if not 0 <= domain <= 0xFFFFFFFF:
            raise ValueError(f"invalid domain {domain}")
        if domain:
            self._domains[request_id] = domain
        else:
            self._domains.pop(request_id, None)

    def domain_for(self, request_id: str) -> int:
        return self._domains.get(request_id, 0)

    # ------------------------------------------------------------- ownership
    def grant(
        self, block_ids: BlockIds, owner: int, group: int, domain: int
    ) -> np.ndarray:
        """T1: grant FREE blocks for writing to `owner`; returns their gens."""
        ids = self._ids(block_ids)
        self._refuse_null(ids, "grant")
        if not 0 <= owner <= _MAX_OWNER:
            raise GrantError(f"grant: invalid owner {owner}")
        if not 0 <= group < _MAX_GROUPS or self._block_sizes[group] == 0:
            raise GrantError(f"grant: invalid group {group}")
        if not 0 <= domain <= 0xFFFFFFFF:
            raise GrantError(f"grant: invalid domain {domain}")
        if np.unique(ids).size != ids.size:
            raise GrantError("grant: duplicate block ids")
        if np.any(self._state[ids] != BlockState.FREE):
            raise GrantError("grant: block is not FREE")
        self._state[ids] = BlockState.EXCLUSIVE
        self._owner[ids] = owner
        self._group[ids] = group
        self._domain[ids] = domain
        self._readers[ids] = 1
        self._valid[ids] = 0
        self._pending_write_acks[ids] = 0
        self._pending_read_acks[ids] = 0
        return self._gen[ids].copy()

    def check_writes(self, owner: int, block_ids: BlockIds, gens) -> bool:
        """True iff every block is EXCLUSIVE, owned by `owner`, at `gens`."""
        return self.find_refused_writes(owner, block_ids, gens).size == 0

    def find_refused_writes(self, owner: int, block_ids: BlockIds, gens) -> np.ndarray:
        """Block ids of `block_ids` that `owner` may not write at `gens`."""
        ids = self._ids(block_ids)
        g = np.asarray(gens).astype(np.int64).astype(np.uint32).reshape(-1)
        if g.size != ids.size:
            raise ValueError("check_writes: block_ids and gens differ in length")
        ok = (
            (self._state[ids] == BlockState.EXCLUSIVE)
            & (self._owner[ids] == owner)
            & (self._gen[ids] == g)
        )
        return ids[~ok]

    def share(
        self,
        block_ids: BlockIds,
        group: int | None = None,
        domain: int | None = None,
    ) -> bool:
        """T3/T5: add a reader. SHARED: readers += 1; CACHED_IDLE -> SHARED;
        EXCLUSIVE: an in-engine read pin, state and owner unchanged.

        `group`/`domain` None means an internal pin and skips the match.
        All-or-nothing: returns False without mutating on any refusal.
        """
        ids = self._non_null(block_ids)
        if ids.size == 0:
            return True
        state = self._state[ids]
        if not np.all(
            (state == BlockState.SHARED)
            | (state == BlockState.CACHED_IDLE)
            | (state == BlockState.EXCLUSIVE)
        ):
            return False
        if group is not None and np.any(self._group[ids] != group):
            return False
        if domain is not None and np.any(self._domain[ids] != domain):
            return False
        uniq, counts = np.unique(ids, return_counts=True)
        if np.any(self._readers[uniq].astype(np.int64) + counts > MAX_READERS):
            return False
        idle = uniq[self._state[uniq] == BlockState.CACHED_IDLE]
        self._state[idle] = BlockState.SHARED
        self._readers[uniq] += counts.astype(np.uint16)
        return True

    def publish(self, block_ids: BlockIds) -> bool:
        """T2: freeze full blocks for the prefix cache (I3).

        Requires caching enabled and every block EXCLUSIVE (or already
        SHARED) with valid == block_size. False, without mutation, otherwise.
        """
        ids = self._ids(block_ids)
        if ids.size == 0:
            return True
        if not self.enable_caching or np.any(ids == NULL_BLOCK_ID):
            return False
        state = self._state[ids]
        if not np.all((state == BlockState.EXCLUSIVE) | (state == BlockState.SHARED)):
            return False
        if np.any(self._valid[ids] != self._sizes_of(self._group[ids])):
            return False
        self._state[ids] = BlockState.SHARED
        self._owner[ids] = NO_OWNER
        return True

    def publish_partial(self, block_ids: BlockIds, num_tokens: int) -> bool:
        """Freeze blocks whose first `num_tokens` slots back a partial
        prefix-cache entry (a scoped relaxation of I3: readers of a partial
        entry read only those slots, and they are confirmed).

        Requires caching enabled, every block EXCLUSIVE or SHARED, no other
        writer (readers <= 1) and valid >= `num_tokens`. The caller seals a
        block this way only when its last holder releases it, so no writer
        remains. False, without mutation, otherwise.
        """
        ids = self._ids(block_ids)
        if ids.size == 0:
            return True
        if not self.enable_caching or np.any(ids == NULL_BLOCK_ID):
            return False
        state = self._state[ids]
        if not np.all((state == BlockState.EXCLUSIVE) | (state == BlockState.SHARED)):
            return False
        if np.any(self._readers[ids] > 1) or np.any(self._valid[ids] < num_tokens):
            return False
        self._state[ids] = BlockState.SHARED
        self._owner[ids] = NO_OWNER
        return True

    def confirm_valid(self, block_ids: BlockIds, valid) -> None:
        """Ack path (I8): raise confirmed slots; never lowers, caps at size.

        Only EXCLUSIVE and SHARED blocks take acks; a late ack for a revoked
        or free block is ignored.
        """
        ids = self._ids(block_ids)
        v = np.broadcast_to(np.asarray(valid, dtype=np.int64), ids.shape)
        if np.any(v < 0):
            raise ValueError("confirm_valid: negative valid")
        keep = (ids != NULL_BLOCK_ID) & (
            (self._state[ids] == BlockState.EXCLUSIVE)
            | (self._state[ids] == BlockState.SHARED)
        )
        ids, v = ids[keep], v[keep]
        if ids.size == 0:
            return
        capped = np.minimum(v, self._sizes_of(self._group[ids]).astype(np.int64))
        # np.maximum.at handles duplicate ids correctly.
        np.maximum.at(self._valid, ids, capped.astype(np.uint32))

    def release(self, block_ids: BlockIds, step_seq: int) -> None:
        """Drop one reader per occurrence; stamp last_use with `step_seq`.

        At readers == 0: SHARED -> CACHED_IDLE (T4); EXCLUSIVE -> revoke
        (T6: gen+1, DRAINING); DRAINING and QUARANTINED stay (and may then
        drain or be reclaimed).
        """
        ids = self._non_null(block_ids)
        if ids.size == 0:
            return
        uniq, counts = np.unique(ids, return_counts=True)
        if np.any(self._readers[uniq].astype(np.int64) < counts):
            raise GrantError("release: block has fewer readers than releases")
        state = self._state[uniq]
        if np.any(
            (state != BlockState.EXCLUSIVE)
            & (state != BlockState.SHARED)
            & (state != BlockState.DRAINING)
            & (state != BlockState.QUARANTINED)
        ):
            raise GrantError("release: block is not held")
        self._readers[uniq] -= counts.astype(np.uint16)
        self._stamp(uniq, step_seq)
        done = uniq[self._readers[uniq] == 0]
        state = self._state[done]
        self._state[done[state == BlockState.SHARED]] = BlockState.CACHED_IDLE
        exclusive = done[state == BlockState.EXCLUSIVE]
        if exclusive.size:
            self._enter_draining(exclusive, step_seq)

    def revoke(self, block_ids: BlockIds, step_seq: int) -> None:
        """T6/T7/T8: bump gen, then enter DRAINING. Readers are kept; a block
        drains only once every holder has released it."""
        ids = self._ids(block_ids)
        self._refuse_null(ids, "revoke")
        ids = np.unique(ids)
        state = self._state[ids]
        if not np.all(
            (state == BlockState.EXCLUSIVE)
            | (state == BlockState.SHARED)
            | (state == BlockState.CACHED_IDLE)
        ):
            raise GrantError("revoke: block is not owned, shared or cached")
        self._enter_draining(ids, step_seq)

    def note_use(self, block_ids: BlockIds, step_seq: int) -> None:
        """Record that step `step_seq` may read or write these blocks."""
        self._stamp(self._non_null(block_ids), step_seq)

    def begin_step(self, sched_seq: int) -> None:
        """Record the latest scheduled step seq (monotonic)."""
        if sched_seq < self._sched_seq:
            raise ValueError("begin_step: sched seq went backwards")
        self._sched_seq = sched_seq

    def on_step_done(
        self,
        processed_seq: int,
        accepted: tuple[BlockIds, object] | None = None,
    ) -> np.ndarray:
        """A step's output was processed: apply its acks
        (`accepted` = (block_ids, valid)) and return drained blocks."""
        if accepted is not None:
            self.confirm_valid(*accepted)
        return self.drain(processed_seq)

    def lease(self, block_ids: BlockIds, kind: str) -> None:
        """Record a peer transfer on our blocks until `ack`."""
        ids = self._ids(block_ids)
        self._refuse_null(ids, "lease")
        if np.any(
            (self._state[ids] == BlockState.FREE)
            | (self._state[ids] == BlockState.QUARANTINED)
        ):
            raise GrantError("lease: block is not held")
        counter = self._lease_counter(kind)
        uniq, counts = np.unique(ids, return_counts=True)
        if np.any(counter[uniq].astype(np.int64) + counts > MAX_READERS):
            raise GrantError("lease: too many pending acks")
        counter[uniq] += counts.astype(np.uint16)

    def ack(self, block_ids: BlockIds, kind: str) -> None:
        """Retire one pending lease per occurrence."""
        ids = self._non_null(block_ids)
        counter = self._lease_counter(kind)
        uniq, counts = np.unique(ids, return_counts=True)
        if np.any(counter[uniq].astype(np.int64) < counts):
            raise GrantError("ack: no pending lease")
        counter[uniq] -= counts.astype(np.uint16)

    def expire_read_leases(self, block_ids: BlockIds) -> None:
        """Read leases may time out: readers re-check the gen after reading."""
        self._pending_read_acks[self._non_null(block_ids)] = 0

    def _lease_counter(self, kind: str) -> np.ndarray:
        if kind == "read":
            return self._pending_read_acks
        if kind == "write":
            return self._pending_write_acks
        raise ValueError(f"unknown lease kind {kind!r}")

    def drain(self, processed_seq: int | None = None) -> np.ndarray:
        """T9: move DRAINING blocks that no step or peer can still touch to
        FREE (I5). Returns their ids, in revoke order, for the free queue."""
        if processed_seq is not None:
            if processed_seq < self._processed_seq:
                raise ValueError("drain: processed seq went backwards")
            self._processed_seq = processed_seq
        if not self._draining:
            return np.empty(0, dtype=np.int64)
        ids = np.fromiter(self._draining, dtype=np.int64, count=len(self._draining))
        ok = (
            (self._readers[ids] == 0)
            & (self._last_use[ids] <= np.uint64(self._processed_seq))
            & (self._pending_write_acks[ids] == 0)
            & (self._pending_read_acks[ids] == 0)
        )
        done = ids[ok]
        if done.size == 0:
            return done
        for block_id in done.tolist():
            del self._draining[block_id]
        self._make_free(done)
        return done

    def quarantine(self, block_ids: BlockIds) -> None:
        """T10: a DRAINING block whose write lease outlived its timeout is
        never reused until `reclaim_quarantined`."""
        ids = self._ids(block_ids)
        self._refuse_null(ids, "quarantine")
        ids = np.unique(ids)
        if np.any(self._state[ids] != BlockState.DRAINING):
            raise GrantError("quarantine: block is not DRAINING")
        for block_id in ids.tolist():
            del self._draining[block_id]
        self._state[ids] = BlockState.QUARANTINED

    def reclaim_quarantined(self, block_ids: BlockIds) -> np.ndarray:
        """T11: the transport confirmed no operation is in flight. Blocks
        still held by a row are refused until released."""
        ids = np.unique(self._ids(block_ids))
        if np.any(self._state[ids] != BlockState.QUARANTINED):
            raise GrantError("reclaim: block is not QUARANTINED")
        if np.any(self._readers[ids] != 0):
            raise GrantError("reclaim: block is still held")
        self._pending_write_acks[ids] = 0
        self._pending_read_acks[ids] = 0
        self._make_free(ids)
        return ids

    def _make_free(self, ids: np.ndarray) -> None:
        self._state[ids] = BlockState.FREE
        self._owner[ids] = NO_OWNER
        self._group[ids] = 0
        self._domain[ids] = 0
        self._valid[ids] = 0

    def take_gen_updates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (block_ids, gens) changed since the last call, and clear."""
        updates = self._gen_updates
        self._gen_updates = {}
        ids = np.fromiter(updates.keys(), dtype=np.int64, count=len(updates))
        gens = np.fromiter(updates.values(), dtype=np.uint32, count=len(updates))
        return ids, gens

    # ------------------------------------------------------------- accessors
    def gens_of(self, block_ids: BlockIds) -> np.ndarray:
        return self._gen[self._ids(block_ids)].copy()

    def gen(self, block_id: int) -> int:
        return int(self._gen[block_id])

    def state_of(self, block_ids: BlockIds) -> np.ndarray:
        return self._state[self._ids(block_ids)].copy()

    def state(self, block_id: int) -> BlockState:
        return BlockState(int(self._state[block_id]))

    def is_writable(self, block_id: int) -> bool:
        return int(self._state[block_id]) == BlockState.EXCLUSIVE

    def owner(self, block_id: int) -> int:
        return int(self._owner[block_id])

    def readers(self, block_id: int) -> int:
        return int(self._readers[block_id])

    def valid(self, block_id: int) -> int:
        return int(self._valid[block_id])

    def group(self, block_id: int) -> int:
        return int(self._group[block_id])

    def domain(self, block_id: int) -> int:
        return int(self._domain[block_id])

    def last_use(self, block_id: int) -> int:
        return int(self._last_use[block_id])

    def pending_acks(self, block_id: int) -> tuple[int, int]:
        """(pending_write_acks, pending_read_acks)."""
        return (
            int(self._pending_write_acks[block_id]),
            int(self._pending_read_acks[block_id]),
        )

    def publishable_mask(self, block_ids: BlockIds) -> np.ndarray:
        """Whether `publish` would accept each block on its own."""
        ids = self._ids(block_ids)
        state = self._state[ids]
        return (
            np.bool_(self.enable_caching)
            & (ids != NULL_BLOCK_ID)
            & ((state == BlockState.EXCLUSIVE) | (state == BlockState.SHARED))
            & (self._valid[ids] == self._sizes_of(self._group[ids]))
        )

    def backs_partial_entry(self, block_id: int, num_slots: int) -> bool:
        """Whether a block's first `num_slots` slots are confirmed, so a
        partial prefix-cache entry may point at them. Cached-idle blocks keep
        their confirmed slots, so they may take extra lookup entries too."""
        state = self._state[block_id]
        return (
            self.enable_caching
            and block_id != NULL_BLOCK_ID
            and state
            in (BlockState.EXCLUSIVE, BlockState.SHARED, BlockState.CACHED_IDLE)
            and int(self._valid[block_id]) >= num_slots
        )

    def shareable(self, block_ids: BlockIds, group: int, domain: int) -> bool:
        """Whether `share(block_ids, group, domain)` would accept, without
        mutating (a prefix lookup checks this before committing to a hit)."""
        ids = self._non_null(block_ids)
        state = self._state[ids]
        return bool(
            np.all(
                (state == BlockState.SHARED)
                | (state == BlockState.CACHED_IDLE)
                | (state == BlockState.EXCLUSIVE)
            )
            and np.all(self._group[ids] == group)
            and np.all(self._domain[ids] == domain)
        )

    def readers_of(self, block_ids: BlockIds) -> np.ndarray:
        return self._readers[self._ids(block_ids)].copy()

    def drainable_mask(self, block_ids: BlockIds) -> np.ndarray:
        """Whether each block could leave DRAINING now if it were revoked."""
        ids = self._ids(block_ids)
        return (
            (self._readers[ids] == 0)
            & (self._last_use[ids] <= np.uint64(self._processed_seq))
            & (self._pending_write_acks[ids] == 0)
            & (self._pending_read_acks[ids] == 0)
        )

    @property
    def num_draining(self) -> int:
        return len(self._draining)

    @property
    def num_held(self) -> int:
        """Blocks some holder (a request row or an in-engine pin) still has."""
        return int(np.count_nonzero(self._readers))

    def draining_ids(self) -> Iterable[int]:
        return tuple(self._draining)

    def count_in_state(self, state: BlockState) -> int:
        return int(np.count_nonzero(self._state == state))

    @property
    def latest_sched_seq(self) -> int:
        return self._sched_seq

    @property
    def processed_seq(self) -> int:
        return self._processed_seq

    @property
    def host_bytes_per_block(self) -> int:
        arrays = (
            self._gen,
            self._state,
            self._group,
            self._domain,
            self._owner,
            self._readers,
            self._valid,
            self._last_use,
            self._pending_write_acks,
            self._pending_read_acks,
        )
        return sum(a.itemsize for a in arrays)
