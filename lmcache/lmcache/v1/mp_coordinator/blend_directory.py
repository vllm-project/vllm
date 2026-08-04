# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide CacheBlend fingerprint directory held by the MP coordinator.

Blend mp-servers publish stored token ranges on STORE and query with the request
tokens on LOOKUP; the coordinator does **all hashing and matching**, so the whole
matching algorithm lives here and can evolve without touching or redeploying
servers. Servers send raw tokens plus the storage mapping they alone know
(``object_key`` per chunk, base token position).

The algorithm matches the local matcher (`BlendTokenRangeMatcherV3`): a table of
non-overlapping chunk polynomial hashes, probed by a strided rolling-hash scan of
the request.

- **Match unit = chunk_size** (fleet config). Each stored chunk is one poly hash.
- **Probe stride** (the querying server's inference block size) controls which
  request offsets can seed a match; sent per query, so servers with different
  (per-machine, dynamic) block sizes interoperate.
- **Scope** = the model name; cross-model content never matches. ``cache_salt``
  is **not** part of the scope: matched hashes are expanded to ObjectKeys with
  the *requester's* salt at retrieve (``ipc_key_to_object_keys``), so a
  cross-salt match simply misses at L2 -- isolation holds with one table per
  model instead of one per ``(model, salt)``.
- **TP rank** is resolved at retrieve (``ipc_key_to_object_keys``), not here.

``object_key`` is the chunk's shared-L2 storage key (``th``), which is
prefix-bound and computed by the storing server -- the coordinator cannot derive
it, so it is supplied with each published range.

Concurrency: fingerprints are partitioned per scope (``_ScopeTable``), each
mutated and probed in place under its own lock; a top-level lock guards only the
scope map and the eviction map. The probe is vectorized, so the locked section
is short.

The probe mirrors the local matcher: the strided rolling hashes index a
per-scope **direct-address table** in one numpy gather, and only occupied slots
reach a short Python loop, where a full-64-bit re-check rejects bucket
collisions. Inserts write the table in place (O(1) per chunk); evictions
tombstone; a rebuild -- which also compacts tombstones -- happens only on the
write path, when the table outgrows its load factor. Tables are sized per scope
(power of two, a few times the entry count), so small scopes stay small, unlike
the local matcher's fixed 2^20 array.

Thread-safe and ephemeral. A stale entry or a (rare) bucket collision only costs
a wasted prefetch or a missed reuse, recomputed downstream -- never wrong KV.

See ``docs/design/v1/mp_coordinator/blend_lookup.md``.
"""

# Standard
from collections import defaultdict
from dataclasses import dataclass
import threading

# Third Party
import numpy as np

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    update_table_id_numba,
)

logger = init_logger(__name__)

# Fleet-constant polynomial base for blend fingerprints. The coordinator owns
# the hashing, so this lives here: the same base hashes published chunks
# (``chunk_hash_windows_numba``) and probes requests (``rolling_hash_windows_numba``)
# so the two align. Constant across a coordinator's lifetime.
POLY_BASE = np.uint64(0x9E3779B97F4A7C15)

# Table size = smallest power of two >= _TABLE_GROWTH * live entries, keeping
# the load factor (and thus bucket-collision recall loss) low for a few
# bytes/chunk.
_TABLE_GROWTH = 4
_MIN_TABLE_SIZE = 1 << 10


@dataclass
class StoreRange:
    """One stored token range published by a blend server.

    Attributes:
        model_scope: Reuse-compatibility scope (the model name).
        tokens: The stored tokens (``token_ids[start:end]``). The coordinator
            chunks these at ``chunk_size`` and hashes each chunk.
        object_keys: Shared-L2 storage key (hex of the ObjectKey chunk hash) per
            chunk, in order; chunk ``i`` maps to ``object_keys[i]``.
        old_st_base: Token position of the range's first token in the stored
            sequence; chunk ``i`` starts at ``old_st_base + i * chunk_size``.
    """

    model_scope: str
    tokens: list[int]
    object_keys: list[str]
    old_st_base: int


@dataclass
class _ChunkLoc:
    """Where a registered chunk lives (anchor index value)."""

    object_key: str
    old_st: int


@dataclass
class GlobalMatch:
    """One matched chunk returned to a querying server.

    Attributes:
        object_key: Shared-L2 storage key of the matched chunk.
        old_st: Token position of the chunk in the stored sequence (re-RoPE).
        cur_st: Token position in the request where the match was found.
    """

    object_key: str
    old_st: int
    cur_st: int


class _ScopeTable:
    """One scope's fingerprint table, mutated in place under ``lock``.

    Mirrors the local matcher: ``slots`` direct-addresses the low bits of a poly
    hash to a compact entry id (``-1`` = empty, last writer wins on collision);
    ``hashes[cid]`` / ``locs[cid]`` hold the full hash (collision rejection) and
    chunk location. ``poly_to_cid`` gives idempotent insert and eviction lookup.
    Eviction tombstones ``locs[cid]``; a rebuild (on load-factor growth, or when
    tombstones outnumber live entries) re-creates ``slots`` compacted.

    All methods require the caller to hold ``lock``.
    """

    def __init__(self) -> None:
        """Initialize an empty table at the minimum size."""
        self.lock = threading.Lock()
        self.poly_to_cid: dict[int, int] = {}
        self.hashes: list[int] = []
        self.locs: "list[_ChunkLoc | None]" = []
        self.slots: np.ndarray = np.full(_MIN_TABLE_SIZE, -1, dtype=np.int64)
        self.mask: np.uint64 = np.uint64(_MIN_TABLE_SIZE - 1)

    def insert(self, poly: int, loc: _ChunkLoc) -> bool:
        """Insert one fingerprint, growing the table when needed.

        Args:
            poly: The chunk's full 64-bit poly hash.
            loc: The chunk's storage location.

        Returns:
            ``True`` if newly inserted, ``False`` if the hash was present.
        """
        if poly in self.poly_to_cid:
            return False
        cid = len(self.hashes)
        self.hashes.append(poly)
        self.locs.append(loc)
        self.poly_to_cid[poly] = cid
        self.slots[poly & int(self.mask)] = cid
        if _TABLE_GROWTH * len(self.poly_to_cid) > self.slots.shape[0]:
            self._rebuild()
        return True

    def evict(self, poly: int) -> bool:
        """Tombstone one fingerprint, compacting when tombstones dominate.

        Args:
            poly: The chunk's full 64-bit poly hash.

        Returns:
            ``True`` if evicted, ``False`` if the hash was absent.
        """
        cid = self.poly_to_cid.pop(poly, None)
        if cid is None:
            return False
        slot = poly & int(self.mask)
        if self.slots[slot] == cid:  # a colliding later insert may own the slot
            self.slots[slot] = -1
        self.locs[cid] = None
        if len(self.poly_to_cid) < len(self.locs) // 2:
            self._rebuild()
        return True

    def _rebuild(self) -> None:
        """Re-create ``slots`` sized to the live entries, dropping tombstones."""
        live = [(poly, self.locs[cid]) for poly, cid in self.poly_to_cid.items()]
        size = _MIN_TABLE_SIZE
        while size < _TABLE_GROWTH * len(live):
            size <<= 1
        self.hashes = [poly for poly, _ in live]
        self.locs = [loc for _, loc in live]
        self.poly_to_cid = {poly: cid for cid, (poly, _) in enumerate(live)}
        self.slots = np.full(size, -1, dtype=np.int64)
        self.mask = np.uint64(size - 1)
        if live:
            update_table_id_numba(
                np.array(self.hashes, dtype=np.uint64),
                self.slots,
                np.arange(len(live), dtype=np.int64),
            )


class GlobalBlendMatcher:
    """Thread-safe fleet-wide chunk fingerprint directory.

    Hashes published token ranges into per-scope direct-address tables, mutated
    in place, and matches request tokens with a vectorized strided rolling-hash
    probe under the scope's lock.
    """

    def __init__(self, chunk_size: int = 256, probe_stride: int = 1) -> None:
        """Initialize an empty directory.

        Args:
            chunk_size: Tokens per chunk (the LMCache chunk size; fleet config).
                The match unit; must be the same value the storing servers use.
            probe_stride: Positions between match probes. With partial-fill reuse
                any offset is usable, so the default ``1`` (probe every offset)
                gives full recall; raise only to trade recall for coordinator CPU.

        Raises:
            ValueError: If ``chunk_size`` or ``probe_stride`` is not positive.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if probe_stride < 1:
            raise ValueError(f"probe_stride must be >= 1, got {probe_stride}")
        self._chunk_size = chunk_size
        self._probe_stride = probe_stride
        # Top-level lock: guards _scopes and _by_key only (cheap, short holds).
        self._lock = threading.Lock()
        self._scopes: dict[str, _ScopeTable] = {}
        # Reverse map for eviction: object_key -> its (scope, poly_hash) keys.
        self._by_key: dict[str, list[tuple[str, int]]] = {}

    def _get_or_create_scope(self, model_scope: str) -> _ScopeTable:
        """Return the table for ``model_scope``, creating it if absent.

        Args:
            model_scope: The reuse scope to fetch a table for.

        Returns:
            The (possibly newly created) per-scope table.
        """
        with self._lock:
            table = self._scopes.get(model_scope)
            if table is None:
                table = _ScopeTable()
                self._scopes[model_scope] = table
            return table

    def register(self, ranges: list[StoreRange]) -> int:
        """Hash and insert published token ranges (idempotent per chunk).

        The coordinator chunks each range's tokens, hashes each chunk, and maps
        the poly hash to its storage key and position. Re-publishing a
        ``(model_scope, poly_hash)`` already present is a no-op.

        A range whose chunk count (``len(tokens) // chunk_size``) does not equal
        its ``object_keys`` count is **skipped** with an error log: the two are
        1:1 by construction, so a mismatch signals a publisher bug or a
        chunk-size disagreement, and registering the aligned prefix would map
        chunks to the wrong storage keys (a shift) -- worse than dropping it.

        Args:
            ranges: Stored token ranges to register.

        Returns:
            Number of chunk fingerprints newly inserted (excludes idempotent
            skips).
        """
        # Hash every range outside any lock; keep only well-formed ranges.
        prepared: list[tuple[str, np.ndarray, list[str], int]] = []
        for rng in ranges:
            arr = np.asarray(rng.tokens, dtype=np.uint64)
            polys = chunk_hash_windows_numba(arr, self._chunk_size, POLY_BASE)
            n_chunks = int(polys.shape[0])
            if n_chunks != len(rng.object_keys):
                logger.error(
                    "blend register: %d chunks from %d tokens (chunk_size=%d) "
                    "but %d object_keys for scope %s; skipping range "
                    "(publisher/chunk_size mismatch)",
                    n_chunks,
                    len(rng.tokens),
                    self._chunk_size,
                    len(rng.object_keys),
                    rng.model_scope,
                )
                continue
            prepared.append((rng.model_scope, polys, rng.object_keys, rng.old_st_base))

        inserted = 0
        for model_scope, polys, object_keys, old_st_base in prepared:
            table = self._get_or_create_scope(model_scope)
            new_keys: list[tuple[str, int]] = []
            with table.lock:
                for i in range(len(object_keys)):
                    poly = int(polys[i])
                    loc = _ChunkLoc(object_keys[i], old_st_base + i * self._chunk_size)
                    if table.insert(poly, loc):
                        new_keys.append((object_keys[i], poly))
            if new_keys:
                inserted += len(new_keys)
                with self._lock:
                    for object_key, poly in new_keys:
                        self._by_key.setdefault(object_key, []).append(
                            (model_scope, poly)
                        )
        return inserted

    def remove(self, object_keys: list[str]) -> int:
        """Evict all fingerprints for the given storage keys.

        Args:
            object_keys: Storage keys of chunks to evict.

        Returns:
            Number of fingerprint entries removed.
        """
        # Collect keys under the top-level lock, then mutate per scope.
        by_scope: dict[str, list[int]] = defaultdict(list)
        with self._lock:
            for object_key in object_keys:
                for model_scope, poly in self._by_key.pop(object_key, []):
                    by_scope[model_scope].append(poly)

        removed = 0
        for model_scope, polys in by_scope.items():
            with self._lock:
                table = self._scopes.get(model_scope)
            if table is None:
                continue
            with table.lock:
                for poly in polys:
                    if table.evict(poly):
                        removed += 1
        return removed

    def match(
        self, model_scope: str, tokens: "list[int] | np.ndarray"
    ) -> list[GlobalMatch]:
        """Match request tokens against the directory.

        Rolls a chunk-window hash over the request, then probes the scope's
        direct-address table every ``probe_stride`` positions in one numpy
        gather; a full 64-bit re-check in the sparse hit loop rejects bucket
        collisions. De-duplicates by ``object_key``. Mirrors the local
        ``BlendTokenRangeMatcherV3.match_sub_sequence``.

        Args:
            model_scope: Scope to match within (the model name).
            tokens: The request tokens (a ``list[int]`` or a ``uint64`` array).

        Returns:
            Matches in ascending ``cur_st`` order; empty if nothing matched.
        """
        arr = np.asarray(tokens, dtype=np.uint64)
        if arr.shape[0] < self._chunk_size:
            return []

        with self._lock:
            table = self._scopes.get(model_scope)
        if table is None:
            return []

        rolling = rolling_hash_windows_numba(arr, self._chunk_size, POLY_BASE)
        probe = rolling[:: self._probe_stride]
        matches: list[GlobalMatch] = []
        seen: set[str] = set()
        stride = self._probe_stride
        with table.lock:
            # One gather: strided hash -> slot's entry index (-1 = empty slot).
            entry_ids = table.slots[probe & table.mask]
            hit_positions = np.nonzero(entry_ids >= 0)[0]
            for p in hit_positions.tolist():
                cid = int(entry_ids[p])
                if int(probe[p]) != table.hashes[cid]:
                    continue  # bucket collision: different content in this slot
                loc = table.locs[cid]
                if loc is None or loc.object_key in seen:
                    continue  # evicted, or already matched
                seen.add(loc.object_key)
                matches.append(
                    GlobalMatch(
                        object_key=loc.object_key,
                        old_st=loc.old_st,
                        cur_st=p * stride,
                    )
                )
        return matches
