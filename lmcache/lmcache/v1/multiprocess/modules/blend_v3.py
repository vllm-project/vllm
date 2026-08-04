# SPDX-License-Identifier: Apache-2.0
"""Blend V3: paged-aware CacheBlend as an EngineModule.

Plugs into the unified MPCacheServer; standard REGISTER_KV_CACHE +
CB_REGISTER_ROPE_V3 for setup; STORE wrapper registers fingerprints;
retrieve scatters into the request's paged blocks.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Empty as QueueEmpty
from queue import Queue
from typing import TYPE_CHECKING, Any
import threading
import time
import weakref

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import (
        BlendCoordinatorClient,
        RemoteMatch,
    )

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import check_interprocess_event_support
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    PrefetchRequestSpec,
    TrimPolicy,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.mp_coordinator.blend_client import PENDING
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import compute_extra_count
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import (
    TokenHasher,
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    update_table_id_numba,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

#: Distinct no-op-success reasons already reported (bounded: a fixed set of
#: call sites), so the log costs nothing after the first occurrence of each.
_NOOP_REASONS_SEEN: set[str] = set()

# Plan-then-execute retrieve: one native call enqueues all fill/rope/scatter
# in a single GIL release, with the plan encoded as numpy int64 tables (one
# pybind crossing). The Python wave loop stays as fallback for c_ops builds
# that predate the op (and for inputs the planner declines).
_HAS_NATIVE_RETRIEVE_PLAN = hasattr(lmc_ops, "execute_cb_retrieve_plan_flat")

# torch dtype -> at::ScalarType (rope dispatch); missing -> Python fallback.
_TORCH_TO_AT_SCALAR = {
    torch.float16: 5,  # at::ScalarType::Half
    torch.float32: 6,  # at::ScalarType::Float
    torch.bfloat16: 15,  # at::ScalarType::BFloat16
}


# Default for cb_register_rope's wire-typed group_rot parameter (legacy: no
# declared windows). Never mutated — the handler only iterates it.
_EMPTY_GROUP_ROT: list[list[int]] = []


@dataclass
class _CBRopeState:
    """Per-instance RoPE state IPC-shared from vLLM; dangles on reallocate.

    Models with per-layer-type RoPE (distinct local/global theta)
    register one cache per distinct rope and a per-layer index into
    ``cos_sin_caches``.
    """

    head_size: int
    is_neox_style: bool  # NeoX = contiguous halves; else GPT-J.
    cos_sin_caches: list[torch.Tensor]
    group_to_cache: list[int]  # engine group idx -> cache idx; empty = cache 0
    # Per-group rotation window ``(offset_elems, width_elems)``; ``None``
    # skips re-RoPE for the group, empty list = legacy inferred geometry.
    # Required for MLA: inference would rotate the latent's content dims.
    group_rot: "list[tuple[int, int] | None]" = field(default_factory=list)

    def rot_for_group(
        self, engine_group_idx: int, dtype: "torch.dtype | None" = None
    ) -> "tuple[int, int] | None":
        """The rotation window for one kernel group.

        Args:
            engine_group_idx: The kernel group's engine group index.
            dtype: The kernel group's buffer dtype, when known. One engine
                group can hold several *kernel* groups (GLM: the bf16 latent
                and the uint8 fp8 index cache both sit in engine group 0), so
                under a DECLARED map a non-float kernel group is skipped —
                the rope kernel cannot rotate quantized rows, and the
                declared window describes the family's float plane. Legacy
                registrations keep today's behavior (no dtype-based skip).

        Returns:
            ``(offset_elems, width_elems)``, or ``None`` when the group's
            re-RoPE is skipped (declared ``[]``, or non-float under a
            declared map). Legacy registrations (empty ``group_rot``) get
            ``(0, head_size)``.

        Raises:
            RuntimeError: If ``engine_group_idx`` is outside a non-empty map.
        """
        if not self.group_rot:
            return (0, self.head_size)
        if dtype is not None and not dtype.is_floating_point:
            return None
        if engine_group_idx >= len(self.group_rot):
            raise RuntimeError(
                f"CB re-RoPE: engine group {engine_group_idx} has no rope "
                f"geometry (map covers {len(self.group_rot)} groups)."
            )
        return self.group_rot[engine_group_idx]

    def cache_for_group(self, engine_group_idx: int) -> torch.Tensor:
        """The cos/sin cache for one engine group.

        Engine groups partition layers by attention type, and rope follows
        attention type (sliding=local theta, full=global theta),
        so each engine group has exactly one cache.

        Args:
            engine_group_idx: The kernel group's engine group index.

        Returns:
            The group's cos/sin cache tensor.

        Raises:
            RuntimeError: If ``engine_group_idx`` is outside the map.
        """
        if not self.group_to_cache:
            return self.cos_sin_caches[0]
        if engine_group_idx >= len(self.group_to_cache):
            raise RuntimeError(
                f"CB re-RoPE: engine group {engine_group_idx} has no rope "
                f"cache mapping (map covers {len(self.group_to_cache)} groups)."
            )
        return self.cos_sin_caches[self.group_to_cache[engine_group_idx]]


@dataclass
class _CBUnifiedJob:
    """Per-request poll state for non-blocking cb_unified_lookup.

    Stashed across polls because the underlying status/found polls are
    consume-once.
    """

    matches: list[CBMatchResult]
    num_tokens: int = 0
    # Prefix leg (blend_v3-owned submit/poll). ``prefix_handle`` is None when
    # there is no GPU context / no full chunk (poll reports 0 coverage).
    prefix_handle: PrefetchHandle | None = None
    prefix_world_size: int = 1
    prefix_chunks: int | None = None  # stashed when the prefix poll completes
    retained_chunks: list[int] | None = None  # SEGMENTED_PREFIX: full gapped set
    sparse_started: bool = False  # prefix done -> sparse leg submitted/skipped
    handle: PrefetchHandle | None = None  # sparse handle, None if no sparse leg
    non_prefix: list[CBMatchResult] | None = None
    per_hash_obj_keys: dict | None = None
    expanded_uidx: list[int] | None = None
    found_uidx: set[int] | None = None  # stashed when the sparse poll completes
    l2_keys: int = 0  # sparse keys needing an L2 load (0 => no L2 read, span skipped)
    coord_submitted: bool = False  # coordinator match query was issued
    coord_deadline: float = 0.0  # time.monotonic() wall-clock cutoff for the leg


class BlendTokenRangeMatcherV3:
    """V3 matcher: token-level probe (any offset) + full-hash collision
    rejection. Self-contained (does not inherit a base matcher)."""

    _TABLE_BITS: int = 20  # 2^20 ~ 1 M entries
    _TABLE_SIZE: int = 1 << _TABLE_BITS
    _BASE: np.uint64 = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci-hashing const

    def __init__(self, chunk_size: int = 256):
        """Initialize the V3 matcher.

        Args:
            chunk_size (int): Tokens per non-overlapping fingerprint chunk.
        """
        self.chunk_size = chunk_size
        # poly_chunk_hash -> compact_chunk_id; -1 = empty
        self._table_id = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        self._mask = np.uint64(self._TABLE_SIZE - 1)
        # compact_chunk_id -> caller token_hash (full bytes); None once evicted
        self._chunk_token_hash: list[bytes | None] = []
        # token_hash -> start position in its registered sequence
        self._token_hash_to_start: dict[bytes, int] = {}
        # compact_chunk_id -> table slot (reverse lookup for eviction)
        self._compact_id_to_slot = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        # token_hash -> compact_chunk_id (for eviction lookup)
        self._token_hash_to_compact_id: dict[bytes, int] = {}
        self._lock = threading.Lock()
        # V3 addition: compact_chunk_id -> full poly hash, for collision reject.
        self._chunk_poly_hash: list[int] = []

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
        start_chunk_idx: int = 0,
        position_offset: int = 0,
    ) -> None:
        """Index a stored sequence's non-overlapping chunks into the matcher.

        Records each new chunk's poly hash + start position so a later
        match_sub_sequence can find it. Thread-safe (holds the matcher lock).

        Args:
            token_ids (list[int]): The stored sequence's token IDs.
            token_hashes (list[bytes]): Per-chunk content hashes (one per
                chunk), used as the dedup/eviction key.
            start_chunk_idx (int): First chunk to index; 1 skips chunk 0 (the
                prefix lookup leg owns it).
            position_offset (int): Added to each recorded start position (for
                indexing a tail-slice of a larger sequence).

        Returns:
            None.
        """
        arr = np.array(token_ids, dtype=np.uint64)
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0 or start_chunk_idx >= n:
            return

        with self._lock:
            new_idxs = [
                i
                for i in range(start_chunk_idx, n)
                if token_hashes[i] not in self._token_hash_to_compact_id
            ]
            if not new_idxs:
                return
            n_new = len(new_idxs)
            new_chunk_hashes = chunk_hashes[new_idxs]

            base_id = len(self._chunk_token_hash)
            if base_id + n_new > self._TABLE_SIZE:
                logger.error(
                    "BlendTokenRangeMatcherV3 compact-ID overflow: %d chunks "
                    "registered, cannot add %d more (limit %d). Skipping.",
                    base_id,
                    n_new,
                    self._TABLE_SIZE,
                )
                return
            if base_id + n_new > int(self._TABLE_SIZE * 0.8):
                logger.warning(
                    "BlendTokenRangeMatcherV3 nearing capacity: %d/%d "
                    "compact IDs used. Hash collision rate is rising; "
                    "hit rate will degrade.",
                    base_id + n_new,
                    self._TABLE_SIZE,
                )
            compact_ids = np.arange(base_id, base_id + n_new, dtype=np.int64)

            update_table_id_numba(new_chunk_hashes, self._table_id, compact_ids)

            for k, orig_i in enumerate(new_idxs):
                th = token_hashes[orig_i]
                cid = int(compact_ids[k])
                poly_hash = int(new_chunk_hashes[k])
                slot = poly_hash & int(self._mask)
                self._chunk_token_hash.append(th)
                self._chunk_poly_hash.append(poly_hash)
                self._token_hash_to_start[th] = (
                    position_offset + orig_i * self.chunk_size
                )
                self._compact_id_to_slot[cid] = slot
                self._token_hash_to_compact_id[th] = cid

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Find every registered chunk reused anywhere in a query sequence.

        Vectorized direct-address probe over all token positions, then a small
        verify loop over the surviving hits (a full poly-hash check rejects
        bucket collisions; evicted/unknown chunks are skipped). Thread-safe.

        Args:
            token_ids (list[int]): The query sequence's token IDs.

        Returns:
            list[CBMatchResult]: One result per unique reused chunk (cur_st
            = its first query position, old_st = its stored position).
            Empty if the query is shorter than one chunk or nothing matched.
        """
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)

        with self._lock:
            if not self._chunk_token_hash:
                return []

            # Vectorized direct-address probe over all positions. The table is
            # sparse (TABLE_SIZE >> registered chunks), so only true matches and
            # a few bucket collisions reach the Python verify loop below.
            cids_at_pos = self._table_id[rolling & self._mask]
            hit_positions = np.nonzero(cids_at_pos >= 0)[0]

            seen_cids: set[int] = set()
            results: list[CBMatchResult] = []
            for pos in hit_positions:
                pos = int(pos)
                cid = int(cids_at_pos[pos])
                if cid in seen_cids:
                    continue
                if int(rolling[pos]) != self._chunk_poly_hash[cid]:
                    continue  # bucket-only collision
                th = self._chunk_token_hash[cid]
                if th is None:
                    continue  # evicted
                old_st = self._token_hash_to_start.get(th)
                if old_st is None:
                    continue
                seen_cids.add(cid)
                results.append(
                    CBMatchResult(
                        old_st=old_st,
                        old_ed=old_st + self.chunk_size,
                        cur_st=pos,
                        cur_ed=pos + self.chunk_size,
                        hash=th,
                    )
                )
            logger.info(
                "[match_probe] n_tok=%d table_hits=%d matches=%d",
                len(token_ids),
                len(hit_positions),
                len(results),
            )
            return results

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict the given chunks from the matcher.

        Clears each chunk's table slot + poly hash so later probes cannot match
        it. Thread-safe.

        Args:
            token_hashes (list[bytes]): Content hashes of the chunks to evict.
        """
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get(th)
                if cid is None:
                    continue
                slot = int(self._compact_id_to_slot[cid])
                if slot < 0:
                    logger.warning(
                        "compact_id %d has no valid table slot; "
                        "entry may have been evicted twice",
                        cid,
                    )
                    continue
                self._table_id[slot] = -1
                self._compact_id_to_slot[cid] = -1
                self._chunk_token_hash[cid] = None
                self._chunk_poly_hash[cid] = 0
                self._token_hash_to_start.pop(th, None)
                del self._token_hash_to_compact_id[th]


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Total token coverage, merging overlapping ranges (sliding-window probe
    can return overlaps; naive sum would double-count)."""
    if not results:
        return 0
    intervals = sorted((r.cur_st, r.cur_ed) for r in results)
    coverage = 0
    cur_end = -1
    for st, ed in intervals:
        if st >= cur_end:
            coverage += ed - st
        elif ed > cur_end:
            coverage += ed - cur_end
        cur_end = max(cur_end, ed)
    return coverage


def _group_slot_mappings(
    resolved_groups: "list[tuple[torch.Tensor, int]]", pos: torch.Tensor
) -> "list[torch.Tensor]":
    """Per-group paged slot ids for the logical positions ``pos``
    (``block_ids[pos // bs] * bs + pos % bs``). The div/mod pair is shared
    across groups with the same block size — dispatch count is the cost that
    matters under the shared GIL."""
    div_mod: "dict[int, tuple[torch.Tensor, torch.Tensor]]" = {}
    mappings: "list[torch.Tensor]" = []
    for group_block_ids, group_bs in resolved_groups:
        pair = div_mod.get(group_bs)
        if pair is None:
            pair = (pos // group_bs, pos % group_bs)
            div_mod[group_bs] = pair
        mappings.append(group_block_ids[pair[0]] * group_bs + pair[1])
    return mappings


def _cb_group_rope_geometry(
    group: Any,
    kv_size: int,
    hidden_dim: int,
    head_size: int,
    group_idx: int,
    rot: "tuple[int, int] | None" = None,
) -> "tuple[bool, int, int, int]":
    """Per-group re-RoPE geometry rules, shared by the batched rope path and
    the retrieve-plan builder so they cannot drift.

    Fused blocks-first K/V packs K+V into a doubled head dim (kv_size==1);
    detect it so only the K half is re-RoPE'd in place. kv_size==1 without
    fused packing is the M3 key-only index side cache; kv_size==2 is main
    K/V. In every case only the K plane is rotated.

    ``rot`` is the declared ``(offset_elems, width_elems)`` rotation window;
    offset > 0 means MLA (rope dims trail the row) — the row is one "head"
    and only ``[offset, offset + width)`` rotates. MLA groups must arrive
    with ``rot`` set: undeclared, a 576-wide latent passes the inference
    checks below as 9 x 64 heads and its content dims get rotated.

    Returns:
        ``(fused_packed, per_head, n_heads, rot_offset)`` — per-head width is
        ``2 * head_size`` for fused-packed layouts; ``rot_offset`` is the
        element offset of the rotation window within each per-head row (0
        for every non-MLA layout).

    Raises:
        RuntimeError: On a compressed (compress_ratio != 1) layout, a
            kv_size other than 2 (K/V) or 1 (key-only index), a
            head_size/hidden_dim mismatch, or a declared window that does
            not fit the row.
    """
    if group.tokens_per_block != group.slots_per_block:
        raise RuntimeError(
            f"CB v3: group {group_idx} is compressed "
            f"(tokens_per_block={group.tokens_per_block}, "
            f"slots_per_block={group.slots_per_block}); "
            f"compressed layouts unsupported."
        )
    if rot is not None and rot[0] > 0:
        rot_offset, rot_width = rot
        if kv_size != 1:
            raise RuntimeError(
                f"CB v3: group {group_idx} declares an MLA rope window "
                f"{rot} but has kv_size={kv_size}; MLA latents are a "
                "single plane (kv_size 1)."
            )
        if rot_offset + rot_width != hidden_dim:
            raise RuntimeError(
                f"CB v3: group {group_idx} rope window {rot} does not end "
                f"the row (hidden_dim={hidden_dim}); MLA latents are "
                "[content | rope]."
            )
        return False, hidden_dim, 1, rot_offset
    if kv_size not in (1, 2):
        raise RuntimeError(
            f"CB v3: group {group_idx} has kv_size={kv_size}; only K/V "
            "(2), fused-packed K/V, key-only (1), and declared-MLA "
            "layouts are supported."
        )
    _ekf = getattr(group, "engine_kv_format", None)
    # Both spellings of blocks-first fused K/V must be recognised. The vLLM
    # detector used to split the trailing 2*head_size axis and report
    # *_TWO_HS; it now keeps the tensor raw and reports *_CS. Matching only
    # *_TWO_HS leaves fused_packed False on current vLLM, which halves
    # per_head, doubles n_heads, and re-RoPEs across the V plane.
    fused_packed = _ekf is not None and int(_ekf) in (
        int(lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS),
        int(lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS),
        int(lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_CS),
        int(lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS),
    )
    per_head = head_size * (2 if fused_packed else 1)
    n_heads = hidden_dim // per_head
    if n_heads * per_head != hidden_dim:
        raise RuntimeError(
            f"CB rope: group {group_idx} hidden_dim ({hidden_dim}) not a "
            f"multiple of per-head width ({per_head}; fused={fused_packed})."
        )
    return fused_packed, per_head, n_heads, 0


class BlendV3Module(InstanceLivenessTarget):
    """Paged-aware V3 CacheBlend. Wraps LMCacheDrivenTransfer STORE to register
    fingerprints; serves CB rope/lookup/retrieve RPCs; reads cross-module
    GPU state via :class:`LMCacheDrivenTransferModule.cache_contexts`."""

    def __init__(
        self,
        ctx: MPCacheServerContext,
        lmcache_driven_transfer: LMCacheDrivenTransferModule,
        coordinator: "BlendCoordinatorClient | None" = None,
        enable_segmented_prefix: bool = False,
    ):
        self._ctx = ctx
        self._transfer_module = lmcache_driven_transfer
        # Server config (--enable-segmented-prefix): retain the gapped prefix on
        # a mid-prefix L2 retrieve failure instead of truncating at the gap.
        self._segmented_prefix = enable_segmented_prefix
        # Optional bridge to the fleet-wide fingerprint directory. ``None`` =>
        # purely local matching (publish/query paths skipped).
        self._coordinator = coordinator

        self._token_range_matcher = BlendTokenRangeMatcherV3(ctx.chunk_size)
        self._event_bus = ctx.event_bus
        self._cb_rope_state: dict[int, _CBRopeState] = {}

        # L2 opt: cache TP-expanded obj_keys at lookup, pop at retrieve.
        self._lookup_obj_keys_cache: dict[str, dict[bytes, list]] = {}
        self._lookup_obj_keys_lock = threading.Lock()

        # vLLM may call retrieve twice per request (partial- then full-block
        # alloc): ranges already scattered, so the repeat call skips them.
        # Bounded LRU keyed by (request id, WORKER id) -- at TP>1 each worker
        # issues its own retrieve and scatters into its own KV buffers, so the
        # key must include the worker or later ranks skip work they never did.
        self._cb_applied_match_ranges: "OrderedDict[tuple[str, int | None], set[tuple[bytes, int, int]]]" = OrderedDict()  # noqa: E501

        # Request-invariant retrieve-plan specs per GPU context (entries die
        # with the context). The cached tuple holds the rope_state it was
        # resolved against so a re-registration invalidates by identity.
        self._cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )
        # Persistent pinned + device slot-mapping staging per GPU context,
        # grown on demand (see _cb_slot_buffers).
        self._cb_slot_staging: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )

        # Non-blocking cb_unified_lookup poll state (submit-once, poll-on-recall)
        # so the handler never holds a worker thread across the L2->L1 loads.
        self._cb_jobs: dict[str, _CBUnifiedJob] = {}
        self._cb_jobs_lock = threading.Lock()

        # Async fingerprint registration: store enqueues, worker drains.
        _FpJob = tuple[list[int], list[bytes], int, int]
        self._fingerprint_queue: "Queue[_FpJob]" = Queue()
        self._fingerprint_stop = threading.Event()
        self._fingerprint_worker = threading.Thread(
            target=self._drain_fingerprint_queue,
            name="cb-fingerprint-worker",
            daemon=True,
        )
        self._fingerprint_worker.start()

        # In-flight fingerprint hashes; storage_gate keeps these from eviction.
        self._pending_fp_hashes: set[bytes] = set()
        self._pending_fp_lock = threading.Lock()

        # Lazy eviction strikes; evict only at threshold so async re-store
        # can refresh the bucket first.
        self._stale_strike: dict[bytes, int] = {}
        self._STALE_STRIKE_THRESHOLD = 2

    # ------------------------------------------------------------------
    # EngineModule protocol
    # ------------------------------------------------------------------

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        # STORE shadows LMCacheDrivenTransfer's; compositor registers V3 last.
        return [
            HandlerSpec(RequestType.STORE, self.store, ThreadPoolType.AFFINITY),
            HandlerSpec(
                RequestType.CB_REGISTER_ROPE_V3,
                self.cb_register_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNREGISTER_ROPE_V3,
                self.cb_unregister_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNIFIED_LOOKUP,
                self.cb_unified_lookup,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.CB_RETRIEVE_PRE_COMPUTED_V3,
                self.cb_retrieve_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        # Meta is derived live from MP server gpu_transfe

        cache_contexts = self._transfer_module.context_entries_snapshot()

        def _meta(iid: int) -> "tuple[str, int] | None":
            entry = cache_contexts.get(iid)
            return (entry.model_name, entry.world_size) if entry is not None else None

        return {
            "registered_cb_rope_instances": list(self._cb_rope_state.keys()),
            "cb_rope_meta": {str(iid): _meta(iid) for iid in self._cb_rope_state},
            "active_cb_lookups": len(self._cb_jobs),
        }

    def close(self) -> None:
        self._fingerprint_stop.set()
        if self._coordinator is not None:
            # Joins the client's daemon thread and closes its httpx.Client;
            # otherwise the coordinator leg leaks both on server shutdown.
            self._coordinator.close()
        self._cb_rope_state.clear()

    # ------------------------------------------------------------------
    # V3 RPCs
    # ------------------------------------------------------------------

    def cb_register_rope(
        self,
        instance_id: int,
        cos_sin_caches_ipc: list[DeviceIPCWrapper],
        head_size: int,
        is_neox_style: bool,
        group_to_cache: list[int],
        # Annotation must equal the protocol payload class exactly — the MQ
        # server's add_handler signature check (mq.py same_type) is strict.
        # Direct (non-wire) callers may still pass tuples/None entries; the
        # normalization below accepts them.
        group_rot: list[list[int]] = _EMPTY_GROUP_ROT,
    ) -> None:
        """Bolt CB re-RoPE state onto an already-registered KV-cache instance.

        Idempotent; ``REGISTER_KV_CACHE`` must precede this. Strips any
        YaRN/longrope mscale baked into each rope cache so re-RoPE stays a
        pure rotation.

        Args:
            instance_id (int): KV-cache instance to attach rope state to.
            cos_sin_caches_ipc (list[DeviceIPCWrapper]): IPC handles to vLLM's
                cos/sin rope cache(s) — one per distinct rope (dual-RoPE
                models send local/global); single-rope models send one.
            head_size (int): Rotary head dimension.
            is_neox_style (bool): True for NeoX (contiguous halves), else GPT-J.
            group_to_cache (list[int]): Per-engine-group index into the
                caches list; empty means every group uses cache 0.
            group_rot: Per-engine-group rotation window ``(offset_elems,
                width_elems)`` into each token row, or ``None`` per entry to
                skip that group's re-RoPE. Empty/omitted = legacy inference
                (rotate ``head_size`` dims at offset 0). MLA models must
                declare this — e.g. GLM/DeepSeek latents are
                ``(kv_lora_rank, qk_rope_head_dim)`` — because a single-plane
                MLA row is indistinguishable from a key-only cache to the
                legacy inference and would get its content dims rotated.

        Raises:
            ValueError: If ``instance_id`` has no registered KV cache, the
                cache list is empty, ``group_to_cache`` references a missing
                cache or does not cover every engine group of the registered
                model, or a ``group_rot`` entry is malformed.
        """
        entry = self._transfer_module.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(
                f"Instance {instance_id} has no paged KV cache registered; "
                "send REGISTER_KV_CACHE before CB_REGISTER_ROPE_V3."
            )
        if not cos_sin_caches_ipc:
            raise ValueError("CB_REGISTER_ROPE_V3 requires >=1 cos/sin cache.")
        if group_to_cache:
            if min(group_to_cache) < 0 or max(group_to_cache) >= len(
                cos_sin_caches_ipc
            ):
                raise ValueError(
                    f"group_to_cache {group_to_cache} contains indices outside "
                    f"[0, {len(cos_sin_caches_ipc)}) for the sent cache(s)."
                )
            # Fail at registration, not mid-retrieve: every engine group of
            # the registered model must have a cache mapping.
            max_eg_idx = max(
                (
                    g.engine_group_idx
                    for g in entry.cache_context.kv_layer_groups_manager.kernel_groups
                ),
                default=-1,
            )
            if len(group_to_cache) <= max_eg_idx:
                raise ValueError(
                    f"group_to_cache covers {len(group_to_cache)} engine "
                    f"group(s) but the registered model has engine groups up "
                    f"to index {max_eg_idx}."
                )

        # Normalize declared rope windows (serialization turns tuples into
        # lists); validate here so a bad registration fails loudly instead of
        # mid-retrieve.
        norm_rot: "list[tuple[int, int] | None]" = []
        for eg_idx, rot_entry in enumerate(group_rot or []):
            if rot_entry is None or len(rot_entry) == 0:
                # None (direct call) / [] (wire encoding): skip this group.
                norm_rot.append(None)
                continue
            if len(rot_entry) != 2 or int(rot_entry[0]) < 0 or int(rot_entry[1]) <= 0:
                raise ValueError(
                    f"group_rot[{eg_idx}] = {rot_entry!r}: expected "
                    "(offset >= 0, width > 0) or None."
                )
            norm_rot.append((int(rot_entry[0]), int(rot_entry[1])))

        cos_sin_caches: list[torch.Tensor] = []
        for cache_idx, cache_ipc in enumerate(cos_sin_caches_ipc):
            cos_sin_cache = cache_ipc.to_tensor()
            # YaRN/longrope bake an mscale m into the rope cache
            # (cos²+sin²=m²≠1). vLLM already folds m into stored K, but CB
            # re-RoPE assumes a pure rotation, so an un-normalized m injects
            # an m² error per K element.
            _c32 = cos_sin_cache.to(torch.float32)
            _half = _c32.shape[1] // 2
            _m = float((_c32[:, :_half] ** 2 + _c32[:, _half:] ** 2).mean().sqrt())
            if abs(_m - 1.0) >= 1e-3:
                logger.info(
                    "CB re-RoPE: cache %d: stripping rope-cache mscale=%.4f "
                    "(m²=%.4f → K inflation if uncorrected) → unit magnitude",
                    cache_idx,
                    _m,
                    _m * _m,
                )
                cos_sin_cache = (_c32 / _m).to(cos_sin_cache.dtype)
            cos_sin_caches.append(cos_sin_cache)

        self._cb_rope_state[instance_id] = _CBRopeState(
            head_size=head_size,
            is_neox_style=is_neox_style,
            cos_sin_caches=cos_sin_caches,
            group_to_cache=list(group_to_cache),
            group_rot=norm_rot,
        )

        logger.info(
            "Registered CB rope state for instance %d "
            "(%d cache(s), shapes=%s dtype=%s, head_size=%d, is_neox=%s, "
            "group_map=%s, group_rot=%s)",
            instance_id,
            len(cos_sin_caches),
            [tuple(c.shape) for c in cos_sin_caches],
            cos_sin_caches[0].dtype,
            head_size,
            is_neox_style,
            "uniform" if not group_to_cache else str(group_to_cache),
            "legacy" if not norm_rot else str(norm_rot),
        )

        # Pre-warm the retrieve-plan invariants + slot-mapping staging for
        # this instance, off the retrieve critical path.
        try:
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            ctx = entry.cache_context if entry is not None else None
            if ctx is not None:
                self._cb_slot_buffers(
                    ctx, ctx.kv_layer_groups_manager.num_kernel_groups, 1 << 16
                )
                rope_state = self._cb_rope_state[instance_id]
                max_batch = ctx.max_batch_size
                if self._cb_plan_invariants.get(ctx) is None:
                    resolved = self._resolve_cb_plan_invariants(
                        ctx, rope_state, max_batch
                    )
                    if resolved is not None:
                        self._cb_plan_invariants[ctx] = (
                            rope_state,
                            max_batch,
                            resolved,
                        )
        except Exception:
            logger.debug("CB plan pre-warm skipped", exc_info=True)

    def cb_unregister_rope(self, instance_id: int) -> None:
        """Drop the instance's CB rope state; the paged KV cache is left intact.

        Args:
            instance_id (int): Instance whose rope state to remove (use
                ``UNREGISTER_KV_CACHE`` to free the KV cache itself).
        """
        self._cb_rope_state.pop(instance_id, None)
        if self._transfer_module.get_and_touch_context_entry(instance_id) is None:
            logger.warning(
                "cb_unregister_rope: instance %d not registered", instance_id
            )
            return
        logger.info("Unregistered CB rope state for instance %d", instance_id)

    def drop_instance_state(self, instance_id: int) -> None:
        """Drop blend state for a reaped instance (InstanceLivenessTarget hook).

        Only the CB rope state is held per instance; the GPU cache context is
        owned by ``LMCacheDrivenTransferModule`` (no mirror here), so reaping
        the GPU entry frees it directly. A no-op if no rope state is held.

        Args:
            instance_id: The reaped worker's instance ID.
        """
        if self._cb_rope_state.pop(instance_id, None) is not None:
            logger.info("Dropped CB rope state for reaped instance %d", instance_id)

    # ------------------------------------------------------------------
    # Unified lookup (CB_UNIFIED_LOOKUP) + shared helpers
    # ------------------------------------------------------------------

    def _drain_fingerprints_sync(self) -> None:
        """Sync-drain pending fingerprint registrations (the async drainer
        races at low max_tokens)."""
        while True:
            try:
                job = self._fingerprint_queue.get_nowait()
            except QueueEmpty:
                break
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset = job
            try:
                self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
            except Exception:
                logger.exception("CB fingerprint registration failed (sync drain)")

    def _match_fingerprints(self, key: IPCCacheServerKey) -> list[CBMatchResult]:
        """Drain pending registrations and fingerprint-match sub-sequences.

        Returns the raw matches (any order, possibly overlapping); the caller
        applies the prefix filter + overlap dedup once via
        :meth:`_non_overlapping_after_prefix`.
        """
        self._drain_fingerprints_sync()
        return self._token_range_matcher.match_sub_sequence(list(key.token_ids))

    @staticmethod
    def _non_overlapping_after_prefix(
        matches: list[CBMatchResult], prefix_tokens: int
    ) -> list[CBMatchResult]:
        """Matches outside the prefix coverage, leftmost-greedy overlap-deduped.

        Drops matches the prefix leg already covers (``cur_st < prefix_tokens``),
        then keeps a left-to-right non-overlapping subset -- two matches over the
        same request range can't both scatter. Filtering precedes the dedup so a
        prefix-covered match cannot suppress a usable one in the greedy pass.

        Args:
            matches: Candidate matches in any order; ``cur_st``/``cur_ed`` are
                request token positions.
            prefix_tokens: Contiguous prefix coverage in tokens; matches starting
                before it are dropped. Pass ``0`` to keep all (dedup only).

        Returns:
            Non-overlapping matches in ascending ``cur_st`` order.
        """
        kept: list[CBMatchResult] = []
        covered_end = -1
        for r in sorted(
            (r for r in matches if r.cur_st >= prefix_tokens),
            key=lambda r: r.cur_st,
        ):
            if r.cur_st >= covered_end:
                kept.append(r)
                covered_end = r.cur_ed
        return kept

    def _resolve_cb_layout_desc(
        self, model_name: str, world_size: int
    ) -> "MemoryLayoutDesc | None":
        """Find the CB KV buffer layout for ``(model_name, world_size)``.

        Reads the thread-safe ``layout_desc_registry`` (populated by
        ``lmcache_driven_transfer`` on KV-cache registration) rather than
        iterating ``cache_contexts``: iteration races concurrent
        register/unregister, and the registry holds the complete multi-group
        descriptor instead of a single-group manual reconstruction.

        Args:
            model_name (str): Model name to match.
            world_size (int): Tensor-parallel world size to match.

        Returns:
            MemoryLayoutDesc | None: The matching layout, or None if no
            registered CB context matches.
        """
        return self._ctx.layout_desc_registry.find(model_name, world_size)

    def _sparse_prefetch_submit(
        self,
        key: IPCCacheServerKey,
        layout_desc: "MemoryLayoutDesc",
        matches: list[CBMatchResult],
    ) -> "tuple[PrefetchHandle, dict[bytes, list], list[int]]":
        """Coalesce all matches into one sparse L2->L1 prefetch and submit it.

        Non-blocking. Dedups object keys before submit (sparse keeps one read
        lock per loaded key, so a duplicate would leak). The caller polls
        ``query_prefetch_status(handle)`` then calls :meth:`_sparse_classify`
        with the found set.

        Args:
            key (IPCCacheServerKey): The request key.
            layout_desc (MemoryLayoutDesc): CB KV buffer layout for L1 alloc.
            matches (list[CBMatchResult]): Non-prefix matches to prefetch.

        Returns:
            tuple[PrefetchHandle, dict[bytes, list], list[int]]: the prefetch
            handle, per-hash TP-expanded object keys, and each expanded
            position's deduped-key index (maps the per-key found set back to
            every chunk).
        """
        world_size = key.world_size
        per_hash_obj_keys: dict[bytes, list] = {}
        all_hashes = [r.hash for r in matches]
        all_obj_keys = ipc_key_to_object_keys(key, all_hashes, [0])[0]
        for i, h in enumerate(all_hashes):
            per_hash_obj_keys[h] = all_obj_keys[i * world_size : (i + 1) * world_size]

        # Dedup keys before submit (sparse keeps one read lock per loaded key;
        # a duplicate would leak). Map each expanded position to its deduped
        # index so the per-key found set resolves back to every chunk.
        uniq_keys: list = []
        key_to_uidx: dict = {}
        expanded_uidx: list[int] = []
        for k in all_obj_keys:
            uidx = key_to_uidx.get(k)
            if uidx is None:
                uidx = len(uniq_keys)
                key_to_uidx[k] = uidx
                uniq_keys.append(k)
            expanded_uidx.append(uidx)

        handle: PrefetchHandle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=uniq_keys,
                layout_desc=layout_desc,
                policy=TrimPolicy.SPARSE,
            ),
            external_request_id=key.request_id,
        )
        return handle, per_hash_obj_keys, expanded_uidx

    def _sparse_classify(
        self,
        key: IPCCacheServerKey,
        matches: list[CBMatchResult],
        found_uidx: set[int],
        per_hash_obj_keys: dict[bytes, list],
        expanded_uidx: list[int],
    ) -> list[CBMatchResult]:
        """Classify each prefetched chunk as found or stale, and finalize state.

        A chunk is found only if every TP rank's key loaded; stale chunks take
        an eviction strike (evicted at threshold, kept while still in-flight).
        Stashes the found chunks' obj_keys for the retrieve path.

        Args:
            key (IPCCacheServerKey): The request key.
            matches (list[CBMatchResult]): The submitted non-prefix matches.
            found_uidx (set[int]): Deduped-key indices that loaded.
            per_hash_obj_keys (dict[bytes, list]): Per-hash TP-expanded keys.
            expanded_uidx (list[int]): Each expanded position's deduped index.

        Returns:
            list[CBMatchResult]: The found subset, in cur_st order.
        """
        world_size = key.world_size
        found_cb_match_result: list[CBMatchResult] = []
        stale_hashes: list[bytes] = []
        for j, r in enumerate(matches):
            base = j * world_size
            if all(expanded_uidx[base + t] in found_uidx for t in range(world_size)):
                found_cb_match_result.append(r)
            else:
                stale_hashes.append(r.hash)
        # Stale drops silently shrink coverage (a fully-stale classify turns a
        # matched request into a full recompute) — log so it is diagnosable.
        if stale_hashes:
            logger.warning(
                "CB sparse classify for %s: %d found, %d stale of %d submitted",
                key.request_id,
                len(found_cb_match_result),
                len(stale_hashes),
                len(matches),
            )

        # Reset strikes for confirmed hashes.
        if found_cb_match_result:
            with self._pending_fp_lock:
                for r in found_cb_match_result:
                    self._stale_strike.pop(r.hash, None)
        # Stale: in-flight keep; >= threshold strikes -> evict.
        if stale_hashes:
            with self._pending_fp_lock:
                truly_evict: list[bytes] = []
                for h in stale_hashes:
                    if h in self._pending_fp_hashes:
                        continue
                    n = self._stale_strike.get(h, 0) + 1
                    if n >= self._STALE_STRIKE_THRESHOLD:
                        truly_evict.append(h)
                        self._stale_strike.pop(h, None)
                    else:
                        self._stale_strike[h] = n
            if truly_evict:
                self._token_range_matcher.remove_chunks(truly_evict)
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_CHUNKS_EVICTED,
                    metadata={"num_chunks": len(stale_hashes)},
                )
            )

        # Stash per-hash obj_keys for retrieve (L2 opt).
        if found_cb_match_result:
            cache_entry = {
                r.hash: per_hash_obj_keys[r.hash]
                for r in found_cb_match_result
                if r.hash in per_hash_obj_keys
            }
            with self._lookup_obj_keys_lock:
                self._lookup_obj_keys_cache[key.request_id] = cache_entry

        return found_cb_match_result

    def _submit_prefix_leg(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
        policy: TrimPolicy,
    ) -> "tuple[PrefetchHandle | None, int]":
        """Submit the CB prefix prefetch (non-blocking).

        Opens the ``cb.prefix_lookup`` span (CB namespace — CB requests no longer
        feed the MP request / mp.lookup_prefetch spans or the MP hit-rate
        aggregate; the CB hit-rate metric carries prefix tokens via
        CB_LOOKUP_END) and writes the shared session (``set_tokens`` +
        ``lookup_ipc_key``) so ``end_session``'s L1 keep-alive touch still
        resolves the request's keys.

        Args:
            key (IPCCacheServerKey): Request key (token IDs, request_id, model,
                world_size).
            tp_size (int): Tensor-parallel size for MLA multi-reader locking.
            policy (TrimPolicy): ``PREFIX`` or ``SEGMENTED_PREFIX``.

        Returns:
            tuple: ``(handle, world_size)``. ``handle`` is None when there is no
            GPU context or no full chunk (the poll then reports 0 coverage).
        """
        rid = key.request_id
        model_name, world_size = key.model_name, key.world_size
        self._event_bus.publish(
            Event(event_type=EventType.CB_PREFIX_LOOKUP_START, session_id=rid)
        )

        layout_desc = self._resolve_cb_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No CB GPU context for model %s ws %d during prefix lookup!",
                model_name,
                world_size,
            )
            return None, world_size

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            return None, world_size

        # Lookup-hash logger (chunk hashes, for debug); guarded so the metadata
        # dict is built only when a subscriber is listening.
        if self._event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=rid,
                    metadata={
                        "request_id": rid,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # Shared session: end_session reads lookup_ipc_key + the rolling hashes
        # to keep the request's KV alive in L1.
        session = self._ctx.session_manager.get_or_create(rid)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        extra_count = compute_extra_count(tp_size, world_size)
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes, [0])[0]
        handle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=obj_keys,
                layout_desc=layout_desc,
                extra_count=extra_count,
                policy=policy,
            ),
            external_request_id=rid,
        )
        return handle, world_size

    def _poll_prefix_leg(
        self, job: "_CBUnifiedJob", rid: str, segmented: bool
    ) -> "tuple[int, list[int] | None] | None":
        """Poll the CB prefix handle; on completion close the cb.prefix_lookup span.

        Consume-once: publishes CB_PREFIX_LOOKUP_END exactly once when the
        prefetch lands. The prefix hit tokens are accounted on the CB hit-rate
        metric at CB_LOOKUP_END, not here. For SEGMENTED_PREFIX also surfaces the
        gapped retained chunk set.

        Args:
            job (_CBUnifiedJob): Poll state holding the prefix handle + world size.
            rid (str): Request ID (event session_id).
            segmented (bool): SEGMENTED_PREFIX active -> also surface the gapped
                retained chunk set.

        Returns:
            tuple | None: ``(leading_chunks, retained_or_None)`` when resident;
            ``None`` while still loading. ``retained`` is the full gapped chunk
            set for SEGMENTED_PREFIX, else None.
        """
        if job.prefix_handle is not None:
            bm = self._ctx.storage_manager.query_prefetch_status(job.prefix_handle)
            if bm is None:
                return None  # still loading
            ws = job.prefix_world_size
            # NOTE(Kuntai): assumes uniform world size and prefix-ordered keys
            # that break at the first miss.
            leading = bm.count_leading_ones() // ws
            # Retain a chunk only if EVERY rank shard loaded (AND across the ws
            # shards); a chunk missing any rank's shard is demoted to a gap.
            if segmented:
                shard_counts: dict[int, int] = {}
                for ki in bm.get_indices_list():
                    c = ki // ws
                    shard_counts[c] = shard_counts.get(c, 0) + 1
                retained = sorted(c for c, n in shard_counts.items() if n == ws)
            else:
                retained = None
        else:
            # No GPU context / no full chunk: nothing loaded.
            leading, retained = 0, ([] if segmented else None)
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_PREFIX_LOOKUP_END,
                session_id=rid,
                metadata={"prefix_chunks": leading},
            )
        )
        return leading, retained

    def cb_unified_lookup(
        self, key: IPCCacheServerKey, tp_size: int
    ) -> CBUnifiedLookupResult | None:
        """Non-blocking single-RPC CB lookup (submit-once, poll-on-recall).

        First call submits the prefix lookup + fingerprint match; later calls
        poll both legs, returning ``None`` until the prefix and the sparse
        non-prefix complement are both resident in L1 (so a worker thread never
        blocks on the L2->L1 loads). The prefix job's L1 read locks persist for
        the retrieve.

        Args:
            key (IPCCacheServerKey): Request key (token IDs, request_id, model,
                world_size).
            tp_size (int): Tensor-parallel size for the prefix lookup.

        Returns:
            CBUnifiedLookupResult | None: ``None`` while either leg is still
            loading (the caller re-issues to poll); on completion, the prefix
            coverage in tokens plus the found non-prefix segments.
        """
        rid = key.request_id
        chunk_size = self._ctx.chunk_size

        with self._cb_jobs_lock:
            job = self._cb_jobs.get(rid)
        if job is None:
            # First invocation: start events + submit prefix + fingerprint match.
            self._event_bus.publish(
                Event(event_type=EventType.CB_REQUEST_START, session_id=rid)
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_START,
                    session_id=rid,
                    metadata={"num_tokens": len(key.token_ids)},
                )
            )
            # SEGMENTED_PREFIX: request the contiguous prefix with gap-tolerant
            # retention so a mid-prefix L2 retrieve failure leaves the post-gap
            # chunks L1-resident (picked up by the sparse leg as L1 hits, hole
            # recomputed) instead of truncating the prefix at the gap.
            prefix_policy = (
                TrimPolicy.SEGMENTED_PREFIX
                if self._segmented_prefix
                else TrimPolicy.PREFIX
            )
            # Prefix leg: blend_v3 owns the submit + the cb.prefix_lookup span
            # (under cb.lookup); prefix hit tokens land on the CB hit-rate
            # metric via CB_LOOKUP_END below.
            prefix_handle, prefix_ws = self._submit_prefix_leg(
                key, tp_size, prefix_policy
            )
            # Local and coordinator matching are mutually exclusive: with a
            # coordinator the fleet directory is the only source, so skip the
            # local matcher (and its span). The coordinator leg is async
            # (submitted below, resolved at poll) and is timed by cb.lookup.
            matches: list[CBMatchResult]
            if self._coordinator is not None:
                matches = []
            else:
                # Local fingerprint match: CPU-bound, tight span.
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINT_MATCH_START,
                        session_id=rid,
                    )
                )
                matches = self._match_fingerprints(key)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINT_MATCH_END,
                        session_id=rid,
                        metadata={"matches": len(matches)},
                    )
                )
            job = _CBUnifiedJob(
                matches=matches,
                num_tokens=len(key.token_ids),
                prefix_handle=prefix_handle,
                prefix_world_size=prefix_ws,
            )
            job.coord_submitted = self._submit_coordinator_match(key)
            if job.coord_submitted and self._coordinator is not None:
                job.coord_deadline = time.monotonic() + self._coordinator.match_budget_s
                # Coordinator match leg: async span, ended on the resolving poll
                # (or deadline) in _poll_coordinator_match.
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_COORDINATOR_MATCH_START,
                        session_id=rid,
                    )
                )
            with self._cb_jobs_lock:
                self._cb_jobs[rid] = job

        segmented = self._segmented_prefix

        # --- Prefix leg: poll (consume-once) until the L1+L2 prefix lands. ---
        if job.prefix_chunks is None:
            res = self._poll_prefix_leg(job, rid, segmented)
            if res is None:
                return None  # prefix still loading -> defer
            job.prefix_chunks, prefix_retained = res
            if segmented:
                job.retained_chunks = prefix_retained
        # Poll above set it (or we returned); narrow for the arithmetic below.
        assert job.prefix_chunks is not None
        prefix_chunks: int = job.prefix_chunks

        # Prefix done: reconcile the complement outside the prefix coverage and
        # submit one sparse prefetch for it (once). Prefix-covered chunks never
        # enter the sparse prefetch, so they cannot leak a read lock.
        if not job.sparse_started:
            prefix_tokens = prefix_chunks * chunk_size
            if self._coordinator is not None:
                candidates = self._poll_coordinator_match(job, rid)
                if candidates is None:
                    return None  # coordinator still in flight (bounded by deadline)
            else:
                candidates = job.matches
            # Under SEGMENTED_PREFIX, a same-position match the prefix leg already
            # retained rides the segmented tail (prefix-class: pure load, no CHECK)
            # -- drop it here so it is not scattered twice. A same-position match
            # the tail does NOT cover is a genuine cross-context hit: keep it as
            # non-prefix (re-RoPE no-ops at delta 0, but it still needs CHECK).
            # Shifted (cur != old) matches are always kept. Then the single
            # prefix-filter + overlap dedup over the rest.
            if segmented:
                retained = set(job.retained_chunks or [])
                candidates = [
                    c
                    for c in candidates
                    if c.old_st != c.cur_st or (c.cur_st // chunk_size) not in retained
                ]
            job.non_prefix = self._non_overlapping_after_prefix(
                candidates, prefix_tokens
            )
            if job.non_prefix:
                layout_desc = self._resolve_cb_layout_desc(
                    key.model_name, key.world_size
                )
                if layout_desc is not None:
                    (
                        job.handle,
                        job.per_hash_obj_keys,
                        job.expanded_uidx,
                    ) = self._sparse_prefetch_submit(key, layout_desc, job.non_prefix)
                    # Only trace the span when the prefetch actually reads L2;
                    # all-L1-resident matches do no L2 work worth a span.
                    job.l2_keys = len(job.handle.l2_orig_indices)
                    if job.l2_keys > 0:
                        self._event_bus.publish(
                            Event(
                                event_type=EventType.CB_SPARSE_PREFETCH_START,
                                session_id=rid,
                                metadata={
                                    "n_chunks": len(job.non_prefix),
                                    "world_size": key.world_size,
                                    "n_keys": len(job.non_prefix) * key.world_size,
                                    "l2_keys": job.l2_keys,
                                },
                            )
                        )
                else:
                    logger.error(
                        "No CB GPU context for model %s ws %d during cb_unified_lookup",
                        key.model_name,
                        key.world_size,
                    )
                    job.non_prefix = []
            job.sparse_started = True

        # --- Sparse leg: poll (consume-once) until the scattered chunks land. ---
        if job.handle is not None and job.found_uidx is None:
            bm = self._ctx.storage_manager.query_prefetch_status(job.handle)
            if bm is None:
                return None  # sparse still loading -> defer
            job.found_uidx = set(bm.get_indices_list())
            if job.l2_keys > 0:
                self._event_bus.publish(
                    Event(
                        event_type=EventType.CB_SPARSE_PREFETCH_END,
                        session_id=rid,
                        metadata={
                            "found_keys": len(job.found_uidx),
                            "l2_keys": job.l2_keys,
                        },
                    )
                )

        # --- BOTH legs ready: classify the complement + finalize. ---
        if job.handle is not None:
            found = self._sparse_classify(
                key,
                job.non_prefix or [],
                job.found_uidx or set(),
                job.per_hash_obj_keys or {},
                job.expanded_uidx or [],
            )
        else:
            found = []

        prefix_tokens = prefix_chunks * chunk_size
        num_tokens = job.num_tokens

        # Segmented tail: post-gap chunks the SEGMENTED_PREFIX prefix leg kept
        # resident (retained index > the leading run). Delivered at their
        # original positions (old_st == cur_st) so the connector tags them
        # ``prefix`` (pure load, no recompute); only the gap is recomputed. The
        # storage key is the same prefix-chained chunk hash the prefix leg used,
        # so no fingerprint match is needed to retrieve them.
        segmented_tail: list[CBMatchResult] = []
        if segmented and job.retained_chunks:
            chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
                list(key.token_ids)
            )
            for i in job.retained_chunks:
                if i < prefix_chunks or i >= len(chunk_hashes):
                    continue  # leading run (already prefix) / sub-chunk tail
                st = i * chunk_size
                segmented_tail.append(
                    CBMatchResult(
                        old_st=st,
                        old_ed=st + chunk_size,
                        cur_st=st,
                        cur_ed=st + chunk_size,
                        hash=chunk_hashes[i],
                    )
                )

        seg_tail_tokens = _unique_token_coverage(segmented_tail)
        non_prefix_hit_tokens = _unique_token_coverage(found)
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=rid,
                metadata={
                    "num_tokens": num_tokens,
                    "fingerprint_hits": len(found),
                    "prefix_hits": job.prefix_chunks,
                    "prefix_chunks": job.prefix_chunks,
                    "storage_hits": len(found),
                    "stale_chunks": len(job.non_prefix or []) - len(found),
                    "no_gpu_context": False,
                    "prefix_hit_tokens": prefix_tokens,
                    "segmented_prefix_hit_tokens": seg_tail_tokens,
                    "non_prefix_hit_tokens": non_prefix_hit_tokens,
                    "hit_tokens": prefix_tokens
                    + _unique_token_coverage(found + segmented_tail),
                    "requested_tokens": (num_tokens // chunk_size) * chunk_size,
                },
            )
        )
        with self._cb_jobs_lock:
            self._cb_jobs.pop(rid, None)
        return CBUnifiedLookupResult(
            prefix_coverage_tokens=prefix_tokens,
            non_prefix_segments=found,
            segmented_prefix_segments=segmented_tail,
        )

    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Paged store, then register the stored chunks as match fingerprints.

        Delegates the KV write to ``LMCacheDrivenTransfer.store``, then (worker 0 only)
        enqueues the chunk hashes for async fingerprint registration ordered
        after the L1 commit. Chunk 0 of a position-0 store is skipped (owned by
        the standard prefix path). Fingerprint failures are logged, never
        raised — they do not affect store correctness.

        Args:
            key (IPCCacheServerKey): Store key (token IDs + ``[start, end)``).
            instance_id (int): Target KV-cache instance.
            gpu_block_ids (list[list[int]]): Per-layer-group paged block IDs.
            event_ipc_handle (bytes): IPC handle to the producer's CUDA event.

        Returns:
            tuple[bytes, bool]: The underlying ``LMCacheDrivenTransfer.store`` result
            (event handle, success).
        """
        result = self._transfer_module.store(
            key, instance_id, gpu_block_ids, event_ipc_handle
        )

        # The matcher is engine-shared; only worker 0 registers.
        if key.worker_id not in (0, None):
            return result

        # Enqueue on cupy_stream so CUDA FIFO ordering puts registration
        # after the L1-commit callback; otherwise lookups see the chunk as
        # not-yet-committed and drop the whole group as stale.
        try:
            session = self._ctx.session_manager.get_or_create(key.request_id)
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h)
                for h in session.get_hashes(key.start, key.end)
            ]
            if not chunk_hashes:
                return result
            tokens_in_range = list(key.token_ids)[key.start : key.end]
            start_chunk_idx = 1 if key.start == 0 else 0
            job = (tokens_in_range, chunk_hashes, start_chunk_idx, key.start)
            with self._pending_fp_lock:
                self._pending_fp_hashes.update(chunk_hashes[start_chunk_idx:])
            entry = self._transfer_module.get_and_touch_context_entry(instance_id)
            gpu_ctx = entry.cache_context if entry is not None else None
            if gpu_ctx is not None and gpu_ctx.cupy_stream is not None:
                gpu_ctx.cupy_stream.launch_host_func(
                    self._fingerprint_queue.put_nowait, job
                )
            else:
                self._fingerprint_queue.put_nowait(job)
        except Exception:
            logger.exception(
                "CB fingerprint enqueue failed for request %s "
                "(does not affect store correctness)",
                key.request_id,
            )

        if self._coordinator is not None:
            self._publish_fingerprints(key, chunk_hashes, tokens_in_range)

        return result

    def _publish_fingerprints(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
        tokens_in_range: list[int],
    ) -> None:
        """Publish this stored range's chunk fingerprints to the coordinator.

        Best-effort and fire-and-forget (enqueue only): one wire
        ``ChunkFingerprint`` per stored chunk -- its content poly-hash (the same
        ``chunk_hash_windows_numba`` the match probes, with the fleet base), its
        shared-L2 ``object_key`` (the chunk storage key ``th``), and its token
        position. Never raises into the store path.

        Args:
            key: The store request key (model/scope/positions).
            chunk_hashes: Per-chunk storage keys (``th``) for the range.
            tokens_in_range: The stored tokens ``token_ids[start:end]``.
        """
        coordinator = self._coordinator
        if coordinator is None or not chunk_hashes:
            return
        try:
            model_scope = key.model_name
            store_range = {
                "model_scope": model_scope,
                "tokens": list(tokens_in_range),
                "object_keys": [h.hex() for h in chunk_hashes],
                "old_st_base": key.start,
            }
            coordinator.enqueue_register([store_range])
        except Exception:
            logger.warning(
                "CB coordinator publish build failed for request %s "
                "(does not affect store correctness)",
                key.request_id,
            )

    def _submit_coordinator_match(self, key: IPCCacheServerKey) -> bool:
        """Issue a fleet directory match query for this request (best-effort).

        Args:
            key: The lookup request key.

        Returns:
            ``True`` if a query was submitted (so the finalize step should poll
            for it), ``False`` when there is no coordinator or submission failed.
        """
        coordinator = self._coordinator
        if coordinator is None:
            return False
        try:
            tokens = list(key.token_ids)
            if len(tokens) < self._ctx.chunk_size:
                return False
            coordinator.submit_match(key.request_id, key.model_name, tokens)
            return True
        except Exception:
            logger.warning(
                "CB coordinator match submit failed for request %s", key.request_id
            )
            return False

    def _poll_coordinator_match(
        self, job: "_CBUnifiedJob", rid: str
    ) -> "list[CBMatchResult] | None":
        """Poll the coordinator match result, deferring until it resolves.

        Mirrors the prefix/sparse legs: ``return None`` to defer while pending.
        A per-lookup wall-clock deadline (``job.coord_deadline``) bounds the
        total wait, including queue/pool time. Past the deadline the leg is
        abandoned and the lookup proceeds local-only (the client's later fill,
        if any, is dropped via ``take_match``).

        Args:
            job: The per-request poll state.
            rid: Request id.

        Returns:
            The global segments (possibly empty) once resolved or timed out, or
            ``None`` to defer (still in flight and within the deadline).
        """
        coordinator = self._coordinator
        if coordinator is None or not job.coord_submitted:
            return []
        poll = coordinator.poll_match(rid)
        if poll is PENDING:
            if time.monotonic() < job.coord_deadline:
                return None  # defer; bounded by job.coord_deadline
            coordinator.take_match(rid)
            logger.warning(
                "CB coordinator match deadline exceeded for %s; local-only", rid
            )
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_COORDINATOR_MATCH_END,
                    session_id=rid,
                    metadata={"matches": 0, "timed_out": True},
                )
            )
            return []
        coordinator.take_match(rid)
        segments = self._build_global_segments(poll) if isinstance(poll, list) else []
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_COORDINATOR_MATCH_END,
                session_id=rid,
                metadata={"matches": len(segments), "timed_out": False},
            )
        )
        return segments

    def _build_global_segments(
        self, matches: "list[RemoteMatch]"
    ) -> list[CBMatchResult]:
        """Convert coordinator matches into chunk-granular retrievable segments.

        Each coordinator ``object_key`` is the hex of the chunk's content hash
        (the same ``th`` a local ``CBMatchResult.hash`` holds), so the matches
        are returned as ``CBMatchResult`` directly: the retrieve path then
        expands ``hash`` to per-rank shared-L2 object keys via
        ``ipc_key_to_object_keys``, identical to local matches.

        Args:
            matches: Matched chunks returned by the coordinator client.

        Returns:
            One :class:`CBMatchResult` per matched chunk (request order).
        """
        chunk_size = self._ctx.chunk_size
        return [
            CBMatchResult(
                old_st=m.old_st,
                old_ed=m.old_st + chunk_size,
                cur_st=m.cur_st,
                cur_ed=m.cur_st + chunk_size,
                hash=bytes.fromhex(m.object_key),
            )
            for m in matches
        ]

    def _drain_fingerprint_queue(self) -> None:
        """Best-effort background drainer for _fingerprint_queue."""
        while not self._fingerprint_stop.is_set():
            try:
                job = self._fingerprint_queue.get(timeout=0.1)
            except QueueEmpty:
                continue
            tokens_in_range, chunk_hashes, start_chunk_idx, position_offset = job
            try:
                self._token_range_matcher.on_new_token_hashes(
                    tokens_in_range,
                    chunk_hashes,
                    start_chunk_idx=start_chunk_idx,
                    position_offset=position_offset,
                )
            except Exception:
                logger.exception("CB fingerprint registration failed (async)")
            finally:
                with self._pending_fp_lock:
                    self._pending_fp_hashes.difference_update(
                        chunk_hashes[start_chunk_idx:]
                    )

    def _apply_cb_rope_batched(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        batch_len: int,
        slots_to_rope: list[tuple[int, int, int]],
    ) -> None:
        """Re-RoPE the given tmp-pool slots in place (K-only, per kernel group).

        Args:
            gpu_context (GPUCacheContext): The instance's GPU cache context.
            rope_state (_CBRopeState): Cached cos/sin + head layout.
            batch_len (int): Number of tmp slots staged for this batch.
            slots_to_rope (list[tuple[int, int, int]]): ``(slot_idx, old_st,
                cur_st)`` per shifted slot — re-RoPE K from stored position
                ``old_st`` to new position ``cur_st``.

        Raises:
            RuntimeError: On a compressed (compress_ratio != 1) layout, a
                kv_size other than 2 (K/V) or 1 (key-only index), or a
                head_size/hidden_dim mismatch.
        """
        if not slots_to_rope:
            return
        num_groups = gpu_context.kv_layer_groups_manager.num_kernel_groups
        for group_idx in range(num_groups):
            group = gpu_context.kv_layer_groups_manager.kernel_groups[group_idx]
            all_slots = [
                gpu_context.get_temp_kernel_group_buffer(slot_idx, group_idx)
                for slot_idx in range(batch_len)
            ]
            rot = rope_state.rot_for_group(group.engine_group_idx, all_slots[0].dtype)
            if rot is None:
                # Skipped group (declared [], or quantized under a declared
                # map): scattered as-is, positions left stale.
                continue
            num_layers, slots, hidden_dim = all_slots[0].shape[1:]
            fused_packed, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
                group,
                int(all_slots[0].shape[0]),
                int(hidden_dim),
                rope_state.head_size,
                group_idx,
                rot,
            )
            # Per-group rope cache: dual-RoPE models rotate each
            # kernel group with its own theta's cos/sin.
            group_cos_sin = rope_state.cache_for_group(group.engine_group_idx)
            if rot_offset > 0 and int(group_cos_sin.shape[1]) != rot[1]:
                raise RuntimeError(
                    f"CB re-RoPE: group {group_idx} declares rope width "
                    f"{rot[1]} but the cos/sin cache has rot_dim "
                    f"{int(group_cos_sin.shape[1])}."
                )
            # slot ramp tiled across layers is invariant per (num_layers,
            # slots) — cache it; each shifted slot then just adds its offset.
            device = all_slots[0].device
            sp_key = (str(device), num_layers, slots)
            sp_cache = getattr(self, "_cb_sp_rep_cache", None)
            if sp_cache is None:
                sp_cache = {}
                self._cb_sp_rep_cache = sp_cache
            slot_positions_rep = sp_cache.get(sp_key)
            if slot_positions_rep is None:
                slot_positions_rep = torch.arange(
                    slots, device=device, dtype=torch.long
                ).repeat(num_layers)
                sp_cache[sp_key] = slot_positions_rep
            for slot_idx, old_st, cur_st in slots_to_rope:
                # reshape returns an in-place view (tmp slots are contiguous).
                k_view = all_slots[slot_idx][0].reshape(
                    num_layers * slots, n_heads, per_head
                )
                if rot_offset > 0:
                    # MLA latent: rotate only the trailing rope window. The
                    # slice advances data_ptr to the window start; the kernel
                    # addresses rows via the explicit head_stride (the full
                    # row width), so the non-contiguous view is safe. It then
                    # rotates rot_dim (= window width) dims from that base —
                    # the content dims [0, rot_offset) are never touched.
                    lmc_ops.rotary_embedding_k_fused_strided(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view[..., rot_offset:],
                        per_head - rot_offset,  # window width
                        per_head,  # head_stride: the full latent row
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )
                elif fused_packed:
                    # Strided kernel rotates only the K half of each slot.
                    lmc_ops.rotary_embedding_k_fused_strided(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view,
                        rope_state.head_size,
                        per_head,  # head_stride: hop over the packed V half
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )
                else:
                    lmc_ops.rotary_embedding_k_fused(
                        old_st + slot_positions_rep,
                        cur_st + slot_positions_rep,
                        k_view,
                        rope_state.head_size,
                        group_cos_sin,
                        rope_state.is_neox_style,
                    )

    def _scatter_batch_to_paged(
        self,
        gpu_context: BaseCacheContext,
        resolved_groups: "list[tuple[torch.Tensor, int]]",
        batch: "list[tuple[CBMatchResult, Any]]",
        head_size: int,
    ) -> None:
        """Scatter one tmp-slot batch into the paged KV, one launch per
        (kernel group, tmp slot), straight from each slot's contiguous buffer
        (no ``torch.cat``). A partially filled slot is narrowed first: the
        kernel scatters ``size(2)`` tokens, so a full-capacity buffer would
        mis-align every later slot against ``slot_mapping``.

        Args:
            gpu_context (GPUCacheContext): The instance's GPU cache context.
            resolved_groups: Per kernel group ``(block_ids, block_size)``.
            batch: ``(match, memory_obj)`` pairs; slot ``i`` holds ``batch[i]``.
            head_size: RoPE head size forwarded to the kernel.
        """
        kgm = gpu_context.kv_layer_groups_manager
        tok_counts = [int(r.cur_ed - r.cur_st) for (r, _) in batch]
        pos = torch.cat(
            [
                torch.arange(
                    r.cur_st,
                    r.cur_ed,
                    device=gpu_context.device,
                    dtype=torch.long,
                )
                for (r, _) in batch
            ]
        )
        slot_mappings = _group_slot_mappings(resolved_groups, pos)
        for group_idx in range(kgm.num_kernel_groups):
            _, group_bs = resolved_groups[group_idx]
            slot_mapping = slot_mappings[group_idx]
            # Per-group block count: under HMA the sliding group has fewer
            # blocks than the full group, so gpu_context.num_blocks (group
            # 0's) would truncate the other groups' bounds check.
            page_buffer_size = kgm.kernel_groups[group_idx].shape_desc.nb * group_bs
            tok_off = 0
            for slot_idx, n_tok in enumerate(tok_counts):
                key_value = gpu_context.get_temp_kernel_group_buffer(
                    slot_idx, group_idx
                )
                if n_tok < key_value.shape[2]:
                    # Partial chunk: narrow to the real token count (the
                    # kernel scatters size(2) tokens). Slicing dim 2 breaks
                    # contiguity, so this one slot pays a small copy.
                    key_value = key_value[:, :, :n_tok].contiguous()
                lmc_ops.multi_layer_kv_transfer(
                    key_value,
                    gpu_context.get_kernel_group_kv_pointers(group_idx),
                    slot_mapping[tok_off : tok_off + n_tok],
                    gpu_context.device,
                    page_buffer_size,
                    lmc_ops.TransferDirection.H2D,
                    gpu_context.get_engine_kv_format(group_idx),
                    block_size=group_bs,
                    head_size=head_size,
                )
                tok_off += n_tok

    def _cb_slot_buffers(
        self, gpu_context: BaseCacheContext, num_groups: int, n_pos: int
    ) -> "tuple[torch.Tensor, Any, torch.Tensor]":
        """Return ``(pinned, pinned_np, device)`` slot-mapping staging of at
        least ``(num_groups, n_pos)``, reused across requests per context."""
        entry = self._cb_slot_staging.get(gpu_context)
        if entry is None or entry[0].shape[0] < num_groups or entry[0].shape[1] < n_pos:
            cap = max(n_pos, 1 << 16)
            if entry is not None:
                cap = max(cap, int(entry[0].shape[1]))
            # Pin only for CUDA contexts (CPU-device unit tests have no CUDA).
            pinned = torch.empty(
                (num_groups, cap),
                dtype=torch.int64,
                pin_memory=(gpu_context.device.type == "cuda"),
            )
            dev = torch.empty(
                (num_groups, cap), dtype=torch.int64, device=gpu_context.device
            )
            entry = (pinned, pinned.numpy(), dev)
            self._cb_slot_staging[gpu_context] = entry
        return entry

    def _resolve_cb_plan_invariants(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        max_batch: int,
    ) -> "tuple[list[Any], list[torch.Tensor]] | None":
        """Resolve the request-invariant plan half (cached per context in
        ``_cb_plan_invariants``).

        Returns ``(group_specs, object_group_buffers)`` with slot-mapping
        fields as placeholders for the per-request stamp, or ``None`` on an
        unsupported layout (compressed / kv_size / dtype / head geometry).
        """
        kgm = gpu_context.kv_layer_groups_manager
        group_specs: list[Any] = []
        for group_idx in range(kgm.num_kernel_groups):
            group = kgm.kernel_groups[group_idx]
            buf0 = gpu_context.get_temp_kernel_group_buffer(0, group_idx)
            num_layers, slot_tokens, hidden_dim = (
                int(buf0.shape[1]),
                int(buf0.shape[2]),
                int(buf0.shape[3]),
            )
            group_bs = group.tokens_per_block
            spec_common = dict(
                paged_kv_ptrs=gpu_context.get_kernel_group_kv_pointers(
                    group_idx
                ).data_ptr(),
                temp_buffer_ptrs=[
                    gpu_context.get_temp_kernel_group_buffer(slot, group_idx).data_ptr()
                    for slot in range(max_batch)
                ],
                num_layers=num_layers,
                slot_tokens=slot_tokens,
                hidden_elems=hidden_dim,
                element_size=buf0.element_size(),
                engine_kv_format=gpu_context.get_engine_kv_format(group_idx),
                page_buffer_size=group.shape_desc.nb * group_bs,
                block_size=group_bs,
                head_size=rope_state.head_size,
                slot_mapping_base=0,
                slot_mapping_capacity=0,
                is_neox=rope_state.is_neox_style,
            )
            rot = rope_state.rot_for_group(group.engine_group_idx, buf0.dtype)
            if rot is None:
                # Skipped group (declared [] or quantized): staging + scatter
                # only. cos_sin_cache == 0 disables native rope, so the
                # rope-only fields are never read and the dtype gate below
                # must not run (a uint8 group would knock every group off
                # the native plan).
                group_specs.append(
                    lmc_ops.CBGroupSpec(
                        cos_sin_cache=0,
                        rot_dim=0,
                        rope_num_kv_heads=1,
                        rope_head_stride=hidden_dim,
                        key_scalar_type=0,
                        **spec_common,
                    )
                )
                continue
            at_scalar = _TORCH_TO_AT_SCALAR.get(buf0.dtype)
            if at_scalar is None:
                return None
            try:
                # Same rules as _apply_cb_rope_batched (shared helper); the
                # planner declines instead of raising -- the Python fallback
                # handles (or reports) the layout.
                _fused, per_head, n_heads, rot_offset = _cb_group_rope_geometry(
                    group,
                    int(buf0.shape[0]),
                    hidden_dim,
                    rope_state.head_size,
                    group_idx,
                    rot,
                )
            except RuntimeError:
                return None
            group_cos_sin = rope_state.cache_for_group(group.engine_group_idx)
            if rot_offset > 0 and int(group_cos_sin.shape[1]) != rot[1]:
                return None

            group_specs.append(
                lmc_ops.CBGroupSpec(
                    cos_sin_cache=group_cos_sin.data_ptr(),
                    rot_dim=int(group_cos_sin.shape[1]),
                    rope_num_kv_heads=n_heads,
                    rope_head_stride=per_head,
                    key_scalar_type=at_scalar,
                    rope_base_offset=rot_offset * buf0.element_size(),
                    **spec_common,
                )
            )
        object_group_buffers = [
            gpu_context.get_temp_object_group_buffer(slot, 0)
            for slot in range(max_batch)
        ]
        return group_specs, object_group_buffers

    def _build_cb_retrieve_plan_flat(
        self,
        gpu_context: BaseCacheContext,
        rope_state: _CBRopeState,
        cpu_block_tables: "list[tuple[np.ndarray, int]]",
        runs: "list[list[tuple[CBMatchResult, Any]]]",
        max_batch: int,
    ) -> "tuple[list[Any], tuple[Any, Any, Any, Any], list[torch.Tensor]] | None":
        """Build the whole native retrieve plan: eligibility gates, cached
        invariant specs stamped with this request's slot mappings, and the
        numpy-vectorized int64 work tables for
        ``execute_cb_retrieve_plan_flat`` (layouts in the pybind docstring).

        Returns:
            ``(group_specs, (staging, ropes, scatters, step_offsets),
            keepalive)`` or ``None`` -> Python fallback loop.
        """
        if not _HAS_NATIVE_RETRIEVE_PLAN or max_batch < 2:
            return None
        pairs = [pair for run in runs for pair in run]
        if not pairs:
            return None
        # Native staging requires the lazy-allocator (pin-chunked) host path.
        for _, memory_obj in pairs:
            if not isinstance(memory_obj.parent(), LazyMemoryAllocator):
                return None

        # Specs are invariant per paged registration except the slot-mapping
        # fields: cache them per context, re-stamp per request (the full
        # resolve costs dozens of torch-view creations under the shared GIL).
        # The cached rope_state reference doubles as the validity check: a
        # re-registration swaps the object, so identity comparison is sound.
        cached = self._cb_plan_invariants.get(gpu_context)
        if cached is not None and not (
            cached[0] is rope_state and cached[1] == max_batch
        ):
            cached = None
        if cached is None:
            resolved = self._resolve_cb_plan_invariants(
                gpu_context, rope_state, max_batch
            )
            if resolved is None:
                return None
            cached = (rope_state, max_batch, resolved)
            self._cb_plan_invariants[gpu_context] = cached
        group_specs, object_group_buffers = cached[2]
        num_groups = len(group_specs)
        wave = max_batch // 2

        n = len(pairs)
        # One row per chunk: cur_st, cur_ed, old_st, src ptr, nbytes, host off.
        chunk_table = np.array(
            [
                (
                    r.cur_st,
                    r.cur_ed,
                    r.old_st,
                    memory_obj.data_ptr,
                    memory_obj.get_size(),
                    memory_obj.meta.address,
                )
                for r, memory_obj in pairs
            ],
            dtype=np.int64,
        )
        cur_st, cur_ed, old_st = (
            chunk_table[:, 0],
            chunk_table[:, 1],
            chunk_table[:, 2],
        )
        src, nbytes, host_off = (
            chunk_table[:, 3],
            chunk_table[:, 4],
            chunk_table[:, 5],
        )
        buf_bytes = int(object_group_buffers[0].nbytes)
        if (nbytes != buf_bytes).any():
            # Size mismatch: the fallback path raises the descriptive error.
            return None

        # Shared logical positions + per-group slot mappings, in numpy on
        # pinned staging with one async H2D per group (persistent buffers).
        # The old device-side arange/div/mod chain cost 25-160 ms per request
        # in CUDA alloc/sync contention with the engine context; this path is
        # sub-ms CPU math + copies that ride the ambient stream (FIFO before
        # the native exec's kernels).
        run_iter = [run for run in runs if run]
        if len(run_iter) == 1:
            pos_np = np.arange(
                run_iter[0][0][0].cur_st, run_iter[0][-1][0].cur_ed, dtype=np.int64
            )
        else:
            pos_np = np.concatenate(
                [
                    np.arange(run[0][0].cur_st, run[-1][0].cur_ed, dtype=np.int64)
                    for run in run_iter
                ]
            )
        n_pos = int(pos_np.shape[0])
        pinned, pinned_np, dev_buf = self._cb_slot_buffers(
            gpu_context, num_groups, n_pos
        )
        div_mod: "dict[int, tuple[np.ndarray, np.ndarray]]" = {}
        for gi, ((block_ids_np, group_bs), spec) in enumerate(
            zip(cpu_block_tables, group_specs, strict=True)
        ):
            pair = div_mod.get(group_bs)
            if pair is None:
                pair = np.divmod(pos_np, group_bs)
                div_mod[group_bs] = pair
            q, rem = pair
            out = pinned_np[gi, :n_pos]
            np.multiply(block_ids_np[q], group_bs, out=out)
            out += rem
            dev_buf[gi, :n_pos].copy_(pinned[gi, :n_pos], non_blocking=True)
            # Safe to mutate: one handler per context, and the native call
            # copies spec contents at call time.
            spec.slot_mapping_base = int(dev_buf[gi].data_ptr())
            spec.slot_mapping_capacity = n_pos
        # The device staging must outlive the native call.
        keepalive = [dev_buf]

        # Waves of `wave` chunks per run, alternating slot halves.
        slot_of = np.empty(n, dtype=np.int64)
        slot_arange = np.arange(wave, dtype=np.int64)
        step_lens: list[int] = []
        i0 = 0
        for run in runs:
            m = len(run)
            for w0 in range(0, m, wave):
                batch_len = min(wave, m - w0)
                slot_base = (len(step_lens) % 2) * wave
                slot_of[i0 : i0 + batch_len] = slot_base + slot_arange[:batch_len]
                step_lens.append(batch_len)
                i0 += batch_len
        chunks_per_step = np.asarray(step_lens, dtype=np.int64)
        n_steps = len(step_lens)

        n_tok = cur_ed - cur_st
        tok_off = np.zeros(n, dtype=np.int64)
        np.cumsum(n_tok[:-1], out=tok_off[1:])

        obj_buf_ptrs = np.asarray(
            [buf.data_ptr() for buf in object_group_buffers], dtype=np.int64
        )
        staging = np.stack([obj_buf_ptrs[slot_of], src, nbytes, host_off], axis=1)

        groups_arr = np.arange(num_groups, dtype=np.int64)
        shifted = old_st != cur_st
        n_shifted = int(shifted.sum())
        ropes = np.stack(
            [
                np.tile(groups_arr, n_shifted),
                np.repeat(slot_of[shifted], num_groups),
                np.repeat(old_st[shifted], num_groups),
                np.repeat(cur_st[shifted], num_groups),
            ],
            axis=1,
        )
        scatters = np.stack(
            [
                np.tile(groups_arr, n),
                np.repeat(slot_of, num_groups),
                np.repeat(tok_off, num_groups),
                np.repeat(n_tok, num_groups),
            ],
            axis=1,
        )

        staging_end = np.cumsum(chunks_per_step)
        step_of_chunk = np.repeat(np.arange(n_steps, dtype=np.int64), chunks_per_step)
        shifted_per_step = np.bincount(
            step_of_chunk[shifted], minlength=n_steps
        ).astype(np.int64, copy=False)
        step_offsets = np.stack(
            [
                staging_end,
                np.cumsum(shifted_per_step) * num_groups,
                staging_end * num_groups,
            ],
            axis=1,
        )
        return (
            group_specs,
            (staging, ropes, scatters, step_offsets),
            keepalive,
        )

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        gpu_block_ids: list[list[int]],
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Scatter every matched token range into the request's paged KV.

        Reuses the lookup's prefetched chunks: fills tmp slots, K-only re-RoPEs
        the shifted (non-prefix) subset, then writes per-token via the slot
        kernel — so non-block-aligned matches and partial vLLM blocks shared
        with recomputed tokens are written correctly (no block-alignment trim).
        Only matches past the currently allocated slots are dropped (vLLM may
        call this twice: partial- then full-block alloc).

        Args:
            key (IPCCacheServerKey): The request key.
            cb_match_result (list[CBMatchResult]): Matched ranges to scatter
                (prefix-hit and shifted), any order.
            gpu_block_ids (list[list[int]]): This request's paged block table
                per engine (kernel) group; single-group models pass [[...]].
                Mirrors the engine RETRIEVE/STORE per-group block-id contract.
            instance_id (int): Target KV-cache instance.
            event_ipc_handle (bytes): IPC handle to the forward's CUDA event.

        Returns:
            tuple[bytes, bool]: The scatter-complete event handle and whether
            the scatter ran (False if the prefetched objects were unavailable).

        Raises:
            ValueError: If the instance has no registered KV cache or rope
                state. MLA layouts are unsupported (raised during re-RoPE).
        """
        entry = self._transfer_module.get_and_touch_context_entry(instance_id)
        if entry is None:
            raise ValueError(
                f"Instance {instance_id} not registered for paged KV cache"
            )
        if instance_id not in self._cb_rope_state:
            raise ValueError(
                f"Instance {instance_id} has no CB rope state; "
                "send CB_REGISTER_ROPE_V3 before CB_RETRIEVE_PRE_COMPUTED_V3."
            )
        gpu_context = entry.cache_context
        rope_state = self._cb_rope_state[instance_id]
        chunk_size = self._ctx.chunk_size

        _retrieve_t0 = time.perf_counter()

        def _noop_success(reason: str = "?") -> tuple[bytes, bool]:
            """Zero-work success return: must export a fresh recorded event
            from THIS process -- echoing the caller's own handle back makes
            the worker re-import it (CUDA "invalid device context").

            ``reason`` is logged once per distinct value. Every one of these
            paths silently turns a matched request into a full recompute --
            recall stays correct, only the speedup vanishes -- so a run that
            drops all its matches is otherwise indistinguishable from a
            working one except by its TTFT. Logging the reason is what makes
            "CB engaged but was not faster" diagnosable.
            """
            if reason not in _NOOP_REASONS_SEEN:
                _NOOP_REASONS_SEEN.add(reason)
                logger.info(
                    "CB v3 retrieve: no-op success (%s) — %d match(es) dropped, "
                    "request falls back to full recompute. Logged once per "
                    "distinct reason.",
                    reason,
                    len(cb_match_result),
                )
            with (
                torch_dev.device(gpu_context.device),
                torch_dev.stream(gpu_context.stream),
            ):
                check_interprocess_event_support()
                done_event = torch_dev.Event(interprocess=True)
                done_event.record()
                handle = done_event.ipc_handle()
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_REQUEST_END,
                    session_id=key.request_id,
                )
            )
            return handle, True

        cb_match_result = sorted(cb_match_result, key=lambda r: r.cur_st)
        # vLLM may call retrieve twice (partial- then full-block alloc); skip
        # ranges already scattered (blocks never move mid-prefill), returning
        # before the obj-key/prefetched-read machinery (~7-20 ms).
        applied_ranges = self._cb_applied_match_ranges
        applied_key = (key.request_id, key.worker_id)
        prior_applied = applied_ranges.get(applied_key)
        if prior_applied:
            cb_match_result = [
                r
                for r in cb_match_result
                if (r.hash, r.cur_st, r.cur_ed) not in prior_applied
            ]
            if not cb_match_result:
                return _noop_success("all ranges already applied for this worker")
        applied_now: "set[tuple[bytes, int, int]]" = set()
        # Partial-alloc first call: every match can be beyond the allocated
        # slots -> return before the obj-key machinery. Read locks stay held
        # for the full-alloc follow-up, as the in-loop drop path leaves them.
        if cb_match_result:
            try:
                slot_bound = min(
                    len(gpu_block_ids[kg.engine_group_idx]) * kg.tokens_per_block
                    for kg in gpu_context.kv_layer_groups_manager.kernel_groups
                )
            except (IndexError, TypeError):
                slot_bound = None
            if slot_bound is not None and all(
                r.cur_ed > slot_bound for r in cb_match_result
            ):
                return _noop_success(f"every match beyond slot_bound={slot_bound}")
        # L2 opt: reuse lookup's obj_keys cache; fall back to re-resolve.
        with self._lookup_obj_keys_lock:
            cached = self._lookup_obj_keys_cache.pop(key.request_id, None)
        if cached is not None and all(r.hash in cached for r in cb_match_result):
            # The lookup cached all-ranks obj keys (world_size per hash). This
            # retrieve is per-worker, so select THIS rank's key -> M objects, not
            # M*world_size (else the zip below silently truncates and mispairs
            # ranks at TP>1). Mirrors the non-cached path's per-worker resolve.
            if key.worker_id is not None and key.world_size > 1:
                all_obj_keys = [cached[r.hash][key.worker_id] for r in cb_match_result]
            else:
                all_obj_keys = [k for r in cb_match_result for k in cached[r.hash]]
        else:
            all_obj_keys = ipc_key_to_object_keys(
                key, [r.hash for r in cb_match_result], [0]
            )[0]

        # Lookup read-locked the full found set, but the connector may have
        # dropped some matches (parent-covered / misaligned) before retrieve,
        # leaking their per-key read locks. Release those orphans now (disjoint
        # from all_obj_keys, which retrieve still consumes; needs the key cache).
        if cached is not None:
            retrieved_hashes = {r.hash for r in cb_match_result}
            orphan_keys = [
                k for h, ks in cached.items() if h not in retrieved_hashes for k in ks
            ]
            if orphan_keys:
                self._ctx.storage_manager.finish_read_prefetched(orphan_keys)
                logger.debug(
                    "CB V3 released %d prefetched-but-unretrieved keys (req=%s)",
                    len(orphan_keys),
                    key.request_id,
                )

        # Non-prefix sparse hits split by re-rope need (not prefix coverage).
        n_non_shifted = sum(1 for r in cb_match_result if r.old_st == r.cur_st)
        n_shifted = len(cb_match_result) - n_non_shifted

        if not all_obj_keys:
            # Same latent hazard as the guards above: this used to echo the
            # caller's own event handle back.
            return _noop_success("no object keys resolved for the matches")

        logger.debug("CB V3 retrieving object keys: %s", all_obj_keys)

        # CB v3 only supports uncompressed single-block-id-space layouts
        # (enforced per group in ``_apply_cb_rope_batched``), so the first
        # kernel group's chunk geometry is representative.
        tokens_per_block = gpu_context.kv_layer_groups_manager.kernel_groups[
            0
        ].tokens_per_block
        if chunk_size % tokens_per_block != 0:
            raise ValueError(
                f"chunk_size {chunk_size} must be a multiple of "
                f"tokens_per_block {tokens_per_block}"
            )
        num_groups = gpu_context.kv_layer_groups_manager.num_kernel_groups

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            # One staged block-id tensor per engine group, indexed by group.
            block_ids_per_group_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            # Resolve each kernel group's block table + block size once. Select
            # by engine_group_idx (kernel groups may share one, e.g. MiniMax-M3).
            # The CPU (numpy) tables feed the native plan's slot-mapping math;
            # the GPU tensors feed the Python fallback loop.
            kgm = gpu_context.kv_layer_groups_manager
            block_ids_np = [np.asarray(b, dtype=np.int64) for b in gpu_block_ids]
            resolved_groups: list[tuple[torch.Tensor, int]] = []
            cpu_block_tables: "list[tuple[np.ndarray, int]]" = []
            for group_idx in range(num_groups):
                eg_idx = kgm.kernel_groups[group_idx].engine_group_idx
                if eg_idx >= len(block_ids_per_group_gpu):
                    # Engine groups have independent block tables under HMA;
                    # substituting another group's table would scatter KV into
                    # the wrong physical blocks (silent corruption).
                    raise ValueError(
                        f"CB retrieve: kernel group {group_idx} maps to engine "
                        f"group {eg_idx}, but only "
                        f"{len(block_ids_per_group_gpu)} block table(s) were "
                        "provided."
                    )
                group_bs = kgm.kernel_groups[group_idx].tokens_per_block
                resolved_groups.append((block_ids_per_group_gpu[eg_idx], group_bs))
                cpu_block_tables.append((block_ids_np[eg_idx], group_bs))

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id=key.request_id,
                    metadata={
                        "num_chunks": len(cb_match_result),
                        "model_name": key.model_name,
                    },
                ),
            )

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC "
                    "event handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # Stage marks for the scatter_ms log line (CPU enqueue wall time):
            # fetch = L1 prefetched read, plan = flat-plan table build,
            # exec = native kernel enqueue (H2D + re-RoPE + scatter).
            _stage_ms: dict[str, float] = {}
            _stage_t = time.perf_counter()
            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    all_obj_keys
                ) as memory_objs:
                    _stage_ms["fetch"] = (time.perf_counter() - _stage_t) * 1000
                    if memory_objs is None:
                        # Read failed: return a valid server event + False, never
                        # the client's own handle (self-import raises
                        # cudaErrorDeviceUninitialized, crashing TP).
                        event.record()
                        return event.ipc_handle(), False

                    # Per-token scatter handles any cur_st; just bound the
                    # matched range to the allocated slots.
                    pairs: list[tuple[CBMatchResult, Any]] = []
                    # Bound by the smallest group: under HMA the sliding group
                    # has fewer blocks than the full group, so [0] isn't safe.
                    num_slots = min(
                        int(block_ids.numel()) * group_bs
                        for block_ids, group_bs in resolved_groups
                    )
                    for r, memory_obj in zip(cb_match_result, memory_objs, strict=True):
                        if r.cur_ed > num_slots:
                            logger.warning(
                                "Dropping CB match cur_st=%d cur_ed=%d: exceeds "
                                "%d slots. Request %s.",
                                r.cur_st,
                                r.cur_ed,
                                num_slots,
                                key.request_id,
                            )
                            continue
                        pairs.append((r, memory_obj))

                    # cb.scatter span (GPU): the L1->paged write of every
                    # applied match. Re-RoPE is folded in (n_shifted) — it is
                    # interleaved per-batch, so not a separate span.
                    self._event_bus.publish_on_stream(
                        gpu_context.cupy_stream,
                        Event(
                            event_type=EventType.CB_SCATTER_START,
                            session_id=key.request_id,
                            metadata={
                                "scattered_tokens": sum(
                                    r.cur_ed - r.cur_st for r, _ in pairs
                                ),
                                "n_prefix": sum(
                                    1 for r, _ in pairs if r.old_st == r.cur_st
                                ),
                                "n_shifted": sum(
                                    1 for r, _ in pairs if r.old_st != r.cur_st
                                ),
                                "dropped": len(cb_match_result) - len(pairs),
                            },
                        ),
                    )

                    # Consecutive matches → one batched scatter per group.
                    runs: list[list[tuple[CBMatchResult, Any]]] = []
                    for r_obj in pairs:
                        r = r_obj[0]
                        if runs and runs[-1][-1][0].cur_ed == r.cur_st:
                            runs[-1].append(r_obj)
                        else:
                            runs.append([r_obj])

                    max_batch = gpu_context.max_batch_size

                    # Fast path: one native call for the whole request; the
                    # per-wave Python loop is the fallback (returns None on old
                    # c_ops, non-lazy objects, max_batch < 2, or size mismatch).
                    _stage_t = time.perf_counter()
                    native_flat = self._build_cb_retrieve_plan_flat(
                        gpu_context, rope_state, cpu_block_tables, runs, max_batch
                    )
                    _stage_ms["plan"] = (time.perf_counter() - _stage_t) * 1000
                    if native_flat is not None:
                        plan_group_specs, plan_tables, _plan_keepalive = native_flat
                        _stage_t = time.perf_counter()
                        lmc_ops.execute_cb_retrieve_plan_flat(
                            gpu_context.device,
                            LazyMemoryAllocator.PIN_CHUNK_SIZE,
                            plan_group_specs,
                            *plan_tables,
                        )
                        _stage_ms["exec"] = (time.perf_counter() - _stage_t) * 1000
                        runs = []  # plan covers every wave; skip the loop

                    for run in runs:
                        for batch_start in range(0, len(run), max_batch):
                            batch = run[batch_start : batch_start + max_batch]
                            batch_len = len(batch)

                            # (a) H2D fill into per-chunk tmp slots.
                            for slot_idx, (_, memory_obj) in enumerate(batch):
                                # Single object group => object_group_idx=0.
                                flat_slot = gpu_context.get_temp_object_group_buffer(
                                    slot_idx, 0
                                )
                                lmcache_memcpy_async_h2d(memory_obj, flat_slot)

                            # (b) Re-RoPE shifted (non-prefix) slots in place.
                            slots_to_rope = [
                                (slot_idx, r.old_st, r.cur_st)
                                for slot_idx, (r, _) in enumerate(batch)
                                if r.old_st != r.cur_st
                            ]
                            self._apply_cb_rope_batched(
                                gpu_context, rope_state, batch_len, slots_to_rope
                            )

                            # (c) Per-token slot scatter: partial vLLM blocks
                            # shared with recomputed tokens stay disjoint.
                            self._scatter_batch_to_paged(
                                gpu_context,
                                resolved_groups,
                                batch,
                                rope_state.head_size,
                            )

                    applied_now = {(r.hash, r.cur_st, r.cur_ed) for r, _ in pairs}

                    self._event_bus.publish_on_stream(
                        gpu_context.cupy_stream,
                        Event(
                            event_type=EventType.CB_SCATTER_END,
                            session_id=key.request_id,
                        ),
                    )
            except Exception:
                logger.exception("Error during retrieving prefetched results")
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={"success": False},
                    ),
                )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_REQUEST_END,
                        session_id=key.request_id,
                    ),
                )
                # Valid server event + False (never echo the client handle; see
                # the memory_objs-None path above).
                event.record()
                return event.ipc_handle(), False

            event.record()
            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_END,
                    session_id=key.request_id,
                    metadata={"success": True},
                ),
            )

        # Record scattered ranges for the repeat-call guard (bounded LRU).
        if applied_now:
            applied_entry = applied_ranges.setdefault(applied_key, set())
            applied_entry.update(applied_now)
            applied_ranges.move_to_end(applied_key)
            while len(applied_ranges) > 4096:
                applied_ranges.popitem(last=False)

        _scatter_ms = (time.perf_counter() - _retrieve_t0) * 1000
        logger.info(
            "Retrieved pre-computed for %d match results into request %s "
            "paged blocks (scatter_ms=%.2f, non_shifted=%d shifted=%d, "
            "stages_ms=%s)",
            len(cb_match_result),
            key.request_id,
            _scatter_ms,
            n_non_shifted,
            n_shifted,
            {k: round(v, 1) for k, v in _stage_ms.items()},
        )
        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=key.request_id,
            ),
        )
        return event.ipc_handle(), True
