# SPDX-License-Identifier: Apache-2.0
"""Blend (context-blend / cross-request KV reuse) module for MPCacheServer."""

# Standard
from typing import Any
import threading
import time

# Third Party
import numpy as np

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import check_interprocess_event_support
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.token_hasher import (
    chunk_hash_windows_numba,
    rolling_hash_windows_numba,
    unique_hits_direct_id_numba,
    update_table_id_numba,
)
from lmcache.v1.platform.cuda.cache_context import PlainGPUCacheContext

logger = init_logger(__name__)


class BlendTokenRangeMatcher:
    """Fast token-range matcher using polynomial rolling/chunk hashes and a
    direct-address lookup table.

    Table layout: poly_chunk_hash (u64) -> compact_chunk_id (i64, sequential 0...N-1).

    Because compact IDs are bounded by _TABLE_SIZE, unique_hits_direct_id_numba
    can use a fixed ``seen`` array of _TABLE_SIZE bytes (~1 MB) rather than one
    sized by an arbitrary max hash -- no memory explosion.

    Auxiliary storage:
      _chunk_token_hash[i]      : token_hash for chunk i (None if evicted)
      _token_hash_to_start      : token_hash -> start position in seq
      _compact_id_to_slot[i]    : table slot for compact_id i
      _token_hash_to_compact_id : token_hash -> compact_chunk_id

    Methods:
      on_new_token_hashes  -- register a sequence; builds fingerprints
                              and writes compact IDs.
      match_sub_sequence   -- sliding-window probe -> compact IDs ->
                              token_hash -> start. Skips evicted entries.
      remove_chunks        -- lazily evict stale entries. Clears the
                              table slot and auxiliary maps.

    Args:
        chunk_size: Number of tokens per chunk for fingerprint computation.
    """

    _TABLE_BITS: int = 20  # 2^20 ~ 1 M entries
    _TABLE_SIZE: int = 1 << _TABLE_BITS
    _BASE: np.uint64 = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci-hashing constant

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        # poly_chunk_hash -> compact_chunk_id; -1 = empty
        self._table_id = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        self._mask = np.uint64(self._TABLE_SIZE - 1)
        # compact_chunk_id -> caller-supplied token_hash (full bytes)
        self._chunk_token_hash: list[bytes | None] = []
        # token_hash -> start position in its registered sequence
        self._token_hash_to_start: dict[bytes, int] = {}
        # compact_chunk_id -> table slot index (for reverse lookup during eviction)
        self._compact_id_to_slot = np.full(self._TABLE_SIZE, -1, dtype=np.int64)
        # token_hash -> compact_chunk_id (for eviction lookup)
        self._token_hash_to_compact_id: dict[bytes, int] = {}
        self._lock = threading.Lock()

    def on_new_token_hashes(
        self,
        token_ids: list[int],
        token_hashes: list[bytes],
    ) -> None:
        """Register a new token sequence and index its non-overlapping chunks.

        Args:
            token_ids: Raw token IDs for the full sequence (num_tokens elements).
                       Used to compute polynomial chunk fingerprints that match
                       the rolling hashes computed in match_sub_sequence.
            token_hashes: Per-chunk bytes hashes supplied by the caller
                          (one per complete chunk of chunk_size tokens).
                          Stored as the storage key returned in CBMatchResult.hash.
        """
        arr = np.array(token_ids, dtype=np.uint64)
        # Polynomial fingerprints for non-overlapping chunks, built from raw
        # token IDs so they match the rolling hashes in match_sub_sequence
        chunk_hashes = chunk_hash_windows_numba(arr, self.chunk_size, self._BASE)
        n = int(chunk_hashes.shape[0])
        if n == 0:
            return

        with self._lock:
            # Filter chunks already registered to avoid duplicate compact-ID
            # allocation.  When both cb_store_pre_computed and cb_store_final
            # fire for the same token sequence they produce identical hashes;
            # registering twice orphans the first compact ID permanently since
            # _token_hash_to_compact_id is overwritten but the old list slot is
            # not freed.
            new_idxs = [
                i
                for i in range(n)
                if token_hashes[i] not in self._token_hash_to_compact_id
            ]
            if not new_idxs:
                return
            n_new = len(new_idxs)
            new_chunk_hashes = chunk_hashes[new_idxs]

            # Compact sequential IDs: bounded by _TABLE_SIZE, safe for seen-array sizing
            # NOTE: base_id grows monotonically (evicted slots are not reused); the hard
            # limit is on total chunks ever registered, not active chunks.
            base_id = len(self._chunk_token_hash)
            if base_id + n_new > self._TABLE_SIZE:
                logger.error(
                    "BlendTokenRangeMatcher compact-ID overflow: %d chunks "
                    "registered, cannot add %d more (limit %d). Skipping.",
                    base_id,
                    n_new,
                    self._TABLE_SIZE,
                )
                return
            if base_id + n_new > int(self._TABLE_SIZE * 0.8):
                logger.warning(
                    "BlendTokenRangeMatcher nearing capacity: %d/%d compact IDs used. "
                    "Hash collision rate is rising; hit rate will degrade.",
                    base_id + n_new,
                    self._TABLE_SIZE,
                )
            compact_ids = np.arange(base_id, base_id + n_new, dtype=np.int64)

            # Write table: poly_chunk_hash -> compact_chunk_id
            update_table_id_numba(new_chunk_hashes, self._table_id, compact_ids)

            # Persist compact_id -> token_hash, token_hash -> start, and reverse maps
            for k, orig_i in enumerate(new_idxs):
                th = token_hashes[orig_i]
                cid = int(compact_ids[k])
                slot = int(new_chunk_hashes[k]) & int(self._mask)
                self._chunk_token_hash.append(th)
                self._token_hash_to_start[th] = orig_i * self.chunk_size
                self._compact_id_to_slot[cid] = slot
                self._token_hash_to_compact_id[th] = cid

    def match_sub_sequence(
        self,
        token_ids: list[int],
    ) -> list[CBMatchResult]:
        """Find stored chunks whose fingerprints appear anywhere in token_ids.

        Uses a sliding-window rolling hash so matches need not be aligned to
        chunk_size boundaries in the query.  Entries previously evicted via
        remove_chunks (token_hash set to None) are silently skipped.

        Args:
            token_ids: Query token sequence to probe (raw token IDs as uint64).

        Returns:
            One CBMatchResult per unique stored chunk that was hit.
              old_st/old_ed : positions in the originally registered sequence
              cur_st/cur_ed : positions in the query (token_ids) where
                              the match was found
              hash          : token_hash bytes (from registration) for cache key lookup
        """
        if len(token_ids) < self.chunk_size:
            return []

        arr = np.array(token_ids, dtype=np.uint64)
        # Sliding-window polynomial hashes over the query
        rolling = rolling_hash_windows_numba(arr, self.chunk_size, self._BASE)

        with self._lock:
            if not self._chunk_token_hash:
                return []

            # Probe table; seen array is _TABLE_SIZE bytes (~1 MB), fixed and safe
            hit_ids = unique_hits_direct_id_numba(
                rolling, self._table_id, self._mask, self._TABLE_SIZE
            )

            if hit_ids.shape[0] == 0:
                return []

            # For each hit compact_id, find the first query position where it matched
            hit_id_set = set(int(cid) for cid in hit_ids)
            cid_to_query_pos: dict[int, int] = {}
            for q_pos in range(rolling.shape[0]):
                idx = int(rolling[q_pos]) & int(self._mask)
                cid = int(self._table_id[idx])
                if cid in hit_id_set and cid not in cid_to_query_pos:
                    cid_to_query_pos[cid] = q_pos
                    if len(cid_to_query_pos) == len(hit_id_set):
                        break

            results: list[CBMatchResult] = []
            for cid in hit_ids:
                cid_int = int(cid)
                th = self._chunk_token_hash[cid_int]
                if th is None:
                    continue
                old_st = self._token_hash_to_start.get(th)
                cur_st = cid_to_query_pos.get(cid_int)
                if old_st is None or cur_st is None:
                    continue
                results.append(
                    CBMatchResult(
                        old_st=old_st,
                        old_ed=old_st + self.chunk_size,
                        cur_st=cur_st,
                        cur_ed=cur_st + self.chunk_size,
                        hash=th,
                    )
                )
            return results

    def remove_chunks(self, token_hashes: list[bytes]) -> None:
        """Evict stale entries whose backing data is no longer in storage.

        Args:
            token_hashes: Token hashes of chunks to remove from the table.
        """
        with self._lock:
            for th in token_hashes:
                cid = self._token_hash_to_compact_id.get(th)
                if cid is None:
                    continue
                # Clear the table slot
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
                # Clean up auxiliary maps
                self._chunk_token_hash[cid] = None
                self._token_hash_to_start.pop(th, None)
                del self._token_hash_to_compact_id[th]

    def has_chunk(self, token_hash: bytes) -> bool:
        """Return True if token_hash is currently registered in the matcher.

        Used before lazy registration to avoid creating duplicate compact-ID
        entries for a hash that is already in the fingerprint table.

        Args:
            token_hash: The storage hash bytes for a single chunk (as returned
                        by TokenHasher.compute_chunk_hashes).

        Returns:
            True if the chunk is registered and not evicted, False otherwise.
        """
        with self._lock:
            return token_hash in self._token_hash_to_compact_id


def _unique_token_coverage(results: list[CBMatchResult]) -> int:
    """Return the number of unique query tokens covered by a set of CBMatchResults.

    match_sub_sequence is a sliding-window probe, so two results from different
    registered chunks can have overlapping [cur_st, cur_ed) ranges.  Summing
    chunk_size per result would double-count the overlapping tokens and produce
    hit_rate > 1.  This function merges the intervals first.

    Args:
        results: Found CBMatchResult objects (each covers [cur_st, cur_ed) tokens).

    Returns:
        Total number of unique query-token positions covered.
    """
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


class BlendModule:
    """Handles blend (context-blend / cross-request KV reuse) operations.

    Owns CB-specific GPU context registrations and the token range matcher.
    Provides handlers for CB register, unregister, store, retrieve, and lookup.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._cb_gpu_contexts: dict[int, PlainGPUCacheContext] = {}
        self._cb_gpu_context_meta: dict[int, tuple[str, int]] = {}
        self._token_range_matcher = BlendTokenRangeMatcher(ctx.chunk_size)
        self._gpu_copy_lock = threading.Lock()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.CB_REGISTER_KV_CACHE,
                self.cb_register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNREGISTER_KV_CACHE,
                self.cb_unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_STORE_PRE_COMPUTED,
                self.cb_store_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.CB_RETRIEVE_PRE_COMPUTED_V2,
                self.cb_retrieve_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.CB_STORE_FINAL,
                self.cb_store_final,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.CB_LOOKUP_PRE_COMPUTED_V2,
                self.cb_lookup_pre_computed,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict:
        """Return blend module status information.

        Returns:
            A dict containing registered CB GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        cb_gpu_context_meta: dict[str, dict] = {}
        for gpu_id, meta in self._cb_gpu_context_meta.items():
            model_name, world_size = meta
            entry: dict = {
                "model_name": model_name,
                "world_size": world_size,
            }
            ctx = self._cb_gpu_contexts.get(gpu_id)
            if ctx is not None:
                # bytes per token = 2 (K+V) * num_layers * hidden_dim_size *
                # itemsize; num_tokens is the cache capacity, not a per-token
                # cost.
                cache_size_per_token = (
                    2 * ctx.num_layers * ctx.hidden_dim_size * ctx.dtype.itemsize
                )
                entry["kv_cache_layout"] = {
                    "num_layers": ctx.num_layers,
                    "num_tokens": ctx.num_tokens,
                    "hidden_dim_size": ctx.hidden_dim_size,
                    "dtype": str(ctx.dtype),
                    "cache_size_per_token": cache_size_per_token,
                }
            cb_gpu_context_meta[str(gpu_id)] = entry

        return {
            "registered_cb_gpu_ids": list(self._cb_gpu_contexts.keys()),
            "cb_gpu_context_meta": cb_gpu_context_meta,
        }

    def close(self) -> None:
        """Release resources owned by this module."""
        self._cb_gpu_contexts.clear()
        self._cb_gpu_context_meta.clear()

    def cb_register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
    ) -> None:
        """Register the KV cache buffer from the blend engine.

        Args:
            instance_id: Unique identifier for the blend engine instance.
            kv_caches: KVCache object containing the GPU buffer pointers.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
        """
        gpu_context = PlainGPUCacheContext(kv_caches, self._ctx.chunk_size)
        self._cb_gpu_contexts[instance_id] = gpu_context
        self._cb_gpu_context_meta[instance_id] = (model_name, world_size)

        layout_desc = MemoryLayoutDesc(
            shapes=[gpu_context.get_kv_buffer_shape(self._ctx.chunk_size)],
            dtypes=[gpu_context.dtype],
        )
        self._ctx.layout_desc_registry.register(model_name, world_size, layout_desc)

        logger.info(
            "Registered CB KV cache for instance_id %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def cb_unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache buffer for the given instance_id.

        Args:
            instance_id: Unique identifier for the blend engine instance
                to unregister.
        """
        if instance_id in self._cb_gpu_contexts:
            model_name, world_size = self._cb_gpu_context_meta[instance_id]
            del self._cb_gpu_contexts[instance_id]
            del self._cb_gpu_context_meta[instance_id]
            self._ctx.layout_desc_registry.unregister(model_name, world_size)
            logger.info("Unregistered CB KV cache for instance_id %d", instance_id)
        else:
            logger.warning(
                "Attempted to unregister non-existent CB KV cache for instance_id %d",
                instance_id,
            )

    def cb_lookup_pre_computed(self, key: IPCCacheServerKey) -> list[CBMatchResult]:
        """Lookup the pre-computed chunks in the underlying storage.

        Uses BlendTokenRangeMatcher for a fast local pre-filter, then submits
        prefetch tasks for matched chunks using their stored hashes directly.
        Chunks that the fingerprint table matched but are no longer present in
        storage are lazily evicted from the matcher via remove_chunks.

        Args:
            key: IPCCacheServerKey containing the token ids to lookup.

        Returns:
            List of CBMatchResult for chunks that were actually found in storage,
            ready to be passed to cb_retrieve_pre_computed.
        """
        num_tokens = len(key.token_ids)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_START,
                session_id=key.request_id,
                metadata={"num_tokens": num_tokens},
            )
        )

        cb_match_result = self._token_range_matcher.match_sub_sequence(
            list(key.token_ids)
        )
        if not cb_match_result:
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=key.request_id,
                    metadata={
                        "num_tokens": num_tokens,
                        "fingerprint_hits": 0,
                        "prefix_hits": 0,
                        "storage_hits": 0,
                        "stale_chunks": 0,
                        "no_gpu_context": False,
                        "hit_tokens": 0,
                        "requested_tokens": (num_tokens // self._ctx.chunk_size)
                        * self._ctx.chunk_size,
                    },
                )
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.CB_REQUEST_END,
                    session_id=key.request_id,
                )
            )
            return []

        # Sort by query position
        cb_match_result.sort(key=lambda r: r.cur_st)

        # The sliding-window probe returns O(table_size) overlapping matches.
        # Greedy leftmost-first picks one chunk per slot; lossless when matches
        # are chunk-aligned (the CB case).
        deduped: list[CBMatchResult] = []
        covered_end = -1
        for r in cb_match_result:
            if r.cur_st >= covered_end:
                deduped.append(r)
                covered_end = r.cur_ed
        cb_match_result = deduped

        # Group consecutive matched chunks
        groups: list[list[CBMatchResult]] = []
        for result in cb_match_result:
            if groups and groups[-1][-1].cur_ed == result.cur_st:
                groups[-1].append(result)
            else:
                groups.append([result])

        prefetch_handles: list[PrefetchHandle] = []
        found_cb_match_result: list[CBMatchResult] = []
        model_name, world_size = key.model_name, key.world_size

        # Find the cb gpu context and calculate the layout desc
        layout_desc: MemoryLayoutDesc | None = None
        for gpu_id, (m_name, w_size) in self._cb_gpu_context_meta.items():
            if m_name == model_name and w_size == world_size:
                cb_ctx = self._cb_gpu_contexts[gpu_id]
                layout_desc = MemoryLayoutDesc(
                    shapes=[cb_ctx.get_kv_buffer_shape(self._ctx.chunk_size)],
                    dtypes=[cb_ctx.dtype],
                )
                break

        if layout_desc is None:
            logger.error(
                "No CB GPU context found for model %s with world size %d "
                "during cb_lookup_pre_computed!",
                model_name,
                world_size,
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.CB_LOOKUP_END,
                    session_id=key.request_id,
                    metadata={
                        "num_tokens": num_tokens,
                        "fingerprint_hits": 0,
                        "prefix_hits": 0,
                        "storage_hits": 0,
                        "stale_chunks": 0,
                        "no_gpu_context": True,
                        "hit_tokens": 0,
                        "requested_tokens": (num_tokens // self._ctx.chunk_size)
                        * self._ctx.chunk_size,
                    },
                )
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.CB_REQUEST_END,
                    session_id=key.request_id,
                )
            )
            return []

        # Submit prefetch for each group.  All candidates use the standard chunk
        # hash computed by token_hasher, which matches the hash used at store
        # time, so ipc_key_to_object_keys resolves correctly.
        for group in groups:
            chunk_hashes = [r.hash for r in group]
            obj_keys = ipc_key_to_object_keys(key, chunk_hashes, [0])[0]
            handle = self._ctx.storage_manager.submit_prefetch_task(
                PrefetchRequestSpec(keys=obj_keys, layout_desc=layout_desc),
                external_request_id=key.request_id,
            )
            prefetch_handles.append(handle)

            logger.debug(
                "Submitted prefetch for %d chunks starting at %d",
                len(group),
                group[0].cur_st,
            )

        # Collect only the CBMatchResults for chunks actually found in storage
        stale_hashes: list[bytes] = []
        for handle, group in zip(prefetch_handles, groups, strict=False):
            found = None
            while True:
                found = self._ctx.storage_manager.query_prefetch_status(handle)
                if found is not None:
                    break
                time.sleep(0.001)

            # Real found count after dedup the TP
            found_count = found.count_leading_ones() // world_size

            start = group[0].cur_st
            end = group[-1].cur_ed
            if found_count > 0:
                found_cb_match_result.extend(group[:found_count])
                # Chunks after found_count in the group are stale
                stale_hashes.extend(r.hash for r in group[found_count:])
                logger.debug(
                    "Found %d pre-computed chunks for range (%d, %d)",
                    found_count,
                    start,
                    end,
                )
            else:
                stale_hashes.extend(r.hash for r in group)
                logger.debug(
                    "No pre-computed chunks found for range (%d, %d)",
                    start,
                    end,
                )

        # Evict stale fingerprint entries; remove_chunks safely skips hashes that
        # were never registered (e.g. prefix-probe candidates not in storage).
        if stale_hashes:
            self._token_range_matcher.remove_chunks(stale_hashes)
            logger.debug(
                "Evicted %d stale chunks from fingerprint table",
                len(stale_hashes),
            )
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.CB_CHUNKS_EVICTED,
                    metadata={"num_chunks": len(stale_hashes)},
                )
            )

        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_LOOKUP_END,
                session_id=key.request_id,
                metadata={
                    "num_tokens": num_tokens,
                    "fingerprint_hits": len(found_cb_match_result),
                    "prefix_hits": 0,
                    "storage_hits": len(found_cb_match_result),
                    "stale_chunks": len(stale_hashes),
                    "no_gpu_context": False,
                    "hit_tokens": _unique_token_coverage(found_cb_match_result),
                    "requested_tokens": (num_tokens // self._ctx.chunk_size)
                    * self._ctx.chunk_size,
                },
            )
        )
        return found_cb_match_result

    def _cb_store_gpu_copy(
        self,
        obj_keys: list[ObjectKey],
        gpu_context: PlainGPUCacheContext,
        offset: int,
        event_ipc_handle: bytes,
        start_event: Event | None = None,
    ) -> tuple[Any, dict]:
        """Helper function to perform GPU-to-CPU copy operations for storing chunks.

        Args:
            obj_keys: List of object keys to store.
            gpu_context: GPU context for the blend engine instance.
            offset: The starting offset in the CB KV cache buffer.
            event_ipc_handle: The IPC handle for the GPU event that signals the
                completion of LLM inference.
            start_event: Optional event to publish on the stream after waiting for
                the vLLM GPU event, marking the true start of the store operation.

        Returns:
            A tuple of (event, reserved_dict) where event is the GPU event and
            reserved_dict is the dictionary of reserved memory objects.
        """
        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            # Not all backends support interprocess Events (CUDA IPC specific)
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            # Wait for vLLM event to finish
            # Not all backends support IPC event handles (CUDA IPC specific)
            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC event "
                    "handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            if start_event is not None:
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream, start_event
                )

            # Prepare for the copy
            num_tokens = self._ctx.chunk_size
            cpu_shape = gpu_context.get_kv_buffer_shape(num_tokens)
            layout_desc = MemoryLayoutDesc(
                shapes=[cpu_shape], dtypes=[gpu_context.dtype]
            )

            reserved_dict = self._ctx.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )

            for idx, obj_key in enumerate(obj_keys):
                if obj_key in reserved_dict:
                    memory_obj = reserved_dict[obj_key]
                else:
                    continue

                offset_start = idx * self._ctx.chunk_size + offset
                offset_end = offset_start + self._ctx.chunk_size

                # Copy from GPU to CPU
                tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
                gpu_kv_slice = gpu_context.slice_kv_cache_on_tokens(
                    offset_start, offset_end
                )
                with self._gpu_copy_lock:
                    tmp_buffer.copy_(gpu_kv_slice, non_blocking=True)
                    lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

            event.record()
        # Call finish_write after the copy is done
        gpu_context.cupy_stream.launch_host_func(
            self._ctx.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )

        return event, reserved_dict

    def cb_store_pre_computed(
        self,
        key: IPCCacheServerKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the pre-computed chunks in the underlying storage for later retrieval.

        Args:
            key: IPCCacheServerKey containing the token ids for which the
                pre-computed chunks are stored.
            offset: The starting offset in the CB KV cache buffer where the
                pre-computed chunks begin.
            instance_id: The instance_id of the blend engine instance to store
                the pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals the
                completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of storing
            the pre-computed chunks, and a boolean flag indicating if the store
            is successful.

        Raises:
            ValueError: If instance_id is not registered for CB KV cache.

        Note:
            The input tokens should not have any separator in it. It should just
            be one "paragraph".
            This function will discard the last partial chunk and only store the
            full chunks.
        """
        num_tokens = key.end - key.start

        if instance_id not in self._cb_gpu_contexts:
            raise ValueError(
                f"Instance ID {instance_id} not registered for CB KV cache"
            )
        gpu_context = self._cb_gpu_contexts[instance_id]

        # CPU-synchronous sentinel: GPU store is about to be enqueued.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_SUBMITTED,
                session_id=key.request_id,
                metadata={"instance_id": instance_id},
            )
        )
        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_START,
                session_id=key.request_id,
                metadata={"instance_id": instance_id, "num_tokens": num_tokens},
            ),
        )

        # Compute normal prefix hashes so these chunks are accessible both via
        # the CB lookup path and via the standard lookup/retrieve path.
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        # convert to object key
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes, [0])[0]

        reserved_dict: dict = {}
        try:
            event, reserved_dict = self._cb_store_gpu_copy(
                obj_keys, gpu_context, offset, event_ipc_handle
            )

            # Register chunk hashes with the local matcher for fast sub-sequence lookup
            token_hashes = list(chunk_hashes)

            # NOTE(Jiayi): We only register the token hashes for worker_id 0 or None
            # to avoid duplicate registration across workers.
            if key.worker_id in [0, None]:
                self._token_range_matcher.on_new_token_hashes(
                    list(key.token_ids), token_hashes
                )
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                        session_id=key.request_id,
                        metadata={
                            "num_chunks": len(token_hashes),
                            "num_tokens": len(list(key.token_ids)),
                        },
                    )
                )

            logger.info(
                "Stored pre-computed doc with %d tokens, num stored chunks: %d",
                key.end - key.start,
                len(reserved_dict),
            )
        except Exception:
            logger.exception("Cannot store pre-computed chunks due to exception")
            self._ctx.event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                    session_id=key.request_id,
                    metadata={
                        "instance_id": instance_id,
                        "num_tokens": num_tokens,
                        "stored_chunks": 0,
                        "success": False,
                    },
                ),
            )
            raise

        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_STORE_PRE_COMPUTED_END,
                session_id=key.request_id,
                metadata={
                    "instance_id": instance_id,
                    "num_tokens": num_tokens,
                    "stored_chunks": len(reserved_dict),
                    "success": True,
                },
            ),
        )
        return event.ipc_handle(), True

    def cb_retrieve_pre_computed(
        self,
        key: IPCCacheServerKey,
        cb_match_result: list[CBMatchResult],
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Retrieve pre-computed chunks from storage and copy them to the CB KV buffer.

        Args:
            key: IPCCacheServerKey containing the token ids for which the
                pre-computed chunks are retrieved.
            cb_match_result: List of CBMatchResult returned by
                cb_lookup_pre_computed, containing the per-chunk hashes and
                query positions.
            offset: The starting offset in the CB KV cache buffer to copy the
                retrieved chunks to.
            instance_id: The instance_id of the blend engine instance to
                retrieve the pre-computed chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals
                the completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of
            retrieving the pre-computed chunks, and a boolean flag indicating
            if the retrieval is successful.

        Raises:
            ValueError: If instance_id is not registered for CB KV cache.

        Note:
            cb_lookup_pre_computed must be called first before calling this
            function.
        """
        if instance_id not in self._cb_gpu_contexts:
            raise ValueError(
                f"Instance ID {instance_id} not registered for CB KV cache"
            )
        gpu_context = self._cb_gpu_contexts[instance_id]

        # One obj_key per match_result, in cur_st order
        cb_match_result = sorted(cb_match_result, key=lambda r: r.cur_st)
        num_chunks = len(cb_match_result)
        chunk_hashes = [r.hash for r in cb_match_result]
        all_obj_keys = ipc_key_to_object_keys(key, chunk_hashes, [0])[0]

        # CPU-synchronous sentinel: GPU retrieve is about to be enqueued.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"instance_id": instance_id},
            )
        )

        logger.debug("DEBUG object keys to retrieve: %s", all_obj_keys)

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            # Not all backends support interprocess Events (CUDA IPC specific)
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            self._ctx.event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_RETRIEVE_START,
                    session_id=key.request_id,
                    metadata={
                        "instance_id": instance_id,
                        "num_chunks": num_chunks,
                    },
                ),
            )

            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    all_obj_keys
                ) as memory_objs:
                    if memory_objs is None:
                        logger.error("Some keys not found during CB retrieve!")
                        self._ctx.event_bus.publish_on_stream(
                            gpu_context.cupy_stream,
                            Event(
                                event_type=EventType.CB_RETRIEVE_END,
                                session_id=key.request_id,
                                metadata={
                                    "instance_id": instance_id,
                                    "num_chunks": num_chunks,
                                    "success": False,
                                },
                            ),
                        )
                        return event.ipc_handle(), False

                    for r, memory_obj in zip(
                        cb_match_result, memory_objs, strict=False
                    ):
                        gpu_st = r.cur_st + offset
                        gpu_ed = gpu_st + self._ctx.chunk_size
                        tmp_buffer = gpu_context.get_tmp_gpu_buffer(
                            self._ctx.chunk_size
                        )
                        target_buffer = gpu_context.slice_kv_cache_on_tokens(
                            gpu_st, gpu_ed
                        )
                        with self._gpu_copy_lock:
                            lmcache_memcpy_async_h2d(memory_obj, tmp_buffer)
                            target_buffer.copy_(tmp_buffer, non_blocking=True)

            except Exception:
                logger.exception("Error during retrieving prefetched results")
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.CB_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "instance_id": instance_id,
                            "num_chunks": num_chunks,
                            "success": False,
                        },
                    ),
                )
                return event.ipc_handle(), False

            finally:
                event.record()
                # TODO: here we simply "unlock" all the keys, which may cause
                # double-unlock if error happens during read_prefetched_results.
                # We should consider not unlocking objects in read_prefetched_results
                # if error happens.
                gpu_context.cupy_stream.launch_host_func(
                    self._ctx.storage_manager.finish_read_prefetched,
                    all_obj_keys,
                )

        logger.info(
            "Retrieved pre-computed for %d match results to GPU offset starting at %d",
            len(cb_match_result),
            offset,
        )
        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=key.request_id,
                metadata={
                    "instance_id": instance_id,
                    "num_chunks": num_chunks,
                    "success": True,
                },
            ),
        )
        return event.ipc_handle(), True

    def cb_store_final(
        self,
        key: IPCCacheServerKey,
        offset: int,
        instance_id: int,
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the final chunks in the underlying storage after processing.

        The stored chunks should be accessible for normal mode LLMs.

        Args:
            key: IPCCacheServerKey containing the token ids for which the
                final chunks are stored.
            offset: The starting offset in the CB KV cache buffer where the
                final chunks are stored.
            instance_id: The instance_id of the blend engine instance to
                store the final chunks for.
            event_ipc_handle: The IPC handle for the CUDA event that signals
                the completion of LLM inference.

        Returns:
            IPC handle bytes for the event that signals the completion of
            storing the final chunks, and a boolean flag indicating if the
            store is successful.

        Raises:
            ValueError: If instance_id is not registered for CB KV cache.
        """
        num_tokens = key.end - key.start

        # Get GPU context
        if instance_id not in self._cb_gpu_contexts:
            raise ValueError(
                f"Instance ID {instance_id} not registered for CB KV cache"
            )
        gpu_context = self._cb_gpu_contexts[instance_id]

        # CPU-synchronous sentinels: SUBMITTED before SESSION_END so the
        # tracing subscriber's in-flight counter is non-zero when SESSION_END
        # arrives and correctly defers root span closure.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_STORE_FINAL_SUBMITTED,
                session_id=key.request_id,
                metadata={"instance_id": instance_id},
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.CB_REQUEST_END,
                session_id=key.request_id,
            )
        )

        # Compute normal hash for the keys
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))

        # convert to object key
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes, [0])[0]

        reserved_dict: dict = {}
        try:
            event, reserved_dict = self._cb_store_gpu_copy(
                obj_keys,
                gpu_context,
                offset,
                event_ipc_handle,
                start_event=Event(
                    event_type=EventType.CB_STORE_FINAL_START,
                    session_id=key.request_id,
                    metadata={
                        "instance_id": instance_id,
                        "num_tokens": num_tokens,
                    },
                ),
            )

            # Register fingerprints so future CB lookups can find these chunks.
            # Mirrors cb_store_pre_computed; without this, chunks stored here are
            # invisible to cb_lookup_pre_computed, causing 0% hit rate on re-requests.
            if key.worker_id in [0, None]:
                self._token_range_matcher.on_new_token_hashes(
                    list(key.token_ids), list(chunk_hashes)
                )
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.CB_FINGERPRINTS_REGISTERED,
                        session_id=key.request_id,
                        metadata={
                            "num_chunks": len(chunk_hashes),
                            "num_tokens": len(list(key.token_ids)),
                        },
                    )
                )

            logger.info(
                "Stored final doc with %d tokens, num stored chunks: %d",
                key.end - key.start,
                len(reserved_dict),
            )
        except Exception:
            logger.exception("Cannot store final chunks due to exception")
            self._ctx.event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.CB_STORE_FINAL_END,
                    session_id=key.request_id,
                    metadata={
                        "instance_id": instance_id,
                        "num_tokens": num_tokens,
                        "stored_chunks": 0,
                        "success": False,
                    },
                ),
            )
            raise

        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.CB_STORE_FINAL_END,
                session_id=key.request_id,
                metadata={
                    "instance_id": instance_id,
                    "num_tokens": num_tokens,
                    "stored_chunks": len(reserved_dict),
                    "success": True,
                },
            ),
        )
        return event.ipc_handle(), True
