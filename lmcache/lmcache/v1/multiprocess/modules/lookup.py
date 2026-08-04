# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass
from functools import partial
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


def compute_extra_count(
    tp_size: int,
    world_size: int,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    TODO: world_size currently carries an overloaded
    meaning (total ranks for non-MLA vs total/tp for
    MLA). Consider a dedicated field in the future.

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


def _get_prefix_hit_length(
    found_prefix_len: int,
    world_size: int,
    num_object_groups: int,
) -> int:
    """Return the prefix hit length in chunks.

    The chunk-major key layout packs ``world_size * num_object_groups`` keys per
    chunk, so a fully-present prefix of ``found_prefix_len`` keys hits
    ``found_prefix_len // (world_size * num_object_groups)`` chunks.

    Args:
        found_prefix_len: Length, in keys, of the contiguous found-key prefix.
        world_size: Number of kv_rank shards per chunk.
        num_object_groups: Number of object groups in the chunk-major layout.

    Returns:
        The prefix hit length in chunks (fully-present chunks of the prefix).

    Example (2 object groups ``g0,g1``; 2 kv_ranks ``r0,r1`` -> 4 keys per
    chunk). A found-key prefix of 6 keys covers one full chunk plus half the
    next::

        [c0g0r0, c0g0r1, c0g1r0, c0g1r1,   # chunk 0: fully present
         c1g0r0, c1g0r1]                    # chunk 1: 2 of 4 -> partial, dropped

    so ``found_prefix_len=6`` hits ``6 // (2 * 2) = 1`` whole chunk.

    Note:
        Correct only under full attention -- every object group present for
        every hit chunk. Once sliding-window prefetch lands, the hit is no
        longer a uniform contiguous key prefix and the hit length must come
        from the fold's reported hit length instead.
    """
    return found_prefix_len // (world_size * num_object_groups)


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 for early-exit paths (no GPU context matches
    # or chunk_hashes is empty).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    num_object_groups: int = 1
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""


class LookupModule:
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS,
    QUERY_PREFETCH_LOOKUP_HITS, FREE_LOOKUP_LOCKS, and END_SESSION
    request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            List of handler specs for lookup-related request types.
        """
        return [
            HandlerSpec(RequestType.LOOKUP, self.lookup, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_STATUS,
                self.query_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.WAIT_PREFETCH_STATUS,
                self.wait_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                self.query_prefetch_lookup_hits,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.FREE_LOOKUP_LOCKS,
                self.free_lookup_locks,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.END_SESSION,
                self.end_session,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        return {
            "active_prefetch_jobs": self._active_prefetch_count(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""
        pass

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    def lookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        model_name, world_size = key.model_name, key.world_size
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._ctx.layout_desc_registry.find(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_found_indices=(),
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        extra_count = compute_extra_count(tp_size, world_size)

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_found_indices=(),
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._ctx.event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        # Lay keys out chunk-major across object groups (see
        # _chunk_major_object_keys); pass the windows to the prefetch policy.
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)

        handle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=obj_keys,
                layout_desc=layout_desc,
                extra_count=extra_count,
                attn_desc=attn_desc,
            ),
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=key.world_size,
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                num_object_groups=attn_desc.num_object_groups,
                model_name=model_name,
                cache_salt=key.cache_salt,
            )
        )

    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found = self._ctx.storage_manager.query_prefetch_lookup_hits(job.handle)
        if found is None:
            return None

        return _get_prefix_hit_length(found, job.world_size, job.num_object_groups)

    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if found is None:
            return None

        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the
        #    first failure
        found_count = _get_prefix_hit_length(
            found.count_leading_ones(), job.world_size, job.num_object_groups
        )

        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    "hit_tokens": found_count * self._ctx.chunk_size,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)

        return found_count

    def wait_prefetch_status(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Block until a prefetch job completes, then return its chunk count.

        Like query_prefetch_status, but waits for the daemon to publish the
        result instead of returning None while the prefetch is still in
        progress, so the caller does not have to busy-poll. The job entry is
        removed once a non-None result is returned (exactly-once semantics).

        Args:
            request_id: The external request ID passed in the lookup key.
            timeout: Maximum number of seconds to wait for the prefetch.

        Returns:
            Chunk count (int) when done, None if the wait timed out, 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if not self._ctx.storage_manager.wait_prefetch_status(job.handle, timeout):
            return None
        return self.query_prefetch_status(request_id)

    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        # Release across every object group, mirroring lookup, which locks keys
        # in every group; releasing only group 0 would leak the rest.
        #
        # NOTE: correct only for full attention, where every locked chunk is a
        # hit chunk. Sliding-window groups do not retain chunks outside their
        # window, so once SWA prefetch lands this must skip those chunks instead
        # of releasing every one -- otherwise chunks the engine still holds can
        # be over-released (e.g. window=512, LMCache hit 1024, vLLM hit 768 ->
        # chunks 512..768 may leak). Revisit when sliding-window prefetch is on.
        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(tp_size, key.world_size)

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys, extra_count=extra_count
        )

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self._ctx.session_manager.remove(request_id)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch",
                request_id,
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = self._chunk_major_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _chunk_major_object_keys(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> list[ObjectKey]:
        """Resolve the flat object-key list across all object groups,
        chunk-major.

        The object-group count is read from the layout registry for
        ``key``'s ``(model_name, world_size)``. The keys are ordered
        ``chunk -> object group -> kv_rank`` so that all keys belonging to one
        chunk are contiguous; a leading-ones prefix over the flat list then maps
        directly to a whole-chunk hit count. Callers that need the full key set
        regardless of order (lock release, touch) use this too.

        Example (2 chunks ``c0,c1``; 2 groups ``g0,g1``; 2 kv_ranks ``r0,r1``)::

            [c0g0r0, c0g0r1, c0g1r0, c0g1r1,   # chunk 0: all groups, all ranks
             c1g0r0, c1g0r1, c1g1r0, c1g1r1]   # chunk 1: ...

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.

        Returns:
            The chunk-major flattened list of object keys across all groups.
        """
        num_groups = self._ctx.layout_desc_registry.find_attn_desc(
            key.model_name, key.world_size
        ).num_object_groups
        per_group = ipc_key_to_object_keys(key, chunk_hashes, list(range(num_groups)))
        if num_groups == 1:
            return per_group[0]
        # Each per-group list is chunk-major / rank-minor of length
        # len(chunk_hashes) * num_ranks; recover num_ranks to slice per chunk.
        num_ranks = len(per_group[0]) // len(chunk_hashes) if chunk_hashes else 0
        obj_keys: list[ObjectKey] = []
        for chunk_idx in range(len(chunk_hashes)):
            lo = chunk_idx * num_ranks
            hi = lo + num_ranks
            for group_keys in per_group:
                obj_keys.extend(group_keys[lo:hi])
        return obj_keys

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
