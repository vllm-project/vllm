# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
import enum
import os
import threading
from dataclasses import dataclass
from typing import Any

# Third Party
import torch
import zmq

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.periodic_thread import PeriodicThread, ThreadLevel, ThreadRunSummary

logger = init_logger(__name__)


class ExtraConfigDefault(enum.Enum):
    """Centralized default values for extra_config keys.

    Each member's *name* is the key used in the extra_config dict,
    and its *value* is the default.
    """

    # Timeout (seconds) for blocking MQ requests: initial
    # chunk-size query, KV cache registration/unregistration,
    # and other synchronous operations.
    mq_timeout = 300.0
    # Interval (seconds) between periodic heartbeat pings
    # to the server.
    heartbeat_interval = 10.0


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
    timeout: float = ExtraConfigDefault.mq_timeout.value,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        timeout: Timeout in seconds for the blocking request.

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result(timeout=timeout)
    return chunk_size


def send_ping(
    mq_client: MessageQueueClient,
    timeout: float,
) -> bool:
    """Send a PING request and return the result.

    Returns:
        True if server is healthy, False on timeout or error.
    """
    try:
        future = send_lmcache_request(mq_client, RequestType.PING, [])
        return future.result(timeout=timeout)
    except TimeoutError:
        return False
    except Exception:
        logger.debug("Ping failed with exception", exc_info=True)
        return False


@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    kv_world_size: int
    """
    The kv world size, kv_world_size may not be equal to the actual_world_size, 
    in the case of mla, it will 'exclude' the effect of TP, the value is 
    calculated by `extract_world_size_and_kv_rank` in `lmcache_mp_connector.py`.
    """

    kv_worker_id: int
    """
    The kv worker id of the sub-process, kv_worker_id may not be equal to the 
    actual_worker_id, in the case of mla, it will 'exclude' the effect of TP, 
    the value is calculated by `extract_world_size_and_kv_rank` in 
    `lmcache_mp_connector.py`.
    """

    actual_world_size: int
    """The actual world size."""

    actual_worker_id: int
    """The actual worker id of the sub-process."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""


class HeartbeatThread(PeriodicThread):
    """Periodically checks server health via PING.

    Manages a threading.Event that adapters use to gate operations.
    When unhealthy, the adapter enters degraded mode; if the server
    recovers, the adapter automatically resumes normal operation.
    """

    def __init__(
        self,
        mq_client: MessageQueueClient,
        health_event: threading.Event,
        interval: float = ExtraConfigDefault.heartbeat_interval.value,
    ):
        """
        Args:
            mq_client: The message queue client used to send PING requests.
            health_event: A threading.Event shared with the adapter.
                Set when the server is healthy, cleared when unhealthy.
                Adapters check this event to decide whether to proceed
                with operations or enter degraded mode.
            interval: Seconds between heartbeat pings and ping timeout.
        """
        super().__init__(
            name="lmcache-heartbeat",
            interval=interval,
            level=ThreadLevel.CRITICAL,
        )
        self._mq_client = mq_client
        self._health_event = health_event
        self._interval = interval

    def _execute(self) -> ThreadRunSummary:
        was_healthy = self._health_event.is_set()
        healthy = send_ping(self._mq_client, timeout=self._interval)

        if healthy:
            self._health_event.set()
            if not was_healthy:
                logger.warning(
                    "LMCache server is healthy again — resuming normal operation"
                )
        else:
            self._health_event.clear()
            if was_healthy:
                logger.warning("LMCache server is unhealthy — entering degraded mode")

        return ThreadRunSummary(
            success=True,
            message="healthy" if healthy else "unhealthy",
        )


@dataclass
class LoadStoreOp:
    token_ids: list[int]
    """Token IDs for the load/store operation"""

    block_ids: list[int]
    """Block ids for the load/store operation"""

    start: int = 0
    """Start token index"""

    end: int = 0
    """End token index"""

    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = bool
LookupResult = int


_EXTRA_CONFIG_KEY_PREFIX = "lmcache.mp."


def _resolve_extra_config(
    extra_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve extra_config against :class:`ExtraConfigDefault`.

    Keys in *extra_config* are expected to carry the
    ``lmcache.mp.`` prefix (e.g. ``lmcache.mp.mq_timeout``).
    The prefix is stripped before matching against
    :class:`ExtraConfigDefault` members.

    All ``lmcache.mp.*`` entries are logged.  Entries whose
    value differs from the default are additionally marked as
    *overridden*.

    Args:
        extra_config: User-supplied config dict (may be *None*).

    Returns:
        A dict keyed by :pyattr:`ExtraConfigDefault` member names.
    """
    # Strip prefix to get a clean mapping
    stripped: dict[str, Any] = {}
    if extra_config is not None:
        for k, v in extra_config.items():
            if k.startswith(_EXTRA_CONFIG_KEY_PREFIX):
                short = k[len(_EXTRA_CONFIG_KEY_PREFIX) :]
                stripped[short] = v

    resolved: dict[str, Any] = {}
    for item in ExtraConfigDefault:
        default = item.value
        raw = stripped.get(item.name)
        value = type(default)(raw) if raw is not None else default
        if value != default:
            logger.info(
                "%s%s = %s (overridden, default: %s)",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
                default,
            )
        else:
            logger.info(
                "%s%s = %s",
                _EXTRA_CONFIG_KEY_PREFIX,
                item.name,
                value,
            )
        resolved[item.name] = value
    return resolved


# TODO(chunxiaozheng): To be compatible with older `lmcache_mp_connector`,
#  world_size, kv_rank, tp_size are saved, use parallel_strategy instead
class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int = 1,
        kv_rank: int = 0,
        vllm_block_size: int = 16,
        tp_size: int = 1,
        parallel_strategy: ParallelStrategy | None = None,
        extra_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            world_size: The world size used for LMCache keys
            kv_rank: The kv rank used for LMCache keys
            vllm_block_size: The block size used in vLLM
            tp_size: Tensor-parallel size for MLA
                multi-reader locking (default 1).
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `kv_world_size`, `kv_worker_id` and so on
            extra_config: Optional dict with keys starting with
                lmcache.mp. (e.g., lmcache.mp.mq_timeout).
        """
        cfg = _resolve_extra_config(extra_config)
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]

        # Lookup state tracking:
        # - _pending_lookups: request_ids submitted but not yet resolved
        # - _finished_lookup_results: cached chunk count keyed by request_id,
        #   so that repeated calls to check_lookup_result return the same value
        #   even after the server has already popped the job (exactly-once).
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self._world_size = world_size
        self._tp_size = tp_size

        # Read chunk size from lmcache
        try:
            self.chunk_size = get_lmcache_chunk_size(
                self.mq_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within "
                f"{cfg[ExtraConfigDefault.mq_timeout.name]}s. "
                f"Is the server running?"
            ) from None
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first lookup
        # request, by which time vLLM is fully ready.
        self._heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return (
            self._world_size
            if self.parallel_strategy is None
            else self.parallel_strategy.kv_world_size
        )

    @property
    def tp_size(self) -> int:
        """The tensor parallel size."""
        return (
            self._tp_size
            if self.parallel_strategy is None
            else self.parallel_strategy.tp_size
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.start()

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
    ):
        """
        Submit a new lookup request to LMCache if there is no ongoing request.

        Sends a LOOKUP request to the server and blocks until a prefetch
        job ID is returned.  The actual prefetch result can then be polled
        via ``check_lookup_result``.

        Args:
            request_id: The ID of the lookup request. The same ID indicates it's
                from the same request
            token_ids: Token IDs to lookup from LMCache

        Returns:
            None

        Notes:
            This function will have a side-effect: submitting a look up request to
            LMCache, which will essentially 'lock' the KV cache chunks in the LMCache
            for later retrieve operations.
            In the meantime, this function will record the lookup request, and the
            status of the look up request can be checked by `check_lookup_result`.
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
        ).no_worker_id_version()

        future = send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [key, self.tp_size],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LOOKUP request timed out after %ss. Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return
        self._pending_lookups.add(request_id)

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """
        Check the result of a previously submitted lookup request.

        Sends a QUERY_PREFETCH_STATUS request to the server and blocks
        until the server responds.  Returns the matched token count
        when the prefetch is complete, or None if still in progress.

        Args:
            request_id: The ID of the lookup request submitted in
                `maybe_submit_lookup_request`

        Returns:
            An integer representing the total number of tokens matched
            in LMCache (prefix matching), or
            None if the lookup request is not finished yet.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # If we have a cached result, return it to handle repeated calls.
            return self._finished_lookup_results.get(request_id, 0)

        if not self.is_healthy:
            # Server went down — give up on this lookup
            return 0

        if request_id in self._finished_lookup_results:
            # Return cached result if the job is already finished
            return self._finished_lookup_results[request_id]

        try:
            result = send_lmcache_request(
                self.mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss. "
                "Marking server as unhealthy.",
                self._mq_timeout,
            )
            self._health_event.clear()
            return 0

        if result is None:
            return None

        token_count = result * self.chunk_size
        self._finished_lookup_results[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """
        Clean up lookup state for a finished request to prevent memory leak.
        Args:
            request_id: The ID of the finished request.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookup_results.pop(request_id, None)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks that
        vLLM has already computed, so they will never be retrieved.  Calling
        this prevents those chunks from holding read locks until TTL expiry.

        Or use this when a request is cancelled or aborted after lookup but
        before retrieve to avoid holding read locks until TTL expiry.

        When ``start`` or ``end`` is not aligned to the chunk size, the
        entire chunk containing start boundary is freed but not end boundary.
        It is caller's responsibility to properly align the boundaries.

        Args:
            token_ids: Token IDs for the key (same as used in lookup).
            start: Start token index.
            end: End token index.
            request_id: The request ID.
        """
        if not self.is_healthy:
            return

        key = self._create_key(
            token_ids, start=start, end=end, request_id=request_id
        ).no_worker_id_version()
        send_lmcache_request(
            self.mq_client,
            RequestType.FREE_LOOKUP_LOCKS,
            [key, self.tp_size],
        )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        if not self.is_healthy:
            return

        send_lmcache_request(
            self.mq_client,
            RequestType.END_SESSION,
            [request_id],
        )

    def report_block_allocations(
        self,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Report vLLM GPU block allocation deltas to LMCache server.

        Fire-and-forget: does not wait for a response. If the server
        is unhealthy the report is silently dropped.

        Args:
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        if not self.is_healthy or not records:
            return

        send_lmcache_request(
            self.mq_client,
            RequestType.REPORT_BLOCK_ALLOCATION,
            [records],
        )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        # NOTE: for the scheduler adapter, we don't have a worker id,
        # so we set it to None in the key.
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        world_size: int = 1,
        kv_rank: int = 0,
        vllm_block_size: int = 16,
        parallel_strategy: ParallelStrategy | None = None,
        extra_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            server_url: The server URL for the LMCache
                message queue
            context: The ZMQ context
            model_name: The model name used for LMCache keys
            world_size: The world size used for LMCache keys
            kv_rank: The kv rank used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes use_mla,
                kv_world_size, kv_worker_id and so on
            extra_config: Optional dict with keys starting
                with lmcache.mp.
                (e.g., lmcache.mp.mq_timeout).
        """
        cfg = _resolve_extra_config(extra_config)
        self.mq_client = MessageQueueClient(server_url, context)
        self._mq_timeout = cfg[ExtraConfigDefault.mq_timeout.name]

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}
        # request_id -> (future, block_ids)
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}

        # Block IDs that failed due to retrieve timeout
        self.error_block_ids: set[int] = set()

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()
        # Request IDs already returned as finished_sending to the scheduler.
        # Prevents re-reporting the same ID after drain clears tracking sets.
        self._returned_finished: set[str] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self._world_size = world_size
        self._worker_id = kv_rank

        # Read chunk size from lmcache
        try:
            chunk_size = get_lmcache_chunk_size(
                self.mq_client, timeout=self._mq_timeout
            )
        except TimeoutError:
            self.mq_client.close()
            raise ConnectionError(
                f"LMCache server did not respond within "
                f"{cfg[ExtraConfigDefault.mq_timeout.name]}s. "
                f"Is the server running?"
            ) from None
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

        # Health state (shared with heartbeat thread)
        self._health_event = threading.Event()
        self._health_event.set()

        # Heartbeat thread is created but NOT started yet.
        # It will be lazily started on the first store or retrieve
        # request, by which time vLLM is fully ready (model loaded,
        # KV caches allocated, warmup & CUDA graph capture done).
        self._heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]
        self._heartbeat: HeartbeatThread | None = None
        self._heartbeat_lock = threading.Lock()

        # request telemetry, used for prefill-decode disagg
        # TODO: pass down the configuration via vLLM connector config
        # instead of env var
        self.request_telemetry = RequestTelemetryFactory.create(
            telemetry_type=os.getenv("LMCACHE_REQUEST_TELEMETRY_TYPE", "noop"),
            config={
                "endpoint": os.getenv(
                    "LMCACHE_REQUEST_TELEMETRY_ENDPOINT",
                    "http://localhost:5768/api/v1/telemetry",
                ),
            },
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the LMCache server is healthy."""
        return self._health_event.is_set()

    @property
    def world_size(self) -> int:
        """Get the kv world size."""
        return (
            self._world_size
            if self.parallel_strategy is None
            else self.parallel_strategy.kv_world_size
        )

    @property
    def worker_id(self) -> int:
        """Get the kv worker id."""
        return (
            self._worker_id
            if self.parallel_strategy is None
            else self.parallel_strategy.kv_worker_id
        )

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        # NOTE: use_mla only used in the latest `lmcache_mp_connector`,
        # and the latest `lmcache_mp_connector` will set the parallel_strategy
        if self.parallel_strategy is None:
            raise RuntimeError("parallel_strategy is not set")
        return self.parallel_strategy.use_mla

    @property
    def is_first_rank_of_pp_group(self) -> bool:
        """Is the first rank of the pipeline parallel group."""
        # NOTE: is_first_rank_of_pp_group only used in the latest
        # `lmcache_mp_connector`, and the latest `lmcache_mp_connector`
        # will set the parallel_strategy
        if self.parallel_strategy is None:
            raise RuntimeError("parallel_strategy is not set")
        return (
            self.parallel_strategy.actual_worker_id % self.parallel_strategy.tp_size
            == 0
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints
        from lmcache.v1.gpu_connector.utils import (
            ensure_contiguous_kv_caches,
        )

        # Register kv cache and send the request
        logger.info("Registering kv caches")

        layout_hints = vllm_layout_hints()
        kv_caches = ensure_contiguous_kv_caches(
            kv_caches, kv_layout=layout_hints.get("kv_layout")
        )

        self.kv_caches = kv_caches

        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self.instance_id,
                wrap_kv_caches(kv_caches),
                self.model_name,
                self.world_size,
                layout_hints,
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None

    def _ensure_heartbeat_started(self) -> None:
        """Lazily start the heartbeat thread on first use."""
        if self._heartbeat is not None:
            return
        with self._heartbeat_lock:
            if self._heartbeat is not None:
                return
            self._heartbeat = HeartbeatThread(
                mq_client=self.mq_client,
                health_event=self._health_event,
                interval=self._heartbeat_interval,
            )
            self._heartbeat.start()

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            return

        assert op.token_ids is not None
        key = self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [key, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        self._ensure_heartbeat_started()

        if not self.is_healthy:
            self.error_block_ids.update(op.block_ids)
            return

        assert op.token_ids is not None
        key = self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                key,
                self.instance_id,
                op.block_ids,
                event.ipc_handle(),
                op.skip_first_n_tokens,
            ],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, list(op.block_ids))

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        for request_id, op in zip(request_ids, ops, strict=False):
            self.submit_store_request(request_id, op, event)

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
        """
        for request_id, op in zip(request_ids, ops, strict=False):
            self.submit_retrieve_request(request_id, op, event)

    def _process_finished_stores(
        self,
        finished_req_ids_from_lmcache: set[str],
        finished_req_ids_from_engine: set[str],
    ) -> set[str]:
        """Merge LMCache-side and engine-side finished store info."""
        self.finished_stores.update(finished_req_ids_from_lmcache)
        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self._returned_finished:
                continue
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)
        ret_stores.update(self._update_and_get_finished_store())
        self._returned_finished.update(ret_stores)
        return ret_stores

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        # If unhealthy, drain all pending futures immediately
        if not self.is_healthy:
            finished_stores = set(self.store_futures.keys())
            finished_retrieves = set()
            for request_id, (
                _r_future,
                r_block_ids,
            ) in self.retrieve_futures.items():
                finished_retrieves.add(request_id)
                self.error_block_ids.update(r_block_ids)
            self.store_futures.clear()
            self.retrieve_futures.clear()

            ret_stores = self._process_finished_stores(
                finished_stores, finished_req_ids_from_engine
            )
            # A request may have a pending retrieve AND appear in
            # finished_req_ids_from_engine (it ran without loading KV after
            # the server died).  The scheduler processes finished_recving
            # first and deletes the request, so we must not also report it
            # in finished_sending.
            ret_stores -= finished_retrieves
            return ret_stores, finished_retrieves

        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, _) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result()
            finished_retrieves.add(request_id)

            if not r_result:
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Update the internal states
        ret_stores = self._process_finished_stores(
            finished_stores, finished_req_ids_from_engine
        )

        # the invocation of `get_finished` means that
        # these requests' KV caches are already fully stored.
        # or the requests normally ends without any store.
        if ret_stores:
            self.request_telemetry.on_request_store_finished(
                request_ids_set=ret_stores,
                model_name=self.model_name,
                world_size=self.world_size,
                kv_rank=self.worker_id,
            )

        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Returns the block IDs that failed due to retrieve timeout,
        then clears the internal set.
        """
        errors = self.error_block_ids.copy()
        self.error_block_ids.clear()
        return errors

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        try:
            send_lmcache_request(
                self.mq_client,
                RequestType.UNREGISTER_KV_CACHE,
                [self.instance_id],
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LMCache server did not respond to unregister within %ss. "
                "Proceeding with shutdown.",
                self._mq_timeout,
            )

        self.mq_client.close()
        self.request_telemetry.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key"""
        return IPCCacheEngineKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
        )
