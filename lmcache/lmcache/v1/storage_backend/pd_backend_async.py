# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import os
import threading
import time
import traceback
import uuid

# Third Party
import msgspec
import torch
import zmq
import zmq.asyncio

# First Party
from lmcache import torch_dev
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.paged_cpu_gpu_memory_allocator import (
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_context
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages"""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int
    # req_id is used by the receiver for per-request chunk accounting and
    # fail-fast detection when C_req > total_chunks.  An empty string
    # means the sender does not provide an identifier (backwards-compatible);
    # in that case per-request chunk accounting and fail-fast detection are
    # skipped for this allocation request.
    req_id: str = ""
    # is_last_batch signals the final batch for this req_id so the receiver
    # can release admission and clean up per-request tracking.
    is_last_batch: bool = False
    total_chunks: int = 0  # total chunks for this request (for receiver reservation)


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (remote) of allocated memory objects (to be written).
    # One entry per key in the request; -1 means allocation failed for that slot.
    remote_indexes: list[int]

    # Indexes (local) of already sent memory objects.
    # Always empty for PDBackendAsync (no dedup), but included for
    # wire-compatibility with sync PDBackend senders that expect this field.
    already_sent_indexes: list[int] = []


class ProxyNotif(PDMsgBase):
    req_id: str  # The request id to notify the proxy


class CancelNotif(PDMsgBase):
    """Sent from sender to receiver when a request is aborted."""

    req_id: str
    keys: list[str]  # keys that receiver should release


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif, CancelNotif]


class ReservationManager:
    """
    Manages reservation-based admission control for the receiver PD buffer.

    Prevents deadlock where N concurrent requests each allocate partial chunks,
    fill the buffer, and none can complete. When a request is admitted, its
    total_chunks are reserved upfront. Subsequent physical allocations draw
    from that reservation.

    Used exclusively on the receiver side via asyncio primitives.
    """

    def __init__(
        self,
        total_chunks: int,
        allocation_timeout: float,
        condition_poll_interval: float,
    ) -> None:
        """Initialize the ReservationManager.

        :param total_chunks: Total number of chunks in the buffer.
        :param allocation_timeout: Max seconds to wait for admission.
        :param condition_poll_interval: Poll interval for condition waits.
        """
        self._total_chunks = total_chunks
        self._allocation_timeout = allocation_timeout
        self._condition_poll_interval = condition_poll_interval

        # Shared data
        self._reservations: dict[str, int] = {}  # req_id -> reserved chunks
        self._total_reserved: int = 0

        # Asyncio variant (receiver, created lazily on receiver event loop)
        self._async_admit_condition: asyncio.Condition | None = None

    def init_async_admit_condition(self) -> None:
        """Create asyncio.Condition bound to the current event loop.

        Must be called from within the target event loop.
        """
        self._async_admit_condition = asyncio.Condition()

    async def async_try_admit(self, req_id: str, total_chunks: int) -> bool:
        """Async (asyncio) admission for receiver.

        :param req_id: The request identifier to admit.
        :param total_chunks: Number of chunks to reserve.
        :return: True if admitted, False if timed out.
        """
        assert self._async_admit_condition is not None
        async with self._async_admit_condition:
            deadline = asyncio.get_running_loop().time() + self._allocation_timeout
            while True:
                available = self._total_chunks - self._total_reserved
                if available >= total_chunks:
                    self._reservations[req_id] = total_chunks
                    self._total_reserved += total_chunks
                    logger.debug(
                        "[ADMIT] req=%s admitted (async), reserved=%d, "
                        "total_reserved=%d/%d",
                        req_id,
                        total_chunks,
                        self._total_reserved,
                        self._total_chunks,
                    )
                    return True
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    logger.warning(
                        "[ADMIT] req=%s async admission timed out: need=%d, "
                        "available=%d, total_reserved=%d/%d",
                        req_id,
                        total_chunks,
                        available,
                        self._total_reserved,
                        self._total_chunks,
                    )
                    return False
                try:
                    await asyncio.wait_for(
                        self._async_admit_condition.wait(),
                        timeout=min(remaining, self._condition_poll_interval),
                    )
                except asyncio.TimeoutError:
                    pass

    async def async_release_reservation(self, req_id: str) -> None:
        """Release reservation (asyncio variant). Notifies all waiters.

        :param req_id: The request identifier whose reservation to release.
        """
        assert self._async_admit_condition is not None
        async with self._async_admit_condition:
            count = self._reservations.pop(req_id, 0)
            if count > 0:
                self._total_reserved -= count
                logger.debug(
                    "[ADMIT] req=%s reservation released (async), freed=%d, "
                    "total_reserved=%d/%d",
                    req_id,
                    count,
                    self._total_reserved,
                    self._total_chunks,
                )
                self._async_admit_condition.notify_all()

    def get_total_chunks(self) -> int:
        """Return the total buffer capacity in chunks.

        :return: Total number of chunks this manager was initialised with.
        :rtype: int
        """
        return self._total_chunks


@dataclass
class PDConfig:
    role: str

    peer_host: str
    peer_init_port: int
    peer_alloc_port: int

    proxy_host: str
    proxy_port: int

    buffer_size: int
    buffer_device: str

    allocation_timeout_sec: float
    shutdown_timeout_sec: float
    condition_poll_interval_sec: float

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        """Convert the LMCacheEngineConfig to PDConfig"""

        role = config.pd_role

        # TODO(Jiayi): Could be both if we want to do dynamic role switch.
        assert role in ["sender", "receiver"], (
            f"Invalid role: {config.pd_role}, must be either sender or receiver"
        )

        assert config.pd_buffer_size is not None
        assert config.pd_buffer_device is not None

        if role == "receiver":
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
        elif role == "sender":
            assert config.pd_proxy_host is not None
            assert config.pd_proxy_port is not None

        corrected_device = get_correct_device(
            config.pd_buffer_device, metadata.worker_id
        )

        if config.pd_peer_alloc_port is not None:
            pd_peer_alloc_port = config.pd_peer_alloc_port[tp_rank]
        else:
            pd_peer_alloc_port = None

        if config.pd_peer_init_port is not None:
            pd_peer_init_port = config.pd_peer_init_port[tp_rank]
        else:
            pd_peer_init_port = None

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=corrected_device,
            allocation_timeout_sec=config.pd_allocation_timeout_sec,
            shutdown_timeout_sec=config.pd_shutdown_timeout_sec,
            condition_poll_interval_sec=config.pd_condition_poll_interval_sec,
        )


class PDBackendAsync(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for PD Disaggregation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        self.running = True

        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        # Cache timing config values as instance attributes for convenient access.
        self._allocation_timeout = self.pd_config.allocation_timeout_sec
        self._condition_poll_interval = self.pd_config.condition_poll_interval_sec

        self.corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Initialize transfer channel
        peer_init_url = None
        self.local_id = ""
        # TODO(Jiayi): both sender and receiver have to have
        # peer_init_url if they want to do instance flip.
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        # Fallback: ensure local_id is never empty so DEALER identity is unique.
        # Senders typically don't set pd_peer_init_port. In xP1D deployments
        # multiple Prefillers may share the same proxy, so proxy_host:proxy_port
        # alone is NOT unique. We include os.getpid() and a UUID fragment to
        # guarantee a globally unique identity.
        if not self.local_id:
            self.local_id = f"sender-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Create the event loop before the transfer channel so it can be passed
        # into the channel constructor for async_mode initialization.
        if self.pd_config.role == "sender":
            self._sender_loop = asyncio.new_event_loop()
            self._sender_thread = threading.Thread(
                target=self._sender_loop.run_forever,
                daemon=True,
                name="pd-sender-async",
            )
            self._sender_thread.start()
            event_loop = self._sender_loop
        elif self.pd_config.role == "receiver":
            self._recv_loop = asyncio.new_event_loop()
            self._recv_thread = threading.Thread(
                target=self._recv_loop.run_forever,
                daemon=True,
                name="pd-receiver-async",
            )
            self._recv_thread.start()
            event_loop = self._recv_loop
        else:
            raise ValueError("Invalid PD role.")

        allocator = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        self.transfer_channel = CreateTransferChannel(
            async_mode=True,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            buffer_ptr=allocator.buffer_ptr,
            buffer_size=allocator.buffer_size,
            align_bytes=allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
            backends=config.nixl_backends,
            device=self.corrected_device,
            event_loop=event_loop,
        )

        if self.pd_config.role == "sender":
            self.initialized_peers: set[str] = set()
            self._peer_connection_lock = threading.Lock()
            # Separate async ZMQ context for sender coroutines
            self._async_zmq_context = zmq.asyncio.Context()
            self._async_alloc_sockets: dict[str, zmq.asyncio.Socket] = {}
            self._async_alloc_locks: dict[str, asyncio.Lock] = {}
            # Physical memory wait condition (woken when RDMA completes and
            # ref_count_down() frees buffer slots).
            self._staging_lock = threading.Lock()
            self._staging_condition = threading.Condition(self._staging_lock)
            # Per-request tracking for ProxyNotif ordering and abort.
            # All accessed only from coroutines on _sender_loop (no extra lock).
            self._completed_chunks: dict[str, int] = {}
            self._req_has_last: dict[str, bool] = {}
            self._req_total_chunks: dict[str, int] = {}
            self._sent_keys: dict[str, list[str]] = {}
            self._req_receiver: dict[str, str] = {}
            self._init_sender()
        elif self.pd_config.role == "receiver":
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            # Reservation-based admission control for receiver.
            self._recv_reservation_mgr = ReservationManager(
                total_chunks,
                self._allocation_timeout,
                self._condition_poll_interval,
            )
            # The asyncio primitives must be created on the receiver event loop.
            future = asyncio.run_coroutine_threadsafe(
                self._create_recv_primitives(), self._recv_loop
            )
            future.result(timeout=5)
            logger.info(
                "PDBackendAsync receiver: reservation-based admission control "
                "initialized with total_chunks=%d "
                "(buffer=%d bytes, chunk=%d bytes)",
                total_chunks,
                self._aligned_buffer_size,
                self._chunk_size_bytes,
            )
            # Per-request key tracking for rollback.
            # Maps req_id → list of key strings allocated across all batches.
            self._req_allocated_keys: dict[str, list[str]] = {}
            self._init_receiver()

        self.full_chunk_size_bytes = config.chunk_size

    def __str__(self):
        return "PDBackend"

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        if self.corrected_device != "cpu":
            logger.info(f"Setting cuda device to {self.corrected_device} ")
            torch_dev.set_device(self.corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()

        init_func = (
            paged_mem_allocator.init_cpu_memory_allocator
            if self.corrected_device == "cpu"
            else paged_mem_allocator.init_gpu_memory_allocator
        )

        # Calculate the chunk size (align_bytes) and align buffer size
        shapes = [torch.Size(metadata.kv_shape)]
        dtypes = [metadata.kv_dtype]
        chunk_size_bytes = get_size_bytes(shapes, dtypes)
        origin_buffer_size = config.pd_buffer_size
        aligned_buffer_size = origin_buffer_size // chunk_size_bytes * chunk_size_bytes

        if aligned_buffer_size == 0 and origin_buffer_size > 0:
            raise ValueError(
                f"pd_buffer_size ({origin_buffer_size}) is smaller than a "
                f"single chunk ({chunk_size_bytes}), resulting in an aligned "
                f"buffer of size 0. Please increase pd_buffer_size to be at "
                f"least {chunk_size_bytes}."
            )

        if aligned_buffer_size != origin_buffer_size:
            logger.info(
                f"Auto align pd_buffer_size, origin: {origin_buffer_size}, "
                f"aligned: {aligned_buffer_size}, chunk size: {chunk_size_bytes}. "
                f"The remaining {origin_buffer_size - aligned_buffer_size} bytes "
                f"will not be allocated."
            )

        self._chunk_size_bytes = chunk_size_bytes
        self._aligned_buffer_size = aligned_buffer_size
        # Number of tokens per chunk (used for capacity checks).
        self._chunk_token_size = metadata.kv_shape[MemoryFormat.KV_2LTD.token_dim()]

        pd_max_prefill_len = config.pd_max_prefill_len
        if pd_max_prefill_len > 0:
            capacity_tokens = (
                aligned_buffer_size // chunk_size_bytes
            ) * self._chunk_token_size
            if capacity_tokens < pd_max_prefill_len:
                raise ValueError(
                    f"PD buffer too small for the configured pd_max_prefill_len "
                    f"(role={self.pd_config.role}): "
                    f"capacity_tokens={capacity_tokens} < "
                    f"pd_max_prefill_len={pd_max_prefill_len}. "
                    f"Inputs: aligned_buffer_size={aligned_buffer_size}, "
                    f"chunk_size={chunk_size_bytes}, "
                    f"chunk_token_size={self._chunk_token_size}. "
                    f"Increase pd_buffer_size so that the buffer holds at least "
                    f"pd_max_prefill_len={pd_max_prefill_len} tokens."
                )

        init_func(
            aligned_buffer_size,
            shapes,
            dtypes,
            MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
        )

        return paged_mem_allocator

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        """Return the underlying paged CPU/GPU memory allocator.

        :return: The memory allocator instance used by this backend.
        :rtype: PagedCpuGpuMemoryAllocator
        """
        return self.memory_allocator

    def get_allocator_backend(self) -> "PDBackendAsync":
        """Return the allocator backend instance (self).

        :return: This backend instance, which implements AllocatorBackendInterface.
        :rtype: PDBackendAsync
        """
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate a single memory object from the PD buffer.

        For the sender role, this method enforces staging buffer flow control:
        it blocks until inflight chunks drop below the threshold, then attempts
        allocation with a configurable timeout. For the receiver role, allocation
        is delegated directly to the underlying memory allocator.

        Note: ``eviction`` and ``busy_loop`` parameters are accepted for interface
        compatibility but are not used in the PD backend.

        :param shapes: Shape(s) of the KV tensors to allocate.
        :param dtypes: Data type(s) of the KV tensors.
        :param fmt: Memory format, defaults to KV_2LTD.
        :param eviction: Unused; kept for interface compatibility.
        :param busy_loop: Unused; kept for interface compatibility.
        :return: The allocated MemoryObj, or None if allocation failed or the
            backend is shutting down.
        :rtype: Optional[MemoryObj]
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # NOTE: no eviction and busy_loop in PD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"

        if self.pd_config.role == "sender":
            # Fast path: try allocation immediately.
            mem_obj = self.memory_allocator.allocate(
                shapes, dtypes, fmt=fmt, allocator_type=alloc_type
            )
            if mem_obj is not None:
                return mem_obj
            # Slow path: staging buffer physically full; wait for RDMA
            # completions to free slots.  _notify_staging_freed() wakes us
            # whenever ref_count_down() returns a slot.
            deadline = time.monotonic() + self._allocation_timeout
            with self._staging_condition:
                while True:
                    if not self.running:
                        return None
                    mem_obj = self.memory_allocator.allocate(
                        shapes, dtypes, fmt=fmt, allocator_type=alloc_type
                    )
                    if mem_obj is not None:
                        return mem_obj
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._staging_condition.wait(
                        timeout=min(remaining, self._condition_poll_interval)
                    )
            logger.error("Sender staging allocation failed after timeout")
            return None
        else:
            return self.memory_allocator.allocate(
                shapes, dtypes, fmt=fmt, allocator_type=alloc_type
            )

    # allocation overhead.
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        """Allocate a batch of memory objects from the PD buffer.

        Delegates directly to the underlying memory allocator without sender
        flow control. Currently a thin wrapper; see the TODO for planned
        improvements.

        :param shapes: Shape(s) of the KV tensors to allocate.
        :param dtypes: Data type(s) of the KV tensors.
        :param batch_size: Number of memory objects to allocate.
        :param fmt: Memory format, defaults to KV_2LTD.
        :param eviction: Unused; kept for interface compatibility.
        :param busy_loop: Unused; kept for interface compatibility.
        :return: A list of allocated MemoryObj instances, or None for slots
            that failed to allocate.
        :rtype: list[Optional[MemoryObj]]
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD

        if self.pd_config.role == "sender":
            return [
                self.allocate(shapes, dtypes, fmt, eviction, busy_loop)
                for _ in range(batch_size)
            ]

        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type=alloc_type
        )

    # NOTE(Jiayi): If two requests have overlapped keys, will
    # the later one cause any problems here?
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether the given key exists in the local data store.

        :param key: The cache engine key to look up.
        :param pin: If True and the key exists, increment the memory object's
            reference count to prevent it from being freed.
        :return: True if the key is present, False otherwise.
        :rtype: bool
        :raises AssertionError: If ``key`` is not a CacheEngineKey instance.
        """
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether the key is pending in any in-flight put task.

        PDBackendAsync does not maintain a put task queue, so this always
        returns False.

        :param key: The cache engine key to check.
        :return: Always False.
        :rtype: bool
        """
        return False

    ############################################################
    # Prefiller functions
    ############################################################
    def _init_sender(self) -> None:
        """Initialize sender-side sockets and locks on the sender event loop."""
        proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
        future = asyncio.run_coroutine_threadsafe(
            self._async_init_proxy_socket(proxy_url),
            self._sender_loop,
        )
        future.result(timeout=10)

    async def _async_init_proxy_socket(self, proxy_url: str) -> None:
        """Create the async ZMQ PUSH socket for ProxyNotif messages.

        Must run on the sender event loop so the socket is loop-bound.

        :param proxy_url: The proxy host:port string.
        """
        self._async_proxy_socket = self._async_zmq_context.socket(zmq.PUSH)
        self._async_proxy_socket.connect(f"tcp://{proxy_url}")
        self._proxy_send_lock = asyncio.Lock()

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        # Fast path: no lock required if already connected.
        if receiver_id in self.initialized_peers:
            return
        with self._peer_connection_lock:
            # Double-check under the lock to prevent duplicate connections when
            # multiple vLLM worker threads call this concurrently.
            if receiver_id in self.initialized_peers:
                return

            receiver_init_url = f"{receiver_host}:{receiver_init_port}"
            receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

            # Establish the connection with the receiver/decoder.
            # The transfer channel uses an async ZMQ context (async_mode=True), so
            # we must call the async version scheduled on the sender event loop.
            future = asyncio.run_coroutine_threadsafe(
                self.transfer_channel.async_lazy_init_peer_connection(
                    local_id=self.local_id,
                    peer_id=receiver_id,
                    peer_init_url=receiver_init_url,
                ),
                self._sender_loop,
            )
            future.result()  # Block until connection is established

            # Schedule socket creation on the sender event loop to avoid
            # cross-thread issues
            future = asyncio.run_coroutine_threadsafe(
                self._async_create_alloc_socket(receiver_id, receiver_mem_alloc_url),
                self._sender_loop,
            )
            future.result(timeout=10)  # Wait for socket to be created

            self.initialized_peers.add(receiver_id)

    async def _async_create_alloc_socket(
        self, receiver_id: str, receiver_mem_alloc_url: str
    ):
        async_alloc_socket = self._async_zmq_context.socket(zmq.DEALER)
        # Use a sender-unique identity so multiple Senders connecting to the
        # same Receiver ROUTER have distinct identities (avoids undefined ZMQ
        # behavior when two DEALER sockets share the same identity string).
        sender_identity = f"{self.local_id}-to-{receiver_id}".encode()
        async_alloc_socket.setsockopt(zmq.IDENTITY, sender_identity)
        async_alloc_socket.connect(f"tcp://{receiver_mem_alloc_url}")
        self._async_alloc_sockets[receiver_id] = async_alloc_socket

    async def _async_remote_allocate(
        self, receiver_id: str, alloc_request: "Union[AllocRequest, CancelNotif]"
    ) -> AllocResponse:
        """Send an allocation or cancellation request to the remote receiver.

        :param receiver_id: The remote receiver identifier.
        :param alloc_request: AllocRequest for allocation or CancelNotif for abort.
        :return: AllocResponse from the receiver.
        """
        if receiver_id not in self._async_alloc_locks:
            self._async_alloc_locks[receiver_id] = asyncio.Lock()
        async with self._async_alloc_locks[receiver_id]:
            socket = self._async_alloc_sockets[receiver_id]
            await socket.send_multipart([b"", msgspec.msgpack.encode(alloc_request)])
            frames = await socket.recv_multipart()
            msg = frames[-1]
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)
        return alloc_response

    def _get_remote_alloc_request(
        self,
        keys: Sequence[CacheEngineKey],
        mem_objs: List[MemoryObj],
        req_id: str = "",
        is_last_batch: bool = False,
        total_chunks: int = 0,
    ) -> AllocRequest:
        """
        Get the allocation request given the keys and memory objects.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size_bytes` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = mem_objs[0].meta.fmt
        shape = mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = mem_objs[-1].meta.shape[token_dim]

        str_keys = [key.to_string() for key in keys]

        return AllocRequest(
            keys=str_keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            req_id=req_id,
            is_last_batch=is_last_batch,
            total_chunks=total_chunks,
        )

    async def _async_transfer_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        receiver_id: str,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        transfer_spec: Any = None,
    ) -> None:
        """Perform a single-batch KV transfer: remote alloc → RDMA write → callbacks.

        Runs as an independent concurrent coroutine on _sender_loop. Multiple
        batches for the same request may execute concurrently. ProxyNotif is
        sent only after ALL batches for a request have completed.

        :param keys: Cache keys for this batch.
        :param memory_objs: Memory objects to transfer (already ref_count_up'd).
        :param receiver_id: Remote receiver identifier.
        :param on_complete_callback: Optional per-key completion callback.
        :param transfer_spec: Carries req_id, is_last_prefill, etc.
        """
        completed_indexes: set[int] = set()
        num_chunks = len(memory_objs)

        req_id: str = (
            getattr(transfer_spec, "req_id", "") if transfer_spec is not None else ""
        )
        is_last_batch: bool = (
            getattr(transfer_spec, "is_last_prefill", False)
            if transfer_spec is not None
            else False
        )

        # Track which receiver this request is going to (for abort).
        if req_id:
            self._req_receiver[req_id] = receiver_id

        # Lazily initialize per-request tracking on first batch seen by the
        # sender loop.
        if req_id and req_id not in self._req_total_chunks:
            tc = (
                getattr(transfer_spec, "total_chunks", 0)
                if transfer_spec is not None
                else 0
            )
            self._req_total_chunks[req_id] = tc
            self._completed_chunks[req_id] = 0
            self._req_has_last[req_id] = False
            self._sent_keys[req_id] = []

        try:
            alloc_request = self._get_remote_alloc_request(
                keys,
                memory_objs,
                req_id=req_id,
                is_last_batch=is_last_batch,
                total_chunks=self._req_total_chunks.get(req_id, 0),
            )
            alloc_response = await self._async_remote_allocate(
                receiver_id, alloc_request
            )
            already_sent_indexes = set(alloc_response.already_sent_indexes)
            remote_indexes = alloc_response.remote_indexes

            num_keys = len(keys)
            if already_sent_indexes:
                if (
                    min(already_sent_indexes) < 0
                    or max(already_sent_indexes) >= num_keys
                ):
                    raise RuntimeError(
                        f"Invalid already_sent_indexes from receiver: "
                        f"{alloc_response.already_sent_indexes}, "
                        f"valid range [0, {num_keys})"
                    )

            expected_send_count = num_keys - len(already_sent_indexes)
            if len(remote_indexes) != expected_send_count:
                raise RuntimeError(
                    f"AllocResponse inconsistency: total_keys={num_keys}, "
                    f"already_sent={len(already_sent_indexes)}, "
                    f"remote_indexes={len(remote_indexes)}, "
                    f"expected={expected_send_count}"
                )

            mem_objs_to_send: list[MemoryObj] = []
            keys_to_send: list[CacheEngineKey] = []
            for idx, (key, mem_obj) in enumerate(zip(keys, memory_objs, strict=True)):
                if idx in already_sent_indexes:
                    mem_obj.ref_count_down()
                    completed_indexes.add(idx)
                else:
                    mem_objs_to_send.append(mem_obj)
                    keys_to_send.append(key)

            # Abort if any remote slot failed to allocate.
            for idx, (mem_obj, remote_addr) in enumerate(
                zip(mem_objs_to_send, remote_indexes, strict=True)
            ):
                if remote_addr == -1:
                    logger.warning(
                        "Receiver allocation failed for key %s (idx=%d), "
                        "aborting entire request.",
                        keys_to_send[idx],
                        idx,
                    )
                    for j, mo in enumerate(memory_objs):
                        if j not in completed_indexes:
                            mo.ref_count_down()
                            completed_indexes.add(j)
                    if req_id:
                        await self._abort_request(req_id)
                    return

            # Track all keys (including deduped) for abort cleanup.
            if req_id:
                sent = self._sent_keys.setdefault(req_id, [])
                sent.extend(k.to_string() for k in keys)

            if mem_objs_to_send:
                channel_transfer_spec = {
                    "receiver_id": receiver_id,
                    "remote_indexes": remote_indexes,
                }

                await self.transfer_channel.async_batched_write(
                    objects=mem_objs_to_send,
                    transfer_spec=channel_transfer_spec,
                )
                for idx, mem_obj in enumerate(memory_objs):
                    if idx in completed_indexes:
                        continue
                    before = mem_obj.get_ref_count()
                    mem_obj.ref_count_down()
                    logger.debug(
                        "[SENDER] chunk %d ref_count: %d -> %d",
                        idx,
                        before,
                        before - 1,
                    )
                    completed_indexes.add(idx)
                logger.debug(
                    "[SENDER] req=%s batch done, freed %d chunks, free_chunks=%d",
                    req_id,
                    num_chunks,
                    self._get_free_chunks(),
                )

            # Update per-request completion tracking.
            if req_id:
                self._completed_chunks[req_id] = (
                    self._completed_chunks.get(req_id, 0) + num_chunks
                )
                if is_last_batch:
                    self._req_has_last[req_id] = True
                await self._check_and_send_proxy_notif(req_id, transfer_spec)
            elif is_last_batch:
                # Legacy path: no req_id, send ProxyNotif immediately.
                await self._send_proxy_notif(transfer_spec)

            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            "on_complete_callback failed for key %s: %s", key, e
                        )
        except BaseException as e:
            if not isinstance(e, asyncio.CancelledError):
                logger.error(
                    "Async transfer task failed: %s\n%s",
                    str(e),
                    traceback.format_exc(),
                )
            for idx, mem_obj in enumerate(memory_objs):
                if idx not in completed_indexes:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        pass
            # Abort the entire request so ProxyNotif is never sent
            # for data that was never written to the receiver.
            if req_id and not isinstance(e, asyncio.CancelledError):
                await self._abort_request(req_id)
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._notify_staging_freed()

    def _notify_staging_freed(self) -> None:
        """Wake threads blocked in allocate() so they can retry after RDMA frees memory.

        Called from _async_transfer_task finally block (on sender event loop)
        after ref_count_down() has returned buffer slots to the allocator.
        threading.Condition.notify_all() is non-blocking and safe to call from
        an asyncio coroutine.
        """
        if self.pd_config.role == "sender":
            with self._staging_condition:
                self._staging_condition.notify_all()

    async def _check_and_send_proxy_notif(
        self, req_id: str, transfer_spec: Any
    ) -> None:
        """Send ProxyNotif once all RDMA batches for req_id are complete.

        Fires only when both conditions hold:
        1. All reserved chunks have completed RDMA (completed >= total), OR
           total_chunks==0 (legacy sender without reservation).
        2. The is_last_prefill batch has been seen (_req_has_last is True).

        :param req_id: The request identifier.
        :param transfer_spec: Used to extract req_id for the notification.
        """
        total = self._req_total_chunks.get(req_id, 0)
        completed = self._completed_chunks.get(req_id, 0)
        has_last = self._req_has_last.get(req_id, False)

        if has_last and (total == 0 or completed >= total):
            await self._send_proxy_notif(transfer_spec)
            # Release sender reservation now that RDMA is complete.
            # Clean up per-request state.
            self._completed_chunks.pop(req_id, None)
            self._req_has_last.pop(req_id, None)
            self._req_total_chunks.pop(req_id, None)
            self._sent_keys.pop(req_id, None)
            self._req_receiver.pop(req_id, None)

    async def _send_proxy_notif(self, transfer_spec: Any) -> None:
        """Encode and send a ProxyNotif to the proxy side channel.

        :param transfer_spec: Provides the req_id for the notification.
        """
        req_id = getattr(transfer_spec, "req_id", "") if transfer_spec else ""
        if not req_id:
            return
        try:
            notif_msg = ProxyNotif(req_id=req_id)
            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
            async with self._proxy_send_lock:
                await self._async_proxy_socket.send(notif_msg_bytes)
        except Exception as e:
            logger.error("Failed to send ProxyNotif for req %s: %s", req_id, e)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Submit a transfer batch as a concurrent coroutine on the sender loop.

        Each batch runs as an independent concurrent coroutine — no per-receiver
        serialization. ProxyNotif is sent after all batches for a request
        complete (tracked via _completed_chunks and _req_has_last).

        :param keys: Cache keys for this batch.
        :param memory_objs: Memory objects to transfer.
        :param transfer_spec: Transfer specification (carries req_id, etc.).
        :param on_complete_callback: Optional per-key completion callback.
        """
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        try:
            receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
            receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
            receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
            receiver_host = transfer_spec.receiver_host

            self._ensure_peer_connection(
                receiver_id=receiver_id,
                receiver_host=receiver_host,
                receiver_init_port=receiver_init_port,
                receiver_alloc_port=receiver_alloc_port,
            )

            asyncio.run_coroutine_threadsafe(
                self._async_transfer_task(
                    keys=list(keys),
                    memory_objs=list(memory_objs),
                    receiver_id=receiver_id,
                    on_complete_callback=on_complete_callback,
                    transfer_spec=transfer_spec,
                ),
                self._sender_loop,
            )
        except Exception as e:
            for mem_obj in memory_objs:
                try:
                    mem_obj.ref_count_down()
                except Exception:
                    pass
            logger.error(
                "batched_submit_put_task failed, ref counts rolled back: %s", e
            )
            raise

    def cancel_request(self, req_id: str) -> None:
        """Cancel an in-flight or pending request.

        Wakes staging allocate() waiters and schedules cleanup on
        the sender event loop.

        :param req_id: The request identifier to cancel.
        """
        logger.info("[CANCEL] req=%s cancel_request called", req_id)
        if hasattr(self, "_staging_condition"):
            with self._staging_condition:
                self._staging_condition.notify_all()
        if hasattr(self, "_sender_loop"):
            asyncio.run_coroutine_threadsafe(
                self._abort_request(req_id), self._sender_loop
            )

    async def _abort_request(self, req_id: str) -> None:
        """Clean up request state and notify receiver of cancellation.

        Sends CancelNotif to the receiver so it can release allocated
        keys and reservation. Then clears all per-request tracking.

        :param req_id: The request identifier to abort.
        """
        logger.info("[ABORT] req=%s _abort_request starting", req_id)
        sent_keys = self._sent_keys.get(req_id, [])
        if sent_keys:
            receiver_id = self._req_receiver.get(req_id)
            if receiver_id:
                try:
                    cancel_notif = CancelNotif(req_id=req_id, keys=sent_keys)
                    await self._async_remote_allocate(receiver_id, cancel_notif)
                except Exception as e:
                    logger.warning(
                        "[ABORT] req=%s failed to send CancelNotif: %s",
                        req_id,
                        e,
                    )

        # Clean up per-request state.
        self._completed_chunks.pop(req_id, None)
        self._req_has_last.pop(req_id, None)
        self._req_total_chunks.pop(req_id, None)
        self._sent_keys.pop(req_id, None)
        self._req_receiver.pop(req_id, None)

    ############################################################
    # Prefiller functions end
    ############################################################

    ############################################################
    # Decoder functions
    ############################################################
    async def _create_recv_primitives(self) -> None:
        """Create asyncio primitives bound to the receiver event loop.

        Must be called from within the receiver event loop.
        """
        self._router_send_lock = asyncio.Lock()
        self._pending_alloc_tasks: set[asyncio.Task] = set()
        # Initialize the async condition for reservation-based admission.
        self._recv_reservation_mgr.init_async_admit_condition()
        # Condition notified when remove() frees a chunk, allowing blocked
        # allocation retries to wake up immediately instead of polling.
        self._alloc_freed_condition: asyncio.Condition = asyncio.Condition()

    def _init_receiver(self):
        """
        Launch the async memory allocation server coroutine on the already-running
        receiver event loop (self._recv_loop, created before the transfer channel).
        """
        asyncio.run_coroutine_threadsafe(
            self._async_mem_alloc_server(), self._recv_loop
        )

    async def _async_mem_alloc_server(self):
        """
        Async ZMQ ROUTER server for memory allocation requests.
        Replaces the blocking _mem_alloc_loop / _mem_alloc_thread.
        Uses a ROUTER socket instead of REP so that multiple concurrent
        senders (xP1D topology) can each have their requests received and
        dispatched independently — admission control inside
        ``_handle_alloc_request`` only blocks the per-request coroutine, not
        the receive loop.
        """
        async_ctx = zmq.asyncio.Context()
        socket = async_ctx.socket(zmq.ROUTER)
        alloc_port = self.pd_config.peer_alloc_port
        socket.bind(f"tcp://*:{alloc_port}")
        logger.info(f"Async mem alloc server listening on port {alloc_port}")
        try:
            while self.running:
                try:
                    frames = await socket.recv_multipart()
                    # ROUTER frames: [identity, empty_delimiter, payload]
                    identity = frames[0]
                    payload = frames[-1]
                    task = asyncio.create_task(
                        self._handle_alloc_request(socket, identity, payload)
                    )
                    self._pending_alloc_tasks.add(task)
                    task.add_done_callback(self._pending_alloc_tasks.discard)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Failed to process async mem alloc: %s", str(e))
                    if self.running:
                        await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            socket.close()
            async_ctx.term()

    async def _handle_alloc_request(
        self,
        socket: zmq.asyncio.Socket,
        identity: bytes,
        payload: bytes,
    ) -> None:
        """Handle a single allocation or cancellation request from a sender.

        Dispatches AllocRequest to _async_allocate_and_put. CancelNotif
        releases allocated keys and the receiver reservation for the request.

        On any exception, sends an error AllocResponse so the sender is
        never left waiting on recv_multipart.

        :param socket: The ROUTER socket to send the response on.
        :param identity: The sender identity frame.
        :param payload: The raw msgpack-encoded message bytes.
        """
        n_keys = 0
        try:
            msg = msgspec.msgpack.decode(payload, type=PDMsg)

            if isinstance(msg, CancelNotif):
                # Release keys and reservation for the cancelled request.
                req_id = msg.req_id
                for key_str in msg.keys:
                    try:
                        key = CacheEngineKey.from_string(key_str)
                        self.remove(key)
                    except Exception as exc:
                        logger.warning(
                            "Failed to remove key %s during cancel for req %s: %s",
                            key_str,
                            req_id,
                            exc,
                        )
                await self._recv_reservation_mgr.async_release_reservation(req_id)
                self._req_allocated_keys.pop(req_id, None)
                # Send a no-op response so the sender's recv_multipart unblocks.
                resp = AllocResponse(remote_indexes=[])
                async with self._router_send_lock:
                    await socket.send_multipart(
                        [identity, b"", msgspec.msgpack.encode(resp)]
                    )
                return

            if not isinstance(msg, AllocRequest):
                raise ValueError(
                    f"Expected AllocRequest from remote peer, got {type(msg).__name__}"
                )
            n_keys = len(msg.keys)
            alloc_resp = await self._async_allocate_and_put(msg)
            resp_bytes = msgspec.msgpack.encode(alloc_resp)
            async with self._router_send_lock:
                await socket.send_multipart([identity, b"", resp_bytes])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Failed to process alloc request from %s: %s",
                identity,
                str(e),
            )
            try:
                error_resp = AllocResponse(remote_indexes=[-1] * max(n_keys, 1))
                async with self._router_send_lock:
                    await socket.send_multipart(
                        [identity, b"", msgspec.msgpack.encode(error_resp)]
                    )
            except Exception:
                logger.warning("Failed to send error response to %s", identity)

    async def _async_allocate_and_put(
        self, alloc_request: AllocRequest
    ) -> AllocResponse:
        """Allocate remote memory slots and register KV objects.

        Uses reservation-based admission: on the first batch for a request,
        reserves total_chunks upfront so the buffer is never over-committed.
        Subsequent batches draw against the existing reservation.

        :param alloc_request: The allocation request from the sender.
        :return: AllocResponse with one remote_index per key (-1 on failure).
        :raises RuntimeError: On fail-fast overflow or allocation timeout.
        """
        total_allocs = len(alloc_request.keys)
        req_id = alloc_request.req_id

        # Reservation-based admission: admit on first batch for this req_id.
        is_first_batch = req_id and (req_id not in self._req_allocated_keys)
        if is_first_batch:
            if alloc_request.total_chunks == 0:
                raise RuntimeError(
                    f"Receiver requires total_chunks > 0 for req {req_id}. "
                    f"Legacy senders (total_chunks=0) are no longer supported."
                )
            # Fail fast: a request whose KV needs more chunks than the entire
            # buffer holds can never be admitted, no matter how much the buffer
            # drains. Without this guard admission blocks for the full
            # pd_allocation_timeout_sec and then fails with a misleading
            # "over-subscribed" timeout, tearing down the transfer mid-flight.
            buffer_total_chunks = self._recv_reservation_mgr.get_total_chunks()
            if alloc_request.total_chunks > buffer_total_chunks:
                raise RuntimeError(
                    f"PD receiver buffer too small for req {req_id}: the request "
                    f"needs {alloc_request.total_chunks} chunks but the buffer "
                    f"holds only {buffer_total_chunks} chunks "
                    f"(pd_buffer_size={self._aligned_buffer_size} bytes). This "
                    f"prefill can never be admitted; increase pd_buffer_size to "
                    f"hold at least {alloc_request.total_chunks} chunks or reduce "
                    f"the prefill length."
                )
            admitted = await self._recv_reservation_mgr.async_try_admit(
                req_id, alloc_request.total_chunks
            )
            if not admitted:
                raise RuntimeError(
                    f"Receiver reservation admission timed out or was aborted "
                    f"for req {req_id} (total_chunks={alloc_request.total_chunks}). "
                    f"Buffer may be over-subscribed."
                )

        # Fail-fast: detect if cumulative chunks exceed declared total_chunks.
        if req_id:
            prev_count = len(self._req_allocated_keys.get(req_id, []))
            new_total = prev_count + total_allocs

            if new_total > alloc_request.total_chunks:
                # Rollback any prior batches
                prior_keys = self._req_allocated_keys.get(req_id, [])
                for prior_key_str in prior_keys:
                    try:
                        self.remove(CacheEngineKey.from_string(prior_key_str))
                    except Exception as e:
                        logger.warning(
                            "Rollback failed for key %s: %s", prior_key_str, e
                        )

                # Clean up tracking and release reservation
                self._req_allocated_keys.pop(req_id, None)
                await self._recv_reservation_mgr.async_release_reservation(req_id)
                raise RuntimeError(
                    f"Request {req_id} protocol violation: declared total_chunks="
                    f"{alloc_request.total_chunks} but attempting {new_total} chunks."
                )
        else:
            logger.debug(
                "AllocRequest has no req_id — per-request chunk accounting "
                "is disabled for this batch"
            )

        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)

        alloc_indexes: list[int] = []
        already_sent_indexes: list[int] = []
        current_batch_keys: list[str] = []

        try:
            for idx, key_str in enumerate(alloc_request.keys):
                key = CacheEngineKey.from_string(key_str)
                if self.contains(key, pin=True):
                    already_sent_indexes.append(idx)
                    current_batch_keys.append(key_str)
                    continue

                if idx == total_allocs - 1:
                    token_dim = fmt.token_dim()
                    shape[token_dim] = alloc_request.last_chunk_toks

                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
                deadline = asyncio.get_running_loop().time() + self._allocation_timeout
                while mem_obj is None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Failed to allocate memory for key {key} after "
                            f"timeout (~{self._allocation_timeout:.0f}s). "
                            f"req_id={req_id}, key_index={idx}/{total_allocs}."
                        )
                    async with self._alloc_freed_condition:
                        try:
                            await asyncio.wait_for(
                                self._alloc_freed_condition.wait(),
                                timeout=min(remaining, self._condition_poll_interval),
                            )
                        except asyncio.TimeoutError:
                            pass
                    mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

                alloc_indexes.append(mem_obj.meta.address)
                self.put(key, mem_obj)
                logger.debug(
                    "[RECEIVER] alloc key=%s addr=%d ref_count=%d",
                    key_str,
                    mem_obj.meta.address,
                    mem_obj.get_ref_count(),
                )

                current_batch_keys.append(key_str)
                logger.debug(
                    "[PD-ALLOC] req=%s alloc chunk %d/%d, free_chunks=%d, data_size=%d",
                    req_id,
                    idx + 1,
                    total_allocs,
                    self._get_free_chunks(),
                    len(self.data),
                )
        except BaseException:
            # Rollback: remove chunks from the current batch.
            for rollback_key_str in current_batch_keys:
                try:
                    rollback_key = CacheEngineKey.from_string(rollback_key_str)
                    self.remove(rollback_key)
                except Exception as re:
                    logger.warning(
                        "Rollback remove failed for key %s: %s",
                        rollback_key_str,
                        re,
                    )

            # If this request has prior successful batches, clean them up too.
            # Any batch failure means the entire request is invalid since the
            # decoder needs all chunks to proceed.
            if req_id:
                prior_keys = self._req_allocated_keys.get(req_id, [])
                if prior_keys:
                    logger.warning(
                        "[PD-ALLOC] req=%s batch failed, rolling back %d chunks "
                        "from prior batches",
                        req_id,
                        len(prior_keys),
                    )
                    for prior_key_str in prior_keys:
                        try:
                            prior_key = CacheEngineKey.from_string(prior_key_str)
                            self.remove(prior_key)
                        except Exception as re:
                            logger.warning(
                                "Rollback remove failed for prior key %s: %s",
                                prior_key_str,
                                re,
                            )

                # Clean up tracking and release reservation.
                self._req_allocated_keys.pop(req_id, None)
                await self._recv_reservation_mgr.async_release_reservation(req_id)
            raise

        # All allocations succeeded.
        if req_id:
            if req_id not in self._req_allocated_keys:
                self._req_allocated_keys[req_id] = []
            self._req_allocated_keys[req_id].extend(current_batch_keys)
            if alloc_request.is_last_batch:
                self._req_allocated_keys.pop(req_id, None)
                await self._recv_reservation_mgr.async_release_reservation(req_id)

        return AllocResponse(
            remote_indexes=alloc_indexes, already_sent_indexes=already_sent_indexes
        )

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ) -> None:
        """Store a memory object in the local data dictionary."""
        with self.data_lock:
            if key in self.data:
                logger.info(
                    "Duplicate put for key %s in PDBackendAsync.put(); "
                    "dropping new object.",
                    key,
                )
                mem_obj.ref_count_down()
                return
            self.data[key] = mem_obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve the memory object for the given key (blocking).

        Since PDBackendAsync uses push-based transfer, the key is expected to
        already be present in local data.

        :param key: The cache engine key to retrieve.
        :return: The corresponding MemoryObj.
        :rtype: MemoryObj
        :raises AssertionError: If the key is not found in local data.
        """
        with self.data_lock:
            # NOTE(Jiayi): we assume that the key must be in local data
            # because we are using a push-based transfer
            mem_obj = self.data.get(key, None)
            assert mem_obj is not None, f"Key {key} not found in local data."
            return mem_obj

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        with self.data_lock:
            mem_obj = self.data.get(key, None)
            if mem_obj is not None:
                before_rc = mem_obj.get_ref_count()
                mem_obj.ref_count_down()
                deleted = False
                if mem_obj.get_ref_count() == 0:
                    del self.data[key]
                    logger.debug(
                        "[PD-FREE] remove key=%s, addr=%d, ref_count_before=%d, "
                        "data_size=%d, free_chunks_before=%d",
                        key,
                        mem_obj.meta.address,
                        before_rc,
                        len(self.data),
                        self._get_free_chunks(),
                    )
                    deleted = True
                # Notify any coroutines blocked waiting for free memory.
                # _alloc_freed_condition and _recv_loop only exist on the
                # receiver; remove() is also called on the sender, so the
                # hasattr guards are intentional.  run_coroutine_threadsafe is
                # used because remove() may be called from any OS thread while
                # the receiver event loop runs on a dedicated thread.
                if (
                    deleted
                    and hasattr(self, "_alloc_freed_condition")
                    and hasattr(self, "_recv_loop")
                ):
                    loop = self._recv_loop
                    if loop.is_running():

                        async def _notify_freed() -> None:
                            async with self._alloc_freed_condition:
                                self._alloc_freed_condition.notify_all()

                        asyncio.run_coroutine_threadsafe(_notify_freed(), loop)
                return True
            logger.debug("[PD-FREE] remove: key=%s NOT FOUND", key)
            return False

    def _get_free_chunks(self) -> int:
        """Return number of free chunks in the PD buffer allocator.

        Reads ``free_blocks`` from the appropriate underlying allocator
        (CPU or GPU depending on ``corrected_device``).  This is a
        best-effort diagnostic value: it is read without holding any
        additional lock, so the count may be stale by the time it is logged.

        :return: Number of currently free blocks in the allocator, or -1 if
            the allocator does not expose a ``free_blocks`` attribute.
        :rtype: int
        """
        alloc = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        try:
            return len(alloc.free_blocks)
        except AttributeError:
            return -1

    def _get_total_chunks(self) -> int:
        """Return total number of chunks in the PD buffer.

        Computed as ``_aligned_buffer_size // _chunk_size_bytes``.

        :return: Total number of fixed-size chunks in the PD buffer.
        :rtype: int
        """
        return self._aligned_buffer_size // self._chunk_size_bytes

    ############################################################
    # Decoder functions end
    ############################################################

    @staticmethod
    def _shutdown_loop(
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        timeout: float = 5.0,
    ) -> None:
        """Cancel all pending tasks on *loop*, stop it, and join the thread.

        Uses a ``threading.Event`` to synchronize shutdown completion so that
        ``thread.join`` is only called after the loop has actually stopped,
        preventing thread or resource leaks when the loop takes time to drain.

        :param loop: The event loop to shut down.
        :param thread: The thread running the event loop.
        :param timeout: Maximum seconds to wait for shutdown and thread join.
        """
        shutdown_done = threading.Event()

        async def _cancel_and_stop() -> None:
            tasks = [
                t
                for t in asyncio.all_tasks(loop)
                if t is not asyncio.current_task() and not t.done()
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()
            shutdown_done.set()

        if loop.is_running():
            loop.call_soon_threadsafe(loop.create_task, _cancel_and_stop())
            shutdown_done.wait(timeout=timeout)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Event loop thread %s did not terminate within %.1fs timeout.",
                thread.name,
                timeout,
            )

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        # Wake up any threads blocked on the sender staging condition so they
        # can observe running=False and exit cleanly.
        if hasattr(self, "_staging_condition"):
            with self._staging_condition:
                self._staging_condition.notify_all()
        for thread in self.running_threads:
            thread.join()
        # Shut down sender async loop if present
        if hasattr(self, "_sender_loop"):
            self._shutdown_loop(
                self._sender_loop,
                self._sender_thread,
                timeout=self.pd_config.shutdown_timeout_sec,
            )
            # Close async alloc sockets
            for sock in self._async_alloc_sockets.values():
                try:
                    sock.close()
                except Exception:
                    pass
            try:
                self._async_zmq_context.term()
            except Exception:
                pass
        # Shut down receiver async loop if present
        if hasattr(self, "_recv_loop"):
            # Wait for any in-flight allocation tasks to finish gracefully
            # before tearing down the loop.
            if hasattr(self, "_pending_alloc_tasks"):
                try:

                    async def _wait_pending() -> None:
                        """Await all pending alloc tasks with a timeout."""
                        pending = list(self._pending_alloc_tasks)
                        if pending:
                            await asyncio.wait(
                                pending,
                                timeout=self.pd_config.shutdown_timeout_sec,
                            )

                    future = asyncio.run_coroutine_threadsafe(
                        _wait_pending(), self._recv_loop
                    )
                    future.result(timeout=self.pd_config.shutdown_timeout_sec + 1)
                except Exception:
                    logger.debug(
                        "Timed out waiting for pending alloc tasks during shutdown"
                    )
            self._shutdown_loop(
                self._recv_loop,
                self._recv_thread,
                timeout=self.pd_config.shutdown_timeout_sec,
            )
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin the memory object for the given key to prevent eviction.

        PDBackendAsync has no eviction mechanism, so this is a no-op that
        always returns True.

        :param key: The cache engine key to pin.
        :return: Always True.
        :rtype: bool
        """
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin the memory object for the given key.

        PDBackendAsync has no eviction mechanism, so this is a no-op that
        always returns True.

        :param key: The cache engine key to unpin.
        :return: Always True.
        :rtype: bool
        """
        return True
