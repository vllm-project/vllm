# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import threading
import time

# Third Party
import msgspec
import torch
import zmq

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
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
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


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (local) of already sent memory objects
    already_sent_indexes: list[int]

    # Indexes (remote) of allocated memory objects (to be written)
    remote_indexes: list[int]


class ProxyNotif(PDMsgBase):
    req_id: str  # The request UUID to notify the proxy


class CacheQueryRequest(PDMsgBase):
    """Query decoder for cached block keys matching a prefix."""

    keys: list[str]  # Keys to check on the decoder


class CacheQueryResponse(PDMsgBase):
    """Response with which keys are cached and their remote memory indices."""

    cached_keys: list[str]  # Keys that exist on the decoder
    cached_indexes: list[int]  # Remote memory indices for cached keys


PDMsg = Union[
    AllocRequest, AllocResponse, ProxyNotif, CacheQueryRequest, CacheQueryResponse
]


@dataclass
class PDConfig:
    role: str

    peer_host: str
    peer_init_port: int
    peer_alloc_port: int
    peer_query_port: Optional[int]

    buffer_size: int
    buffer_device: str

    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    skip_proxy_notification: bool = False

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        """Convert the LMCacheEngineConfig to PDConfig"""

        role = config.pd_role

        # Support bidirectional mode: "both" means this instance can
        # read cached KV from peers AND write new KV to peers.
        assert role in ["sender", "receiver", "both"], (
            f"Invalid role: {config.pd_role}, must be sender, receiver, or both"
        )

        assert config.pd_buffer_size is not None
        assert config.pd_buffer_device is not None

        if role == "receiver":
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
        elif role == "sender":
            if not config.pd_skip_proxy_notification:
                assert config.pd_proxy_host is not None
                assert config.pd_proxy_port is not None
        elif role == "both":
            # "both" role needs peer info (to read from decoder)
            # AND proxy info (to notify after write)
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
            if config.pd_proxy_host is not None:
                pass  # proxy notification is optional for "both"

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

        if config.pd_peer_query_port is not None:
            pd_peer_query_port = config.pd_peer_query_port[tp_rank]
        else:
            pd_peer_query_port = None

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            peer_query_port=pd_peer_query_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=corrected_device,
            skip_proxy_notification=config.pd_skip_proxy_notification,
        )


class PDBackend(AllocatorBackendInterface):
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
        self.config = config

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        self.corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        # Async transfer support: use a dedicated NIXL worker thread with a
        # queue so the vLLM worker thread is not blocked during KV transfer.
        # All NIXL GPU operations run on this single thread to avoid CUDA
        # context contention. This prevents RPC timeouts in vLLM v0.19.0's
        # multiprocess executor.
        # Standard
        import queue as queue_mod

        self._nixl_queue: queue_mod.Queue = queue_mod.Queue()
        # Started after transfer_channel init
        self._nixl_thread: threading.Thread | None = None
        # Serializes all mutations and reads of the NIXL agent state
        # (peer handshake, xfer handlers, batched_write, batched_read).
        # The worker thread holds it around each GPU op; the main thread
        # holds it around peer-connection setup so the two never touch
        # nixl_agent concurrently.
        self._nixl_agent_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        # TODO(Jiayi): add async zmq context if we want better asynchrony.
        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Initialize transfer channel
        peer_init_url = None
        self.local_id = ""
        # The receiver binds a listener on peer_init_url so senders can connect.
        # The sender (and "both" role) connects lazily via _ensure_peer_connection,
        # so they should NOT start a listener on the decoder's address.
        if (
            self.pd_config.peer_init_port is not None
            and self.pd_config.role == "receiver"
        ):
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        allocator = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        self.transfer_channel = CreateTransferChannel(
            async_mode=False,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            buffer_ptr=allocator.buffer_ptr,
            buffer_size=allocator.buffer_size,
            align_bytes=allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,  # type: ignore[arg-type]
            backends=config.nixl_backends,
            device=self.corrected_device,
        )
        self._nixl_backends = config.nixl_backends or ["UCX"]

        # Start the NIXL worker thread now that transfer_channel is ready
        self._nixl_thread = threading.Thread(
            target=self._nixl_worker_loop,
            name=f"nixl-worker-tp{metadata.worker_id}",
            daemon=True,
        )
        self._nixl_thread.start()
        self.running_threads.append(self._nixl_thread)

        # Shared state for sender and "both" roles
        self.initialized_peers: set[str] = set()
        self.mem_alloc_sockets: dict[str, zmq.Socket] = {}
        self.cache_query_sockets: dict[str, zmq.Socket] = {}

        if self.pd_config.role == "sender":
            self._init_sender()
        elif self.pd_config.role == "receiver":
            self._init_receiver()
        elif self.pd_config.role == "both":
            # Bidirectional: init sender capabilities (write to decoder)
            # AND ability to read cached KV from decoder
            self._init_sender()
        else:
            raise ValueError("Invalid PD role.")

        self.full_chunk_size_bytes = config.chunk_size

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        if self.corrected_device != "cpu":
            logger.info(f"Setting device to {self.corrected_device} ")
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

        init_func(
            aligned_buffer_size,
            shapes,
            dtypes,
            MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
        )

        return paged_mem_allocator

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        return self.memory_allocator

    def get_allocator_backend(self):
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # NOTE: no eviction and busy_loop in PD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type=alloc_type
        )

    # TODO(Jiayi): Please implement batched allocate to reduce memory
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
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type=alloc_type
        )

    # NOTE(Jiayi): If two requests have overlapped keys, will
    # the later one cause any problems here?
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    ############################################################
    # Prefiller functions
    ############################################################
    def _init_sender(self):
        if self.pd_config.skip_proxy_notification:
            logger.info(
                "pd_skip_proxy_notification=True, "
                "skipping ZMQ PUSH proxy notification setup. "
                "This mode is for external orchestrators only "
                "(e.g., vLLM Production Stack router). "
                "Do not use with LMCache's built-in disagg proxy."
            )
            self.proxy_side_channel = None
        else:
            proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
            self.proxy_side_channel = get_zmq_socket(
                self.zmq_context,
                proxy_url,
                "tcp",
                zmq.PUSH,
                "connect",
            )

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        if receiver_id in self.initialized_peers:
            return

        receiver_init_url = f"{receiver_host}:{receiver_init_port}"
        receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

        # lazy_init_peer_connection mutates nixl_agent state
        # (remote_xfer_handlers_dict) that the worker thread also touches,
        # so serialize with the worker's GPU ops.
        with self._nixl_agent_lock:
            self.transfer_channel.lazy_init_peer_connection(
                local_id=self.local_id,
                peer_id=receiver_id,
                peer_init_url=receiver_init_url,
            )

        # Set up the memory allocation socket
        mem_alloc_socket = get_zmq_socket(
            self.zmq_context,
            receiver_mem_alloc_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        self.mem_alloc_sockets[receiver_id] = mem_alloc_socket

        self.initialized_peers.add(receiver_id)

    def _remote_allocate(
        self, receiver_id: str, alloc_request: AllocRequest
    ) -> AllocResponse:
        side_channel = self.mem_alloc_sockets[receiver_id]
        side_channel.send(msgspec.msgpack.encode(alloc_request))
        msg = side_channel.recv()
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)

        return alloc_response

    def _get_remote_alloc_request(
        self, keys: Sequence[CacheEngineKey], mem_objs: List[MemoryObj]
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
        )

    # TODO(Jiayi): make this async in the future
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """
        Submit batched put tasks to transfer KV caches to peer.

        The NIXL transfer is offloaded to a background thread to avoid blocking
        the vLLM worker thread (which would cause RPC timeouts in vLLM v0.19.0's
        multiprocess executor).

        :param on_complete_callback: Optional callback invoked once per key
            after the transfer completes. Callback exceptions are caught and logged.
        :return: A single-element list containing a Future that resolves to the
            number of objects written once the NIXL RDMA transfer completes and
            the proxy notification has been sent. ``result()`` raises if the
            transfer fails. Returns ``None`` for local (no-transfer) requests.
            Callers can ignore it for fire-and-forget semantics or ``result()``
            it for synchronous completion.
        """
        # Skip PD transfer for local requests (no transfer_spec).
        # With conditional routing, direct-to-decoder requests have no disagg_spec.
        if transfer_spec is None:
            return None

        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

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

        # Allocate remote memory objects
        alloc_request = self._get_remote_alloc_request(keys, memory_objs)
        alloc_response = self._remote_allocate(receiver_id, alloc_request)
        already_sent_indexes = alloc_response.already_sent_indexes
        remote_indexes = alloc_response.remote_indexes

        # Filter out already sent memory objects and free them
        mem_objs_to_send = []
        for idx, mem_obj in enumerate(memory_objs):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
            else:
                mem_objs_to_send.append(mem_obj)

        completion_future: Future = Future()

        if mem_objs_to_send:
            # Construct transfer spec
            channel_transfer_spec = {
                "receiver_id": receiver_id,
                "remote_indexes": remote_indexes,
            }

            # Submit to the dedicated NIXL worker thread via queue.
            # The worker thread handles the blocking batched_write() call.
            self._nixl_queue.put(
                (
                    "write",
                    mem_objs_to_send,
                    channel_transfer_spec,
                    keys,
                    on_complete_callback,
                    transfer_spec,
                    completion_future,
                )
            )
        else:
            logger.debug(
                "All memory objects have been already sent to the remote peer."
                " Skipping transfer."
            )
            # Route notification through worker thread to avoid ZMQ
            # thread-safety issues (all socket access on one thread).
            if transfer_spec.is_last_prefill or on_complete_callback is not None:
                self._nixl_queue.put(
                    (
                        "notify_only",
                        keys,
                        on_complete_callback,
                        transfer_spec,
                        completion_future,
                    )
                )
            else:
                # Nothing to do — resolve immediately with 0 writes.
                completion_future.set_result(0)

        return [completion_future]

    def _nixl_worker_loop(self) -> None:
        """Dedicated NIXL worker thread. Processes transfer requests from queue.

        All NIXL GPU operations (batched_write, batched_read) run on this
        single thread to avoid CUDA context contention with the vLLM worker.
        """
        # Standard
        import queue as queue_mod

        while self.running:
            try:
                item = self._nixl_queue.get(timeout=1.0)
            except queue_mod.Empty:
                continue

            if item is None:
                break  # Shutdown signal

            op_type = item[0]
            if op_type == "write":
                (
                    _,
                    mem_objs,
                    channel_spec,
                    keys,
                    callback,
                    transfer_spec,
                    completion_future,
                ) = item
                success = False
                write_error: Optional[BaseException] = None
                num_written = 0
                try:
                    with self._nixl_agent_lock:
                        num_written = self.transfer_channel.batched_write(
                            objects=mem_objs,
                            transfer_spec=channel_spec,
                        )
                    success = True
                except Exception as e:
                    write_error = e
                    logger.error(f"NIXL write failed in worker thread: {e}")
                finally:
                    for mem_obj in mem_objs:
                        mem_obj.ref_count_down()

                if success and transfer_spec.is_last_prefill:
                    if self.proxy_side_channel is not None:
                        notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                        notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
                        self.proxy_side_channel.send(notif_msg_bytes)

                if success and callback is not None:
                    for key in keys:
                        try:
                            callback(key)
                        except Exception as e:
                            logger.warning(
                                f"on_complete_callback failed for key {key}: {e}"
                            )

                # Resolve the Future AFTER proxy notification + callbacks so
                # that future.result() returning means everything downstream
                # has observed the completion.
                if success:
                    completion_future.set_result(num_written)
                else:
                    completion_future.set_exception(
                        write_error or RuntimeError("NIXL write failed")
                    )
            elif op_type == "notify_only":
                _, keys, callback, transfer_spec, completion_future = item
                try:
                    if transfer_spec.is_last_prefill:
                        if self.proxy_side_channel is not None:
                            notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                            notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
                            self.proxy_side_channel.send(notif_msg_bytes)
                    if callback is not None:
                        for key in keys:
                            try:
                                callback(key)
                            except Exception as e:
                                logger.warning(
                                    f"on_complete_callback failed for key {key}: {e}"
                                )
                    completion_future.set_result(0)
                except Exception as e:
                    completion_future.set_exception(e)
            elif op_type == "read":
                _, buffers, channel_spec, completion_future = item
                try:
                    with self._nixl_agent_lock:
                        num_read = self.transfer_channel.batched_read(
                            buffers=buffers,
                            transfer_spec=channel_spec,
                        )
                    completion_future.set_result(num_read)
                except Exception as e:
                    logger.error(f"NIXL read failed in worker thread: {e}")
                    completion_future.set_exception(e)

    ############################################################
    # Prefiller functions end
    ############################################################

    ############################################################
    # Bidirectional NIXL: Prefiller reads cached KV from decoder
    ############################################################

    def _ensure_cache_query_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_query_port: int,
    ) -> None:
        """Set up ZMQ socket for querying decoder's cache."""
        if receiver_id in self.cache_query_sockets:
            return

        query_url = f"{receiver_host}:{receiver_query_port}"
        query_socket = get_zmq_socket(
            self.zmq_context,
            query_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        self.cache_query_sockets[receiver_id] = query_socket

    def query_remote_cache(
        self,
        receiver_id: str,
        keys: Sequence[CacheEngineKey],
    ) -> CacheQueryResponse:
        """
        Query the decoder for which keys are cached in its GPU memory.

        Returns a CacheQueryResponse with cached_keys and cached_indexes.
        Uses a timeout to avoid blocking the vLLM worker indefinitely.
        On timeout, the REQ socket is closed and removed so it will be
        recreated on the next call (REQ requires strict send/recv
        alternation — a missed recv leaves the socket in an unusable state).
        """
        query_socket = self.cache_query_sockets[receiver_id]
        str_keys = [key.to_string() for key in keys]
        query = CacheQueryRequest(keys=str_keys)
        query_socket.send(msgspec.msgpack.encode(query))
        # Use poll with timeout to avoid blocking indefinitely
        if query_socket.poll(timeout=5000):  # 5 second timeout
            resp_bytes = query_socket.recv()
        else:
            logger.warning(
                "Cache query timed out after 5s for receiver %s, resetting socket",
                receiver_id,
            )
            # Close the stuck socket — REQ socket is in recv-expected
            # state after send() without recv(), making it unusable.
            query_socket.close()
            del self.cache_query_sockets[receiver_id]
            return CacheQueryResponse(cached_keys=[], cached_indexes=[])
        resp = msgspec.msgpack.decode(resp_bytes, type=PDMsg)
        assert isinstance(resp, CacheQueryResponse)
        return resp

    def batched_submit_read_task(
        self,
        keys: Sequence[CacheEngineKey],
        local_memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> int:
        """
        Read cached KV blocks from the decoder's GPU memory into local buffers.

        This is the bidirectional NIXL read path: the prefiller queries the
        decoder for cached blocks, allocates local buffers, and reads the
        cached KV data via NIXL READ (RDMA).

        Args:
            keys: Cache keys to check on the decoder.
            local_memory_objs: Pre-allocated local MemoryObj buffers to read into.
            transfer_spec: Must contain receiver_host, receiver_init_port,
                receiver_alloc_port, and receiver_query_port.

        Returns:
            Number of blocks successfully read from the decoder.
        """
        # Skip read for local requests (no transfer_spec)
        if transfer_spec is None:
            return 0

        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
        receiver_query_port = transfer_spec.receiver_query_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host

        # Ensure NIXL peer connection is established
        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_init_port=receiver_init_port,
            receiver_alloc_port=receiver_alloc_port,
        )

        # Ensure cache query connection
        self._ensure_cache_query_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_query_port=receiver_query_port,
        )

        # Query decoder for cached blocks
        cache_resp = self.query_remote_cache(receiver_id, keys)

        if not cache_resp.cached_keys:
            logger.debug("No cached blocks found on decoder for this prefix.")
            return 0

        # Map cached keys to local buffer indices
        key_to_local_idx = {}
        for idx, key in enumerate(keys):
            key_to_local_idx[key.to_string()] = idx

        local_objs_to_read = []
        remote_indexes = []
        for cached_key, remote_idx in zip(
            cache_resp.cached_keys,
            cache_resp.cached_indexes,
            strict=False,
        ):
            local_idx = key_to_local_idx.get(cached_key)
            if local_idx is not None and local_idx < len(local_memory_objs):
                local_objs_to_read.append(local_memory_objs[local_idx])
                remote_indexes.append(remote_idx)

        if not local_objs_to_read:
            logger.debug("No matching local buffers for cached remote blocks.")
            return 0

        # Perform NIXL READ on the worker thread so that writes, reads,
        # and peer-handshake never touch nixl_agent concurrently.
        channel_transfer_spec = {
            "sender_id": receiver_id,
            "remote_indexes": remote_indexes,
        }

        read_future: Future = Future()
        self._nixl_queue.put(
            (
                "read",
                local_objs_to_read,
                channel_transfer_spec,
                read_future,
            )
        )
        num_read = read_future.result()

        logger.info(
            "Bidirectional NIXL: read %d/%d cached blocks from decoder %s",
            num_read,
            len(keys),
            receiver_id,
        )

        return num_read

    ############################################################
    # Bidirectional NIXL end
    ############################################################

    ############################################################
    # Decoder functions
    ############################################################
    def _init_receiver(self):
        # Initialize initialization side channels
        receiver_alloc_url = (
            f"{self.pd_config.peer_host}:{self.pd_config.peer_alloc_port}"
        )
        self.alloc_side_channel = get_zmq_socket(
            self.zmq_context, receiver_alloc_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self.alloc_side_channel)

        # Start the memory allocation thread
        self.mem_alloc_thread = threading.Thread(
            target=self._mem_alloc_loop, daemon=True
        )
        self.mem_alloc_thread.start()
        self.running_threads.append(self.mem_alloc_thread)

        # Start cache query listener if query port is configured
        # (enables bidirectional NIXL: prefiller can query decoder's cache)
        if self.pd_config.peer_query_port is not None:
            query_url = f"{self.pd_config.peer_host}:{self.pd_config.peer_query_port}"
            self.query_side_channel = get_zmq_socket(
                self.zmq_context, query_url, "tcp", zmq.REP, "bind"
            )
            self.side_channels.append(self.query_side_channel)

            self.cache_query_thread = threading.Thread(
                target=self._cache_query_loop, daemon=True
            )
            self.cache_query_thread.start()
            self.running_threads.append(self.cache_query_thread)
            logger.info(
                "Bidirectional NIXL: cache query listener started on %s",
                query_url,
            )

    def _allocate_and_put(self, alloc_request: AllocRequest) -> AllocResponse:
        total_allocs = len(alloc_request.keys)
        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = alloc_request.shape

        alloc_indexes = []
        already_send_indexes = []

        for idx, key_str in enumerate(alloc_request.keys):
            key = CacheEngineKey.from_string(key_str)
            if self.contains(key, pin=True):
                already_send_indexes.append(idx)
                continue

            if idx == total_allocs - 1:
                num_alloc_tokens = alloc_request.last_chunk_toks
                token_dim = fmt.token_dim()
                shape[token_dim] = num_alloc_tokens
            else:
                num_alloc_tokens = self.full_chunk_size_bytes

            mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

            # TODO(Jiayi): make busy loop allocation part of
            # memory allocator instead of backend as both PD
            # and CPU offloading might need this.
            wait_time = 0.01
            while mem_obj is None:
                logger.warning(
                    "Failed to allocate memory object, retrying...",
                )
                time.sleep(wait_time)
                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

            alloc_indexes.append(mem_obj.meta.address)

            self.put(key, mem_obj)

        return AllocResponse(
            already_sent_indexes=already_send_indexes, remote_indexes=alloc_indexes
        )

    def _mem_alloc_loop(self):
        """
        Running the memory allocation loop.
        """
        while self.running:
            try:
                # receive alloc request
                alloc_req_bytes = self.alloc_side_channel.recv()
                alloc_req = msgspec.msgpack.decode(alloc_req_bytes, type=PDMsg)
                assert isinstance(alloc_req, AllocRequest), (
                    "The request from the remote peer is not a AllocRequest"
                )

                # NOTE: it's okay to put the memory objs into the storage backend
                # first because decode vllm will not be able to see the decode
                # request until proxy receives the ack.
                alloc_resp = self._allocate_and_put(alloc_req)

                # send back response
                self.alloc_side_channel.send(msgspec.msgpack.encode(alloc_resp))

            except Exception as e:
                logger.error("Failed to process mem alloc loop: %s", str(e))
                if self.running:
                    time.sleep(0.01)

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ):
        with self.data_lock:
            self.data[key] = mem_obj

    def _cache_query_loop(self):
        """
        Listen for cache query requests from prefiller (bidirectional NIXL).
        The prefiller sends a list of keys and the decoder responds with
        which keys are cached and their memory indices.
        """
        while self.running:
            try:
                query_bytes = self.query_side_channel.recv()
            except Exception as e:
                if self.running:
                    logger.error("Cache query recv failed: %s", str(e))
                    time.sleep(0.01)
                continue

            # After recv, we MUST send a reply (ZMQ REP pattern).
            # If processing fails, send an empty response.
            try:
                query = msgspec.msgpack.decode(query_bytes, type=PDMsg)
                assert isinstance(query, CacheQueryRequest), (
                    f"Expected CacheQueryRequest, got {type(query)}"
                )

                cached_keys = []
                cached_indexes = []
                with self.data_lock:
                    for key_str in query.keys:
                        key = CacheEngineKey.from_string(key_str)
                        if mem_obj := self.data.get(key, None):
                            cached_keys.append(key_str)
                            cached_indexes.append(mem_obj.meta.address)

                resp = CacheQueryResponse(
                    cached_keys=cached_keys,
                    cached_indexes=cached_indexes,
                )
                self.query_side_channel.send(msgspec.msgpack.encode(resp))

                logger.info(
                    "Cache query: %d/%d keys cached",
                    len(cached_keys),
                    len(query.keys),
                )

            except Exception as e:
                logger.error("Failed to process cache query: %s", str(e))
                # Send empty response to unblock the REP socket
                try:
                    empty_resp = CacheQueryResponse(cached_keys=[], cached_indexes=[])
                    self.query_side_channel.send(msgspec.msgpack.encode(empty_resp))
                except Exception:
                    pass  # Socket may be broken, nothing we can do

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
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
        # TODO(Jiayi): The logic here is confusing. Ref count down
        # will be done after this function call in cache engine.
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if mem_obj.get_ref_count() == 1:
                    del self.data[key]
                return True
            return False

    ############################################################
    # Decoder functions end
    ############################################################

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        # Signal the NIXL worker thread to exit
        self._nixl_queue.put(None)
        for thread in self.running_threads:
            thread.join(timeout=5.0)
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True
