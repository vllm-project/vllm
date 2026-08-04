# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
import asyncio
import enum

# Third Party
import msgspec
import torch
import zmq
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    ErrorMsg,
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
from lmcache.v1.rpc_utils import (
    DEFAULT_SOCKET_RECV_TIMEOUT_MS,
    DEFAULT_SOCKET_SEND_TIMEOUT_MS,
    get_zmq_context,
    get_zmq_socket_with_timeout,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    P2PInitSideMsg,
    P2PInitSideRetMsg,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller import LMCacheWorker

logger = init_logger(__name__)


class P2PMsgBase(msgspec.Struct, tag=True):
    """Base class for all P2P-related messages"""

    pass


class BatchedLookupAndGetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    lookup_id: str

    receiver_id: str

    # CacheEngineKey in string form
    keys: list[str]

    # Indexes (remote) of allocated memory objects (to be written)
    mem_indexes: list[int]


class BatchedLookupAndGetRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of hit chunks
    num_hit_chunks: int


class BatchedLookupAndPutMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    sender_id: str

    # CacheEngineKey in string form
    keys: list[str]

    # Number of tokens for each chunk
    offsets: list[int]

    # Indexes (remote) of allocated memory objects (to be read)
    mem_indexes: list[int]


class BatchedLookupAndPutRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of read chunks
    num_read_chunks: int


class P2PErrorCode(enum.Enum):
    """P2P error codes enumeration"""

    P2P_SERVER_ERROR = enum.auto()
    UNKNOWN_MSG_TYPE = enum.auto()
    REMOTE_XFER_HANDLER_NOT_INITIALIZED = enum.auto()


class P2PErrorMsg(P2PMsgBase):
    """
    Error message, return error code to client.

    -1 represents unknown msg type;
    -2 represents remote xfer handler not initialized,
        call `_ensure_peer_connection` first;
    -3 represents p2p peer_request_handler error;
    """

    error_code: P2PErrorCode


P2PMsg = Union[
    BatchedLookupAndGetMsg,
    BatchedLookupAndGetRetMsg,
    BatchedLookupAndPutMsg,
    BatchedLookupAndPutRetMsg,
    P2PErrorMsg,
]


@dataclass
class PeerInfo:
    """Peer information"""

    peer_init_url: str  # peer id
    peer_lookup_url: str
    lookup_lock: asyncio.Lock
    lookup_socket: zmq.asyncio.Socket

    def update_peer_lookup_url(self, new_peer_lookup_url: str):
        if self.peer_lookup_url != new_peer_lookup_url:
            logger.info(
                "Target peer %s lookup url changed from %s to %s",
                self.peer_init_url,
                self.peer_lookup_url,
                new_peer_lookup_url,
            )
            self.peer_lookup_url = new_peer_lookup_url

    def update_lookup_socket(self, new_lookup_socket: zmq.asyncio.Socket):
        try:
            self.lookup_socket.close(linger=0)
        except Exception as e:
            logger.error("Failed to close peer %s lookup socket", self.peer_init_url, e)
        self.lookup_socket = new_lookup_socket


# TODO(Jiayi): handle asymmetric TP.
class P2PBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_worker: "LMCacheWorker",
    ):
        self.config = config
        self.loop = loop
        self.lmcache_worker = lmcache_worker
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        assert config.p2p_host is not None, "p2p_host must be specified"
        assert config.p2p_init_ports is not None, "p2p_init_ports must be specified"
        assert config.p2p_lookup_ports is not None, "p2p_lookup_ports must be specified"

        # Load timeout configurations from extra_config (in milliseconds)
        self.socket_recv_timeout_ms = config.get_extra_config_value(
            "p2p_socket_recv_timeout_ms", DEFAULT_SOCKET_RECV_TIMEOUT_MS
        )
        self.socket_send_timeout_ms = config.get_extra_config_value(
            "p2p_socket_send_timeout_ms", DEFAULT_SOCKET_SEND_TIMEOUT_MS
        )

        # Load max retry count from extra_config
        self.max_retry_count = config.get_extra_config_value("p2p_max_retry_count", 3)

        # tp rank is worker id for now
        self.tp_rank = metadata.worker_id

        self.peer_host = config.p2p_host
        self.peer_init_port = config.p2p_init_ports[self.tp_rank]
        self.peer_init_url = f"{self.peer_host}:{self.peer_init_port}"

        self.peer_lookup_port = config.p2p_lookup_ports[self.tp_rank]
        self.peer_lookup_url = f"{self.peer_host}:{self.peer_lookup_port}"

        self.lmcache_instance_id = config.lmcache_instance_id

        # A CacheEngineKey (in int form) -> a list of
        # (peer_init_url, peer_lookup_url, location)
        self.local_lookup_cache: dict[int, tuple[str, str, str]] = {}
        # the target peer info mapping
        self.target_peer_info_mapping: dict[str, PeerInfo] = {}
        # the lock for updating target peer info mapping
        self.update_peer_lock = asyncio.Lock()

        # A lookup_id -> (peer_init_url, location)
        # TODO(chunxiaozheng): location is not used for now
        self.lookup_id_to_peer_mapping: dict[str, tuple[str, str]] = {}

        # TODO(Jiayi): support gpu and local storage p2p as well.
        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.full_size_shapes = self.memory_allocator.cpu_allocator.shapes
        self.dtypes = self.memory_allocator.cpu_allocator.dtypes
        self.fmt: MemoryFormat = (
            MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        )
        self.chunk_size = config.chunk_size

        device_type = (
            "cpu" if config.nixl_buffer_device is None else config.nixl_buffer_device
        )
        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=True,
            role="both",
            buffer_ptr=self.memory_allocator.cpu_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.cpu_allocator.buffer_size,
            align_bytes=self.memory_allocator.cpu_allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=self.peer_init_url,
            peer_lookup_url=self.peer_lookup_url,
            backends=config.nixl_backends,
            event_loop=loop,
            device=device_type,
        )

        self.running = asyncio.Event()
        self.running.set()
        self.async_context: Optional[zmq.asyncio.Context] = None
        self.async_peer_socket: Optional[zmq.asyncio.Socket] = None
        asyncio.run_coroutine_threadsafe(
            self._run_peer_request_handler_with_recovery(), loop
        )

    def __str__(self) -> str:
        return "P2PBackend"

    async def _run_peer_request_handler_with_recovery(self) -> None:
        """
        Wrapper method that runs _handle_peer_requests with exception handling.
        This ensures the handler keeps running even if unexpected errors occur.
        """
        while self.running.is_set():
            try:
                await self._handle_peer_requests()
                # If _handle_peer_requests exits normally, break the loop
                break
            except asyncio.CancelledError:
                logger.info("Peer request handler cancelled, shutting down")
                break
            except Exception as e:
                logger.error(
                    "Peer request handler crashed: %s",
                    e,
                    exc_info=True,
                )
                # Fast failure: log error but continue running
                # Add small delay to prevent tight error loop
                await asyncio.sleep(0.1)
                if self.async_peer_socket is not None:
                    logger.warning("Closing async peer socket.")
                    try:
                        self.async_peer_socket.close(linger=0)
                    except Exception as e:
                        logger.warning(
                            "Failed to close peer socket: %s",
                            e,
                            exc_info=True,
                        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        # Convert to hashes (int form)
        hashes = [key.chunk_hash for key in keys]

        # Tier 1 lookup: local lookup cache
        # TODO(Jiayi): Please implement the local lookup cache.

        # Tier 2 lookup in controller
        msg = BatchedP2PLookupMsg(
            instance_id=self.lmcache_instance_id,
            worker_id=self.tp_rank,
            hashes=hashes,
        )
        ret_msg = await self.lmcache_worker.async_put_and_wait_msg(msg)

        if isinstance(ret_msg, ErrorMsg):
            logger.error(
                "Controller returned error for batched P2P lookup: %s",
                ret_msg.error,
            )
            return 0

        assert isinstance(ret_msg, BatchedP2PLookupRetMsg), (
            f"Expected BatchedP2PLookupRetMsg, got {type(ret_msg)}"
        )

        # NOTE(Jiayi): For now we only support one peer hit.
        layout_info = ret_msg.layout_info[0]
        _, location, num_hit_chunks, target_peer_init_url = layout_info

        logger.info(f"Got layout info from controller: {layout_info}")

        if num_hit_chunks > 0:
            try:
                await self._ensure_peer_connection(target_peer_init_url)
                self.lookup_id_to_peer_mapping[lookup_id] = (
                    target_peer_init_url,
                    location,
                )
            except Exception as e:
                logger.error(
                    "Failed to ensure peer connection for lookup_id %s: %s",
                    lookup_id,
                    e,
                    exc_info=True,
                )
                return 0

        # TODO(Jiayi): We could potentially update the local cache here.
        # Or we can update after tier 3 lookup.

        # NOTE(Jiayi): Tier 3 lookup is batched together with get
        # in function `batched_get_non_blocking`.

        return num_hit_chunks

    async def _handle_peer_requests(self):
        """
        Handle `BatchedLookupAndGetMsg` issued by peers in `batched_get_non_blocking`.
        """

        logger.info(
            "Starting P2P backend batched get handler at %s", self.peer_lookup_url
        )
        self.async_context = get_zmq_context()
        self.async_peer_socket = get_zmq_socket_with_timeout(
            self.async_context,
            self.peer_lookup_url,
            "tcp",
            zmq.REP,
            "bind",
            self.socket_recv_timeout_ms,
            self.socket_send_timeout_ms,
        )

        while self.running.is_set():
            msg_bytes = await self.async_peer_socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=P2PMsg)

            num_tokens = len(msg.mem_indexes) * self.chunk_size
            monitor_req_id = self.stats_monitor.on_p2p_transfer_request(num_tokens)

            if isinstance(msg, BatchedLookupAndGetMsg):
                ret_msg = await self._handle_batched_lookup_and_get(msg)
            elif isinstance(msg, BatchedLookupAndPutMsg):
                ret_msg = await self._handle_batched_lookup_and_put(msg)
            else:
                logger.error("Unknown message type: %s", type(msg))
                ret_msg = P2PErrorMsg(error_code=P2PErrorCode.UNKNOWN_MSG_TYPE)

            logger.info(f"P2P transfer finished for request {monitor_req_id}")
            self.stats_monitor.on_p2p_transfer_finished(monitor_req_id)

            await self.async_peer_socket.send(msgspec.msgpack.encode(ret_msg))

    async def _handle_batched_lookup_and_get(
        self, msg: BatchedLookupAndGetMsg
    ) -> P2PMsgBase:
        lookup_id = msg.lookup_id
        mem_objs = None
        try:
            logger.info(
                "Received P2P batched lookup and get msg, lookup_id: %s", lookup_id
            )
            receiver_id = msg.receiver_id
            if not self.transfer_channel.remote_xfer_handler_exists(receiver_id):
                logger.error(
                    "Receiver %s does not exist in transfer channel",
                    receiver_id,
                )
                return P2PErrorMsg(
                    error_code=P2PErrorCode.REMOTE_XFER_HANDLER_NOT_INITIALIZED
                )

            remote_mem_indexes = msg.mem_indexes
            keys = [CacheEngineKey.from_string(key) for key in msg.keys]

            # TODO(Jiayi): Optimally, there's no need to use async call
            # for some backends (e.g., local cpu) as there's overhead for
            # async function call.
            num_hit_chunks = await self.local_cpu_backend.batched_async_contains(
                lookup_id=lookup_id,
                keys=keys,
                pin=True,
            )

            mem_objs = await self.local_cpu_backend.batched_get_non_blocking(
                lookup_id=lookup_id,
                keys=keys[:num_hit_chunks],
            )

            channel_transfer_spec = {
                "receiver_id": receiver_id,
                "remote_indexes": remote_mem_indexes[:num_hit_chunks],
            }
            await self.transfer_channel.async_batched_write(
                objects=mem_objs,
                transfer_spec=channel_transfer_spec,
            )

            return BatchedLookupAndGetRetMsg(num_hit_chunks=num_hit_chunks)
        except Exception as e:
            logger.error(
                "Error during P2P batched lookup and get operation "
                "for lookup_id %s: %s",
                lookup_id,
                e,
                exc_info=True,
            )
            return P2PErrorMsg(error_code=P2PErrorCode.P2P_SERVER_ERROR)
        finally:
            if mem_objs is not None:
                for mem_obj in mem_objs:
                    mem_obj.ref_count_down()
                    mem_obj.unpin()

    async def _handle_batched_lookup_and_put(
        self, msg: BatchedLookupAndPutMsg
    ) -> BatchedLookupAndPutRetMsg:
        try:
            logger.info("Received P2P batched lookup and put msg")
            sender_id = msg.sender_id
            r_mem_indexes = msg.mem_indexes
            keys = [CacheEngineKey.from_string(key) for key in msg.keys]
            offsets = msg.offsets

            # TODO(Jiayi): Need to support more backend
            r_mem_indexes_to_read = []
            keys_to_read = []
            local_mem_objs = []
            keys_len = len(keys)
            for idx, key in enumerate(keys):
                if self.local_cpu_backend.contains(key, pin=False):
                    continue
                r_mem_indexes_to_read.append(r_mem_indexes[idx])
                if not self.config.save_unfull_chunk or idx < keys_len - 1:
                    shapes = self.full_size_shapes
                else:
                    shapes = self._get_unfull_chunk_shapes(offsets[idx])
                local_mem_obj = self.local_cpu_backend.allocate(
                    shapes, self.dtypes, self.fmt
                )
                local_mem_objs.append(local_mem_obj)
                keys_to_read.append(key)

            channel_transfer_spec = {
                "sender_id": sender_id,
                "remote_indexes": r_mem_indexes_to_read,
            }
            await self.transfer_channel.async_batched_read(
                buffers=local_mem_objs,
                transfer_spec=channel_transfer_spec,
            )

            self.local_cpu_backend.batched_submit_put_task(
                keys=keys_to_read,
                memory_objs=local_mem_objs,
            )

            return BatchedLookupAndPutRetMsg(num_read_chunks=len(local_mem_objs))
        except Exception as e:
            logger.error(
                "Error during P2P batched lookup and put operation: %s",
                e,
                exc_info=True,
            )
            return BatchedLookupAndPutRetMsg(num_read_chunks=0)

    async def _ensure_peer_connection(
        self,
        target_peer_init_url: str,
        force_update: bool = False,
    ) -> None:
        if not force_update and target_peer_init_url in self.target_peer_info_mapping:
            return

        async with self.update_peer_lock:
            # double check
            if (
                not force_update
                and target_peer_init_url in self.target_peer_info_mapping
            ):
                return

            init_side_msg = P2PInitSideMsg()
            init_ret_msg = await self.transfer_channel.async_lazy_init_peer_connection(
                local_id=self.peer_init_url,
                peer_id=target_peer_init_url,
                peer_init_url=target_peer_init_url,
                init_side_msg=init_side_msg,
            )
            assert isinstance(init_ret_msg, P2PInitSideRetMsg)

            peer_lookup_url = init_ret_msg.peer_lookup_url
            peer_info = self.target_peer_info_mapping.get(target_peer_init_url, None)
            lookup_socket = get_zmq_socket_with_timeout(
                self.async_context,
                peer_lookup_url,
                "tcp",
                zmq.REQ,
                "connect",
                self.socket_recv_timeout_ms,
                self.socket_send_timeout_ms,
            )
            if peer_info is not None:
                peer_info.update_peer_lookup_url(peer_lookup_url)
                peer_info.update_lookup_socket(lookup_socket)
            else:
                self.target_peer_info_mapping[target_peer_init_url] = PeerInfo(
                    peer_init_url=target_peer_init_url,
                    peer_lookup_url=peer_lookup_url,
                    lookup_lock=asyncio.Lock(),
                    lookup_socket=lookup_socket,
                )

        logger.info(
            "Established connection to peer_init_url: %s, peer_lookup_url: %s",
            target_peer_init_url,
            peer_lookup_url,
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        target_peer_init_url, _ = self.lookup_id_to_peer_mapping.pop(lookup_id)

        assert isinstance(transfer_spec, dict)
        cum_chunk_lengths = transfer_spec.get("cum_chunk_lengths", None)
        assert cum_chunk_lengths is not None, "cum_chunk_lengths must be provided"
        assert isinstance(cum_chunk_lengths, list), "cum_chunk_lengths must be a list"

        mem_objs = []
        str_keys = []
        keys_len = len(keys)
        for idx, key in enumerate(keys):
            if not self.config.save_unfull_chunk or idx < keys_len - 1:
                shapes = self.full_size_shapes
            else:
                shapes = self._get_unfull_chunk_shapes(
                    cum_chunk_lengths[idx + 1] - cum_chunk_lengths[idx]
                )
            mem_obj = self.local_cpu_backend.allocate(shapes, self.dtypes, self.fmt)
            mem_objs.append(mem_obj)
            str_keys.append(key.to_string())

        local_indexes = self.transfer_channel.get_local_mem_indices(mem_objs)

        # NOTE(Jiayi): Tier 3 lookup is batched with retrieval.
        msg = BatchedLookupAndGetMsg(
            lookup_id=lookup_id,
            receiver_id=self.peer_init_url,
            keys=str_keys,
            mem_indexes=local_indexes,
        )

        retry_count = 0
        while retry_count < self.max_retry_count:
            peer_info = self.target_peer_info_mapping[target_peer_init_url]
            lookup_lock = peer_info.lookup_lock
            async with lookup_lock:
                lookup_socket = peer_info.lookup_socket
                try:
                    retry_count += 1
                    await lookup_socket.send(msgspec.msgpack.encode(msg))
                    ret_msg_bytes = await lookup_socket.recv()
                    ret_msg = msgspec.msgpack.decode(ret_msg_bytes, type=P2PMsg)
                    if (
                        isinstance(ret_msg, P2PErrorMsg)
                        and ret_msg.error_code
                        == P2PErrorCode.REMOTE_XFER_HANDLER_NOT_INITIALIZED
                    ):
                        logger.warning(
                            "Peer connection not initialized for lookup_id %s, "
                            "ensure peer connection first, retry count: %s",
                            lookup_id,
                            retry_count,
                        )
                        await self._ensure_peer_connection(target_peer_init_url, True)
                    else:
                        break
                except zmq.ZMQError as e:
                    logger.error(
                        "ZMQ error occurred for lookup_id %s. Error: %s",
                        lookup_id,
                        e,
                    )
                    await self._ensure_peer_connection(target_peer_init_url, True)
                    if retry_count == self.max_retry_count:
                        logger.error(
                            "Max retry count reached for lookup_id %s",
                            lookup_id,
                        )
                        self._cleanup_memory_objects(mem_objs)
                        return []
                except Exception as e:
                    logger.error(
                        "Error during P2P get operation for lookup_id %s: %s",
                        lookup_id,
                        e,
                        exc_info=True,
                    )
                    self._cleanup_memory_objects(mem_objs)
                    return []

        if isinstance(ret_msg, P2PErrorMsg):
            logger.error(
                "P2P error for lookup_id %s, error code: %s",
                lookup_id,
                ret_msg.error_code,
            )
            num_hit_chunks = 0
        else:
            num_hit_chunks = ret_msg.num_hit_chunks

        hit_mem_objs = mem_objs[:num_hit_chunks]
        for hit_mem_obj in hit_mem_objs:
            hit_mem_obj.pin()
        for missed_mem_obj in mem_objs[num_hit_chunks:]:
            missed_mem_obj.ref_count_down()
        return hit_mem_objs

    def _get_unfull_chunk_shapes(self, num_tokens: int) -> list[torch.Size]:
        shapes = []
        for shape in self.full_size_shapes:
            shape_list = list(shape)
            shape_list[self.fmt.token_dim()] = num_tokens
            shapes.append(torch.Size(shape_list))
        return shapes

    # NOTE: put-related functions are not supported for now.
    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        # TODO(baoloongmao): Add exception handling for socket operations
        # Code path for `move` operation in controller.
        assert isinstance(transfer_spec, dict)
        assert "target_peer_init_url" in transfer_spec
        assert "offsets" in transfer_spec

        target_peer_init_url = transfer_spec["target_peer_init_url"]
        offsets = transfer_spec["offsets"]

        await self._ensure_peer_connection(transfer_spec["target_peer_init_url"])

        str_keys = [key.to_string() for key in keys]
        local_indexes = self.transfer_channel.get_local_mem_indices(objs)

        msg = BatchedLookupAndPutMsg(
            sender_id=self.peer_init_url,
            keys=str_keys,
            offsets=offsets,
            mem_indexes=local_indexes,
        )

        peer_info = self.target_peer_info_mapping[target_peer_init_url]
        lookup_lock = peer_info.lookup_lock
        async with lookup_lock:
            lookup_socket = peer_info.lookup_socket
            await lookup_socket.send(msgspec.msgpack.encode(msg))
            ret_msg_bytes = await lookup_socket.recv()
        ret_msg = msgspec.msgpack.decode(ret_msg_bytes, type=P2PMsg)

        return ret_msg.num_read_chunks

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def _cleanup_memory_objects(self, mem_objs: list[MemoryObj]) -> None:
        """Safely cleanup memory objects by decrementing reference counts"""
        for mem_obj in mem_objs:
            try:
                mem_obj.ref_count_down()
            except Exception as e:
                logger.error("Error cleaning up memory object: %s", e)

    def close(
        self,
    ) -> None:
        """
        Close the P2P backend and cleanup resources.
        """
        logger.info("Closing P2P backend")
        self.running.clear()

        # Close all lookup sockets
        for peer_info in self.target_peer_info_mapping.values():
            try:
                peer_info.lookup_socket.close(linger=0)
            except Exception as e:
                logger.warning("Failed to close lookup socket: %s", e)
        self.target_peer_info_mapping.clear()

        # Close async peer socket
        if self.async_peer_socket is not None:
            try:
                self.async_peer_socket.close(linger=0)
            except Exception as e:
                logger.warning("Failed to close async peer socket: %s", e)
            self.async_peer_socket = None

        # Close transfer channel
        if hasattr(self, "transfer_channel") and self.transfer_channel is not None:
            try:
                self.transfer_channel.close()
            except Exception as e:
                logger.warning("Failed to close transfer channel: %s", e)

    ############################################################
    # Not-supported functions
    ############################################################

    # NOTE: synchronous contain is not supported for now.
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return False

    # NOTE: put-related functions are not supported for now.
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """P2P backend does not support put operations."""
        pass

    # NOTE: Synchronous get is not supported for now.
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        raise NotImplementedError

    # NOTE: pin is useless for P2P backend now.
    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    # NOTE: unpin is useless for P2P backend now.
    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    # NOTE: remove is useless for P2P backend now.
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return False
