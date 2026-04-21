# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.encoder_cache_transfer_buffer import (
    EncoderCacheTransferBuffer,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

MOONCAKE_IMPORT_ERROR_MSG = (
    "Please install mooncake by following the instructions at "
    "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
    "to run VLLM with MooncakeTransferEngine."
)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(MOONCAKE_IMPORT_ERROR_MSG)
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.request import Request

MMHash = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"


@dataclass(frozen=True)
class Key:
    mm_hash: MMHash
    req_id: ReqId


class MooncakeECAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    mm_hashes: list[tuple[MMHash, list[ReqId]]]
    remote_mm_addrs: list[int]
    remote_token_bytes: list[int]


@dataclass
class MMHashMeta:
    num_encoder_tokens: int
    mm_addr: int


@dataclass
class RecvMMHashMeta:
    mm_hash_meta: MMHashMeta
    remote_host: str
    remote_port: int


@dataclass
class FinishedSendMMHashSet:
    set: set[Key]
    lock: threading.Lock


@dataclass
class FinishedReceiveMMHashSet:
    set: set[MMHash]
    finish_recv_cond: asyncio.Condition


@dataclass
class FailedReceiveMMHashSet:
    """Track mm_hashes that failed to receive."""

    set: set[MMHash]
    lock: asyncio.Lock


class MooncakeECConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_hashes_to_recv: dict[Key, RecvMMHashMeta] = {}
        # mm_hashes whose encoder cache should be saved from
        # EncodeCacheManager to external storage on the producer side.
        self.mm_hashes_to_save: dict[MMHash, MMHashMeta] = {}

    def add_recv_req(
        self,
        req_id: ReqId,
        mm_hash: MMHash,
        mm_hash_meta: MMHashMeta,
        remote_host: str,
        remote_port: int,
    ):
        """Add a request to receive encoder cache from remote."""
        self.mm_hashes_to_recv[Key(mm_hash, req_id)] = RecvMMHashMeta(
            mm_hash_meta=mm_hash_meta,
            remote_host=remote_host,
            remote_port=remote_port,
        )

    def add_save_req(
        self,
        mm_hash: MMHash,
        mm_hash_meta: MMHashMeta,
    ) -> None:
        """Add a request to save encoder cache to external storage."""
        self.mm_hashes_to_save[mm_hash] = mm_hash_meta


class MooncakeECConnector(ECConnectorBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
    ):
        super().__init__(vllm_config, role)

        assert vllm_config.ec_transfer_config is not None

        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeECConnectorScheduler | None = (
                MooncakeECConnectorScheduler(vllm_config)
            )
            self.connector_worker: MooncakeECConnectorWorker | None = None
        elif role == ECConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeECConnectorWorker(vllm_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def has_cache_item(
        self,
        identifier: str,
        request: "Request | None" = None,
    ) -> bool:
        """Check if encoder cache exists in remote producer for a single mm item."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier, request)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        encoder_cache_manager=None,
    ) -> ECConnectorMetadata:
        """Build connector metadata for this step."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(
            scheduler_output, encoder_cache_manager
        )

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes, returns transfer params if needed."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_encoder_cache(self, transfer_buffer) -> None:
        """Register encoder cache - no-op as buffer is managed internally.

        This method exists for API compatibility with gpu_model_runner.
        The actual buffer is created and registered in
        MooncakeECConnectorWorker.__init__.
        """
        # Buffer is already initialized in connector_worker.__init__
        pass

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """Start loading encoder caches from remote via Mooncake."""
        assert self.connector_worker is not None
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeECConnectorMetadata)

        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def wait_for_load(self) -> set[str]:
        assert self.connector_worker is not None
        return self.connector_worker.wait_for_load()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Producer save encoder cache to transfer buffer."""
        assert self.connector_worker is not None
        self.connector_worker.save_caches(encoder_cache, mm_hash)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def maybe_update_remote_cache_state(self, encoder_cache, **kwargs) -> None:
        """
        Maybe update the remote cache state based on the local encoder cache.

        This method can be used to synchronize or update the state of the
        remote cache based on changes in the local encoder cache.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
        """
        assert self.connector_worker is not None
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeECConnectorMetadata)

        return self.connector_worker.maybe_update_remote_cache_state(
            encoder_cache, metadata
        )


class MooncakeECConnectorScheduler:
    """Runs in scheduler side process. Transfer params, metadata for recv/send"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)

        assert vllm_config.ec_transfer_config is not None
        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer
        self.is_consumer = vllm_config.ec_transfer_config.is_ec_consumer

        # Track mm_hashes that need to be loaded from remote
        self._mm_hashes_need_recv: dict[Key, tuple[Request, int]] = {}

    def has_cache_item(
        self,
        identifier: str,
        request: "Request | None",
    ) -> bool:
        """Check if encoder cache exists in remote producer for this mm item.

        Optimistic scheduling: if ec_transfer_params exist with do_remote_encode=True,
        we assume the cache can be transferred. If transfer fails, the failure
        handling mechanism will rollback and reschedule for local encoding.

        This approach aligns with KV transfer's optimistic scheduling strategy.
        """
        if not self.is_consumer:
            # Producer-only nodes don't load remote cache.
            return False

        try:
            ec_transfer_params = getattr(request, "ec_transfer_params", {})
            mm_hash_params = ec_transfer_params.get(identifier, {})

            # Check if we have valid transfer params and remote encoding is enabled
            has_remote_host = mm_hash_params.get("remote_host") is not None
            has_remote_port = mm_hash_params.get("remote_port") is not None
            do_remote_encode = mm_hash_params.get("do_remote_encode", False)

            return has_remote_host and has_remote_port and do_remote_encode

        except Exception as e:
            logger.error("[EC_SCHEDULER] Error checking EC transfer params: %s", e)
            return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        ec_transfer_params = getattr(request, "ec_transfer_params", None)
        if not ec_transfer_params:
            return

        mm_hash = request.mm_features[index].identifier

        # ec_transfer_params is now a dict keyed by mm_hash: {mm_hash: {...}}
        # Extract params for this specific mm_hash
        mm_hash_params = ec_transfer_params.get(mm_hash)
        if not mm_hash_params:
            return

        if mm_hash_params.get("do_remote_encode"):
            if all(p in mm_hash_params for p in ("remote_host", "remote_port")):
                num_encoder_tokens = request.get_num_encoder_embeds(index)
                self._mm_hashes_need_recv[Key(mm_hash, request.request_id)] = (
                    request,
                    num_encoder_tokens,
                )
            else:
                logger.warning(
                    "[EC_SCHEDULER] ✗ Invalid ECTransferParams for mm_hash %s: %s. "
                    "This request will not utilize EC transfer",
                    mm_hash,
                    mm_hash_params,
                )

            # Only trigger 1 EC transfer per mm_hash
            mm_hash_params["do_remote_encode"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
        encoder_cache_manager=None,
    ) -> ECConnectorMetadata:
        meta = MooncakeECConnectorMetadata()

        # Convert mm_hashes to metadata
        for key, (request, num_encoder_tokens) in self._mm_hashes_need_recv.items():
            mm_hash = key.mm_hash
            ec_transfer_params = getattr(request, "ec_transfer_params", None)
            if ec_transfer_params:
                mm_hash_params = ec_transfer_params.get(mm_hash)
                if mm_hash_params:
                    meta.add_recv_req(
                        req_id=request.request_id,
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_encoder_tokens,
                            mm_addr=0,
                        ),
                        remote_host=mm_hash_params["remote_host"],
                        remote_port=mm_hash_params["remote_port"],
                    )

        # Clear the lists once workers start the transfers
        self._mm_hashes_need_recv.clear()

        # 2. Save any EncoderCacheManager-cached items to external storage.
        # Only producer needs to save.
        if self.is_producer and encoder_cache_manager is not None:
            scheduled_mm_hashes = self._collect_scheduled_mm_hashes(scheduler_output)

            for mm_hash, num_token in scheduled_mm_hashes.items():
                has_hbm = encoder_cache_manager.has_cache(mm_hash)

                if has_hbm:
                    meta.add_save_req(
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_token,
                            mm_addr=0,
                        ),
                    )

        return meta

    def request_finished(
        self,
        request: "Request",
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if not self.is_producer:
            # Consumer doesn't return params
            logger.debug("[EC_SCHEDULER] Consumer node, no params to return")
            return False, None

        # Build params for all mm_hashes in this request
        result_params: dict[MMHash, dict[str, Any]] = {}
        for idx, feature in enumerate(request.mm_features):
            mm_hash = feature.identifier
            # Return params keyed by mm_hash for proxy aggregation
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
            }
        return len(result_params) > 0, result_params if result_params else None

    def _collect_scheduled_mm_hashes(
        self, scheduler_output: SchedulerOutput
    ) -> dict[str, int]:
        """
        Collect all mm_hashes from scheduled requests.

        Args:
            scheduler_output: The scheduler output containing scheduled requests

        Returns:
            dict: mm_hash -> num_encoder_tokens mapping
        """
        mm_hashes = {}

        # Collect from scheduled_new_reqs
        for req in scheduler_output.scheduled_new_reqs:
            if hasattr(req, "mm_features") and req.mm_features:
                for feature in req.mm_features:
                    mm_hash = feature.identifier
                    num_tokens = feature.mm_position.get_num_embeds
                    mm_hashes[mm_hash] = num_tokens

        return mm_hashes


class MooncakeECConnectorWorker:
    """Runs in worker side process. Handle actual send/receive with Mooncake"""

    # Default buffer size: 1GB
    DEFAULT_BUFFER_SIZE = 1073741824

    def __init__(self, vllm_config: VllmConfig):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")

        self.vllm_config = vllm_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        self.engine = TransferEngine()
        self.hostname = get_ip()
        assert vllm_config.ec_transfer_config is not None
        device_name = vllm_config.ec_transfer_config.ec_connector_extra_config.get(
            "device_name"
        )
        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", "rdma", device_name
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "[EC_WORKER] Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        # Mooncake handshake port.
        self.side_channel_port: int = get_mooncake_side_channel_port(vllm_config)

        # Encoder cache registration
        self.dtype = (
            vllm_config.model_config.dtype
            if isinstance(vllm_config.model_config.dtype, torch.dtype)
            else getattr(torch, vllm_config.model_config.dtype)
        )
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        self.embed_size = vllm_config.model_config.get_inputs_embeds_size()
        self.byte_per_token = self.embed_size * dtype_size
        self.device_type = current_platform.device_type

        # Transfer buffer - initialized immediately
        self._buffer_size = int(
            vllm_config.ec_transfer_config.ec_connector_extra_config.get(
                "transfer_buffer_size", self.DEFAULT_BUFFER_SIZE
            )
        )
        self.transfer_buffer: EncoderCacheTransferBuffer = EncoderCacheTransferBuffer(
            buffer_size=self._buffer_size,
            device="cpu",
        )

        # stored addr of mm tensor in registered caches (external tensor pool)
        self.local_mm_addrs: dict[MMHash, int] = {}
        # reverse map for pool eviction callback: addr -> mm_hash
        self._addr_to_mm_hash: dict[int, MMHash] = {}
        # Lock protecting both maps above; used by save/send/evict paths.
        self._mm_lock = threading.Lock()

        # Keep local_mm_addrs in sync when the buffer evicts
        self.transfer_buffer.on_free = self._on_pool_free

        # Register buffer with Mooncake for transfer
        ret_value = self.engine.register_memory(
            self.transfer_buffer.base_address, self.transfer_buffer.buffer_size
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        logger.info(
            "[EC_WORKER] Initialized and registered EC transfer buffer: size=%d bytes, "
            "base_addr=0x%x",
            self._buffer_size,
            self.transfer_buffer.base_address,
        )

        self.num_workers = int(
            vllm_config.ec_transfer_config.ec_connector_extra_config.get(
                "num_workers", 10
            )
        )
        self.mm_hashes_need_recv: set[MMHash] = set()

        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer
        self.is_consumer = vllm_config.ec_transfer_config.is_ec_consumer
        self._mooncake_sender_t: threading.Thread | None = None
        self._mooncake_receiver_t: threading.Thread | None = None

        if self.is_producer:
            # Background thread for sending cache to P.
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="vllm-mooncake-ec-sender",
            )
            logger.debug("[EC_PRODUCER] Use %d workers for transfer", self.num_workers)

        if self.is_consumer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._receiver_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("[EC_CONSUMER] Start receiver thread")

        self.finished_sending_mm_hashes: FinishedSendMMHashSet = FinishedSendMMHashSet(
            set(), threading.Lock()
        )
        self.finished_recving_mm_hashes: FinishedReceiveMMHashSet = (
            FinishedReceiveMMHashSet(set(), asyncio.Condition())
        )
        self.failed_recving_mm_hashes: FailedReceiveMMHashSet = FailedReceiveMMHashSet(
            set(), asyncio.Lock()
        )

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeECAgentMetadata)

        # Launch sender thread for producer node
        if self.is_producer:
            ready_event = threading.Event()
            self._mooncake_sender_t = threading.Thread(
                target=self._mooncake_sender,
                args=(ready_event, self.side_channel_port, self.tp_rank),
                daemon=True,
                name="ec_mooncake_sender",
            )
            self._mooncake_sender_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        if hasattr(self, "zmq_ctx"):
            self.zmq_ctx.term()
        if hasattr(self, "async_zmq_ctx"):
            self.async_zmq_ctx.term()
        if getattr(self, "is_producer", False) and hasattr(self, "_sender_executor"):
            self._sender_executor.shutdown(wait=False)
            mooncake_sender_t = getattr(self, "_mooncake_sender_t", None)
            if mooncake_sender_t:
                mooncake_sender_t.join()
        receiver_loop = getattr(self, "receiver_loop", None)
        if (
            getattr(self, "is_consumer", False)
            and receiver_loop is not None
            and receiver_loop.is_running()
        ):
            receiver_loop.call_soon_threadsafe(receiver_loop.stop)
            if self._mooncake_receiver_t:
                self._mooncake_receiver_t.join()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        addr = self.transfer_buffer.store_tensor(encoder_cache[mm_hash])

        # Update bookkeeping for this mm_hash. We intentionally do this
        # after store_tensor returns to avoid deadlock if the pool evicts
        # internally and calls back into _on_pool_free.
        with self._mm_lock:
            old_addr = self.local_mm_addrs.get(mm_hash)
            if old_addr is not None:
                self._addr_to_mm_hash.pop(old_addr, None)
            self.local_mm_addrs[mm_hash] = addr
            self._addr_to_mm_hash[addr] = mm_hash
            logger.debug(
                "[EC_PRODUCER] Updated bookkeeping: local_mm_addrs[%s]=0x%x, "
                "total_cached=%d",
                mm_hash,
                addr,
                len(self.local_mm_addrs),
            )

    def _on_pool_free(self, addr: int) -> None:
        """Called by the tensor pool when a block is freed (evict or explicit free)."""
        with self._mm_lock:
            mm_hash = self._addr_to_mm_hash.pop(addr, None)
            if mm_hash is not None:
                if self.local_mm_addrs.get(mm_hash) == addr:
                    self.local_mm_addrs.pop(mm_hash, None)
            else:
                logger.debug(
                    "[EC_WORKER] Pool freed unknown addr=0x%x (not in tracking)", addr
                )

    def _receiver_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _mooncake_sender(
        self, ready_event: threading.Event, base_port: int, tp_rank: int
    ):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)
        logger.debug("Mooncake sender starting listening on path: %s", frontend_path)

        backend_path = make_zmq_path("inproc", str(uuid.uuid4()))
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, msg_bytes = frontend.recv_multipart()
                    # Handle transfer request asynchronously
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        msg_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.exception("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.exception(
                "Error in Mooncake sender thread: %s. Exiting thread.", str(e)
            )
        finally:
            frontend.close()
            backend.close()

    def _sender_worker(
        self, identity: bytes, metadata_bytes: bytes, worker_channel_path: str
    ):
        status = TRANS_ERROR

        try:
            metadata = self._decoder.decode(metadata_bytes)
            self.send_ec_cache(metadata)
            status = TRANS_DONE
        except Exception as e:
            logger.exception("Error from Mooncake ECConnector: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.exception(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_ec_cache(self, meta: MooncakeECAgentMetadata):
        send_mm_hashes = [mm_hash for (mm_hash, _) in meta.mm_hashes]
        for mm_hash, req_ids in meta.mm_hashes:
            logger.debug(
                "[EC_PRODUCER] Will send mm_hash=%s for req_ids=%s",
                mm_hash,
                [rid for rid in req_ids],
            )

        self._send_caches(send_mm_hashes, meta)

        with self.finished_sending_mm_hashes.lock:
            keys: list[Key] = []
            for mm_hash, req_ids in meta.mm_hashes:
                keys.extend([Key(mm_hash, req_id) for req_id in req_ids])
            self.finished_sending_mm_hashes.set.update(keys)

    def _send_caches(
        self,
        send_mm_hashes: list[MMHash],
        agent_meta: MooncakeECAgentMetadata,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        remote_mm_addrs = agent_meta.remote_mm_addrs
        remote_token_bytes = agent_meta.remote_token_bytes
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for mm_hash, remote_token_byte, remote_mm_addr in zip(
            send_mm_hashes, remote_token_bytes, remote_mm_addrs
        ):
            if remote_token_byte == 0:
                logger.warning(
                    "[EC_PRODUCER] Skip mm_hash=%s (remote_token_byte=0)",
                    mm_hash,
                )
                continue

            with self._mm_lock:
                addr = self.local_mm_addrs.get(mm_hash)
            if addr is None:
                raise RuntimeError(
                    f"[EC_PRODUCER] ✗ No buffer entry for mm_hash={mm_hash}: "
                    "Failing transfer."
                )
                continue

            src_ptrs.append(addr)
            dst_ptrs.append(remote_mm_addr)
            lengths.append(remote_token_byte)

        if not src_ptrs:
            logger.warning("[EC_PRODUCER] No valid transfers in batch, skipping")
            return

        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )

        if ret_value != 0:
            logger.error(
                "[EC_PRODUCER] ✗ batch_transfer_sync_write FAILED: ret=%d",
                ret_value,
            )
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

    async def fetch_finished_recving_mm_hashes(self) -> tuple[set[MMHash], set[MMHash]]:
        """Fetch finished and failed receiving mm_hashes.

        Returns:
            Tuple of (finished_mm_hashes, failed_mm_hashes)
        """
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            finished_recving_mm_hashes = self.finished_recving_mm_hashes.set
            self.finished_recving_mm_hashes.set = set()

        async with self.failed_recving_mm_hashes.lock:
            failed_recving_mm_hashes = self.failed_recving_mm_hashes.set
            self.failed_recving_mm_hashes.set = set()

        logger.debug(
            "[EC_CONSUMER] finished_recving_mm_hashes=%s, failed_recving_mm_hashes=%s",
            finished_recving_mm_hashes,
            failed_recving_mm_hashes,
        )
        return finished_recving_mm_hashes, failed_recving_mm_hashes

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.

        Returns:
            Tuple of (finished_sending, finished_recving, failed_recving)
        """
        fut = None
        if self.is_consumer:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_mm_hashes(), self.receiver_loop
            )

        if self.is_producer:
            with self.finished_sending_mm_hashes.lock:
                finished_sending_mm_hashes = set(
                    [key.mm_hash for key in self.finished_sending_mm_hashes.set]
                )
                self.finished_sending_mm_hashes.set = set()
        else:
            finished_sending_mm_hashes = set()

        if fut:
            finished_recving_mm_hashes, failed_recving_mm_hashes = fut.result()
        else:
            finished_recving_mm_hashes = set()
            failed_recving_mm_hashes = set()

        if (
            finished_sending_mm_hashes
            or finished_recving_mm_hashes
            or failed_recving_mm_hashes
        ):
            logger.debug(
                "[EC_WORKER] Rank %s, get_finished: %s items done sending, "
                "%s items done recving, %s items failed recving",
                self.tp_rank,
                len(finished_sending_mm_hashes),
                len(finished_recving_mm_hashes),
                len(failed_recving_mm_hashes),
            )
        return (
            finished_sending_mm_hashes or None,
            finished_recving_mm_hashes or None,
            failed_recving_mm_hashes or None,
        )

    async def receive_ec(
        self,
        path: str,
        mm_hash_items: list[tuple[tuple[MMHash, list[ReqId]], MMHashMeta]],
        encoder_cache: dict[str, torch.Tensor],
    ):
        mm_hashes, mm_hashes_meta = map(list, zip(*mm_hash_items))
        mm_hash_list = [mm_hash for (mm_hash, _) in mm_hashes]

        metadata = MooncakeECAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            mm_hashes=mm_hashes,
            remote_mm_addrs=[meta.mm_addr for meta in mm_hashes_meta],
            remote_token_bytes=[
                meta.num_encoder_tokens * self.byte_per_token for meta in mm_hashes_meta
            ],
        )

        encoded_data = self._encoder.encode(metadata)

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

        transfer_failed = True
        try:
            await sock.send(encoded_data)

            ret_msg = await sock.recv()

            if ret_msg == TRANS_DONE:
                transfer_failed = False
            else:
                logger.error(
                    "[EC_CONSUMER] Transfer FAILED: got %s instead "
                    "of TRANS_DONE for mm_hashes=%s",
                    ret_msg,
                    [h for h in mm_hash_list],
                )
        except zmq.Again:
            logger.exception(
                "[EC_CONSUMER] Transfer TIMEOUT after 1s for mm_hashes=%s",
                [h for h in mm_hash_list],
            )
        except zmq.ContextTerminated:
            logger.exception(
                "[EC_CONSUMER] ZMQ context terminated, exiting receiver thread."
            )
        except Exception as e:
            logger.exception(
                "[EC_CONSUMER] ✗ Transfer request failed for mm_hashes=%s: %s",
                [h for h in mm_hash_list],
                e,
            )
        finally:
            sock.close()

        if transfer_failed:
            # Free the receive buffer slots allocated in group_ec_pull().
            # The remote side never wrote valid data into them (or the
            # transfer was incomplete), so release the space immediately
            # rather than waiting for LRU eviction.
            for meta in mm_hashes_meta:
                try:
                    self.transfer_buffer.free(meta.mm_addr)
                except Exception as e:
                    logger.warning(
                        "[EC_CONSUMER] Unable to free buffer space at %d. %s",
                        meta.mm_addr,
                        e,
                    )

            # Mark these mm_hashes as failed
            async with self.failed_recving_mm_hashes.lock:
                self.failed_recving_mm_hashes.set.update(mm_hash_list)
            logger.warning(
                "[EC_CONSUMER] Marked %d mm_hashes as failed: %s",
                len(mm_hash_list),
                [h for h in mm_hash_list],
            )
            async with self.finished_recving_mm_hashes.finish_recv_cond:
                if self._all_mm_hashes_resolved():
                    self.finished_recving_mm_hashes.finish_recv_cond.notify_all()
            return

        # Load tensors from received buffer
        try:
            for (mm_hash, _), addr, num_bytes in zip(
                metadata.mm_hashes,
                metadata.remote_mm_addrs,
                metadata.remote_token_bytes,
            ):
                logger.debug(
                    "[EC_CONSUMER] Loading mm_hash=%s from addr=0x%x, size=%d bytes",
                    mm_hash,
                    addr,
                    num_bytes,
                )

                encoder_cache[mm_hash] = self.transfer_buffer.load_tensor(
                    addr,
                    self.dtype,
                    (num_bytes // self.byte_per_token, self.embed_size),
                    device=self.device_type,
                    copy=True,
                )
        except Exception as e:
            logger.exception(
                "[EC_CONSUMER] ✗ Failed to load tensors for mm_hashes=%s: %s",
                [h for h in mm_hash_list],
                e,
            )
            # Free all receive buffer slots. Some entries may have been
            # partially loaded (written to encoder_cache) before the exception;
            # their GPU entries will be cleaned up via freed/free_encoder_mm_hashes.
            for addr in metadata.remote_mm_addrs:
                try:
                    self.transfer_buffer.free(addr)
                except Exception as e:
                    logger.warning(
                        "[EC_CONSUMER] Unable to free buffer space at %d. %s",
                        addr,
                        e,
                    )

            # Mark as failed
            async with self.failed_recving_mm_hashes.lock:
                self.failed_recving_mm_hashes.set.update(mm_hash_list)
            async with self.finished_recving_mm_hashes.finish_recv_cond:
                if self._all_mm_hashes_resolved():
                    self.finished_recving_mm_hashes.finish_recv_cond.notify_all()
            return

        async with self.finished_recving_mm_hashes.finish_recv_cond:
            self.finished_recving_mm_hashes.set.update(mm_hash_list)
            if self._all_mm_hashes_resolved():
                self.finished_recving_mm_hashes.finish_recv_cond.notify_all()

    def group_ec_pull(self, metadata: MooncakeECConnectorMetadata):
        ec_pulls: dict[str, dict[MMHash, tuple[list[ReqId], MMHashMeta]]] = defaultdict(
            dict
        )
        for key, meta in metadata.mm_hashes_to_recv.items():
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            mm_hashes_meta = ec_pulls[path]

            if key.mm_hash not in mm_hashes_meta:
                # Allocate receive buffer
                alloc_size = meta.mm_hash_meta.num_encoder_tokens * self.byte_per_token
                meta.mm_hash_meta.mm_addr = self.transfer_buffer.allocate(alloc_size)

                mm_hashes_meta[key.mm_hash] = ([key.req_id], meta.mm_hash_meta)
            else:
                req_ids, _ = mm_hashes_meta[key.mm_hash]
                req_ids.append(key.req_id)
                logger.debug(
                    "[EC_CONSUMER] mm_hash=%s already in group, appending req_id=%s",
                    key.mm_hash,
                    key.req_id,
                )

        return ec_pulls

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        metadata: MooncakeECConnectorMetadata,
    ):
        self.mm_hashes_need_recv = set(
            [key.mm_hash for key in metadata.mm_hashes_to_recv]
        )

        ec_pulls = self.group_ec_pull(metadata)
        for path, mm_hashes_meta in ec_pulls.items():
            mm_hash_items = [
                ((mm_hash, req_ids), meta)
                for mm_hash, (req_ids, meta) in mm_hashes_meta.items()
            ]
            logger.debug(
                "[EC_CONSUMER] start_load_caches for mm_hash_items %s",
                mm_hash_items,
            )
            asyncio.run_coroutine_threadsafe(
                self.receive_ec(path, mm_hash_items, encoder_cache), self.receiver_loop
            )

    def _all_mm_hashes_resolved(self) -> bool:
        """Return True when every hash in mm_hashes_need_recv has reached a
        terminal state — either successfully received or failed."""
        resolved = (
            self.finished_recving_mm_hashes.set | self.failed_recving_mm_hashes.set
        ) == self.mm_hashes_need_recv
        return resolved

    async def _wait_for_load(self) -> None:
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            await self.finished_recving_mm_hashes.finish_recv_cond.wait_for(
                self._all_mm_hashes_resolved
            )

    def wait_for_load(self) -> set[str]:
        fut = asyncio.run_coroutine_threadsafe(
            self._wait_for_load(), self.receiver_loop
        )
        fut.result()  # Block until complete
        # Return a snapshot of failed hashes
        # get_finished() will clear the live set later.
        return set(self.failed_recving_mm_hashes.set)

    def has_cache_in_buffer(
        self,
        identifier: str,
    ) -> bool:
        """Worker to check if encoder cache exists in its own buffer
        for a single mm item.
        """
        with self._mm_lock:
            return identifier in self.local_mm_addrs

    def maybe_update_remote_cache_state(
        self, encoder_cache, metadata: MooncakeECConnectorMetadata, **kwargs
    ) -> None:
        for mm_hash in metadata.mm_hashes_to_save:
            # make sure is producer, and mm_hash exists in local
            # EncodeCacheManager encoder cache
            if (not self.is_producer) or (mm_hash not in encoder_cache):
                continue

            # Check if transfer buffer doesn't have it but HBM does
            if not self.has_cache_in_buffer(mm_hash):
                logger.debug(
                    "update_remote_cache_state for hash %s",
                    mm_hash,
                )
                self.save_caches(
                    encoder_cache=encoder_cache,
                    mm_hash=mm_hash,
                )


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_EC_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
