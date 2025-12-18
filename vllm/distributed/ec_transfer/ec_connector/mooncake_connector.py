# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
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
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.request import Request

EngineId = str
MMHash = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)


class MooncakeECAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    mm_hashes: list[MMHash]
    enc_base_addrs: list[int]
    enc_token_bytes: list[int]


@dataclass
class MMHashMeta:
    num_encoder_tokens: int
    mm_base_addr: int


@dataclass
class RecvMMHashMeta:
    mm_hash: MMHash
    mm_hash_meta: MMHashMeta
    remote_host: str
    remote_port: int


@dataclass
class SendMMHashMeta:
    mm_hash_meta: MMHashMeta
    ready: threading.Event
    expire_time: float = float("inf")


@dataclass
class SendMeta:
    mm_hashes: dict[MMHash, SendMMHashMeta]
    lock: threading.Lock


@dataclass
class FinishedSendMMHashSet:
    set: set[MMHash]
    lock: threading.Lock


@dataclass
class FinishedReceiveMMHashSet:
    set: set[MMHash]
    lock: asyncio.Lock


class MooncakeECConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_hashes_to_recv: dict[MMHash, RecvMMHashMeta] = {}
        self.mm_hashes_to_send: dict[MMHash, MMHashMeta] = {}

    def add_recv_req(
        self,
        mm_hash: MMHash,
        mm_hash_meta: MMHashMeta,
        remote_host: str,
        remote_port: int,
    ):
        """Add a request to receive encoder cache from remote."""
        self.mm_hashes_to_recv[mm_hash] = RecvMMHashMeta(
            mm_hash=mm_hash,
            mm_hash_meta=mm_hash_meta,
            remote_host=remote_host,
            remote_port=remote_port,
        )


class MooncakeConnector(ECConnectorBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: ECConnectorRole,
    ):
        super().__init__(vllm_config, role)

        assert vllm_config.ec_transfer_config is not None
        assert vllm_config.ec_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.ec_transfer_config.engine_id

        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeECConnectorScheduler | None = (
                MooncakeECConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeECConnectorWorker | None = None
        elif role == ECConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeECConnectorWorker(
                vllm_config, self.engine_id
            )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def has_caches(self, request: "Request") -> list[bool]:
        """Check if encoder cache exists remotely for each mm_data."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_caches(request)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Update state after encoder cache allocation."""
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def update_mm_hash_addrs_from_output(
        self, ec_connector_output: ECConnectorOutput
    ) -> None:
        assert self.connector_scheduler is not None
        assert ec_connector_output is not None
        logger.debug("hero: update_mm_hash_addrs_from_output in connector!!!!")
        self.connector_scheduler.update_mm_hash_addrs_from_output(ec_connector_output)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Build connector metadata for this step."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes, returns transfer params if needed."""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_encoder_cache(
        self,
        ec_cache: TensorMemoryPool,
    ):
        """Register encoder cache tensors with Mooncake."""
        assert self.connector_worker is not None
        # For NIXL, we register the main encoder cache tensor
        # Individual mm_hash caches are handled via recv tensors
        if (
            hasattr(self.connector_worker, "encoder_cache")
            and self.connector_worker.encoder_cache is not None
        ):
            # Already registered
            return
        # The encoder_cache will be registered when it's first set
        # via register_encoder_cache method
        self.connector_worker.register_encoder_cache(ec_cache)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """Start loading encoder caches from remote via Mooncake."""
        assert self.connector_worker is not None
        metadata: MooncakeECConnectorMetadata = self._get_connector_metadata()

        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache to remote (handled by request_finished)."""
        assert self.connector_worker is not None
        logger.debug("hero: save_caches!!!!")
        if mm_hash in encoder_cache:
            logger.debug(f"hero: base_addr: {encoder_cache[mm_hash].data_ptr()}")
            self.connector_worker._ENCODER_MM_BASE_ADDRS[mm_hash] = encoder_cache[
                mm_hash
            ].data_ptr()
            logger.debug(
                f"hero: self.connector_worker._ENCODER_MM_BASE_ADDRS: {self.connector_worker._ENCODER_MM_BASE_ADDRS}"
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_mm_hash_addrs(self):
        """Get dict of addresses of encoder cache tensor by mm hash"""
        assert self.connector_worker is not None
        return self.connector_worker._ENCODER_MM_BASE_ADDRS.copy()


class MooncakeECConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)
        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        # Track mm_hashes that need to be loaded from remote
        # mm_hash -> (request, num_encoder_tokens)
        self._mm_hashes_need_recv: dict[MMHash, tuple[Request, int]] = {}
        # Track mm_hashes that need to be sent (for producer role)
        self._mm_hashes_need_send: dict[MMHash, tuple[int, int]] = {}

        # TODO: find a more elegant way to store & manage mm_base_addr
        self._ENCODER_MM_BASE_ADDRS: dict[EngineId, dict[MMHash, int]] = {}

    def has_caches(self, request: "Request") -> list[bool]:
        """Check if encoder cache exists remotely for each mm_data."""
        result = []

        ec_transfer_params = getattr(request, "ec_transfer_params", None)

        for feature in request.mm_features:
            mm_hash = feature.identifier

            if self.is_producer:
                has_cache = False
            else:
                mm_hash_params = (
                    ec_transfer_params.get(mm_hash) if ec_transfer_params else None
                )
                has_cache = bool(
                    mm_hash_params
                    and mm_hash_params.get("num_encoder_tokens", 0) > 0
                    and all(p in mm_hash_params for p in ("remote_host", "remote_port"))
                )
            result.append(has_cache)

        logger.debug(f"has_caches results: {result}")
        return result

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
            logger.debug(
                "No ec_transfer_params found for mm_hash %s in request %s",
                mm_hash,
                request.request_id,
            )
            return

        if mm_hash_params.get("do_remote_encode"):
            if all(p in mm_hash_params for p in ("remote_host", "remote_port")):
                # Get num_encoder_tokens from the request
                num_encoder_tokens = request.get_num_encoder_tokens(index)
                self._mm_hashes_need_recv[mm_hash] = (
                    request,
                    num_encoder_tokens,
                )
                logger.debug(
                    "Added mm_hash %s to recv queue with num_encoder_tokens: %d",
                    mm_hash,
                    num_encoder_tokens,
                )
            else:
                logger.warning(
                    "Got invalid ECTransferParams for mm_hash %s: %s. This "
                    "request will not utilize EC transfer",
                    mm_hash,
                    mm_hash_params,
                )

            # Only trigger 1 EC transfer per mm_hash
            mm_hash_params["do_remote_encode"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        meta = MooncakeECConnectorMetadata()

        # Convert mm_hashes to metadata
        for mm_hash, (request, num_encoder_tokens) in self._mm_hashes_need_recv.items():
            ec_transfer_params = getattr(request, "ec_transfer_params", None)
            if ec_transfer_params:
                # Extract params for this specific mm_hash
                mm_hash_params = ec_transfer_params.get(mm_hash)
                logger.debug(f"hero: mm_hash_params for {mm_hash}: {mm_hash_params}")
                if mm_hash_params:
                    meta.add_recv_req(
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_encoder_tokens,
                            mm_base_addr=mm_hash_params["mm_base_addr"],
                        ),
                        remote_host=mm_hash_params["remote_host"],
                        remote_port=mm_hash_params["remote_port"],
                    )
                else:
                    logger.warning(
                        "No ec_transfer_params found for mm_hash %s in request %s",
                        mm_hash,
                        request.request_id,
                    )

        meta.mm_hashes_to_send = self._mm_hashes_need_send

        # Clear the lists once workers start the transfers
        self._mm_hashes_need_recv.clear()
        self._mm_hashes_need_send.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if not self.is_producer:
            # Consumer doesn't return params
            return False, None

        # Build params for all mm_hashes in this request
        result_params: dict[MMHash, dict[str, Any]] = {}
        for idx, feature in enumerate(request.mm_features):
            mm_hash = feature.identifier
            num_encoder_tokens = request.get_num_encoder_tokens(idx)

            # Mark mm_hash to be sent asynchronously
            mm_base_addr = self._ENCODER_MM_BASE_ADDRS.get(mm_hash)
            self._mm_hashes_need_send[mm_hash] = MMHashMeta(
                num_encoder_tokens, mm_base_addr
            )
            logger.debug(f"hero: mm_base_addr is {mm_base_addr} for mm_hash {mm_hash}")
            logger.debug(
                f"hero: self._ENCODER_MM_BASE_ADDRS {self._ENCODER_MM_BASE_ADDRS}"
            )

            # Return params keyed by mm_hash for proxy aggregation
            # Format: {mm_hash: {do_remote_encode, num_encoder_tokens, remote_engine_id, ...}}
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "num_encoder_tokens": num_encoder_tokens,
                "mm_base_addr": mm_base_addr,
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
            }

        return len(result_params) > 0, result_params if result_params else None


class MooncakeECConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", "rdma", "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        # Mooncake handshake port.
        self.side_channel_port: int = get_mooncake_side_channel_port(vllm_config)

        # Encoder cache registration
        dtype_size = torch.tensor(
            [], dtype=vllm_config.model_config.dtype
        ).element_size()
        self.byte_per_token = (
            vllm_config.model_config.get_inputs_embeds_size() * dtype_size
        )
        # TODO: find a more elegant way to store & manage mm_base_addr
        self._ENCODER_MM_BASE_ADDRS: dict[MMHash, int] = {}

        self.num_workers = vllm_config.ec_transfer_config.ec_connector_extra_config.get(
            "num_workers", 10
        )
        self.mm_hashes_need_send: SendMeta = SendMeta(
            mm_hashes={}, lock=threading.Lock()
        )

        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

        if self.is_producer:
            # Background thread for sending cache to P.
            self._mooncake_sender_t: threading.Thread | None = None
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="vllm-mooncake-ec-sender",
            )
            logger.debug(
                "Mooncake Encoder: use %d workers to send eccaches", self.num_workers
            )
        else:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._receiver_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Prefiller: start receiver thread")

        self.finished_sending_mm_hashes: FinishedSendMMHashSet = FinishedSendMMHashSet(
            set(), threading.Lock()
        )
        self.finished_recving_mm_hashes: FinishedReceiveMMHashSet = (
            FinishedReceiveMMHashSet(set(), asyncio.Lock())
        )

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeECAgentMetadata)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if not self.is_producer:
            self._sender_executor.shutdown(wait=False)
            if self._mooncake_sender_t:
                self._mooncake_sender_t.join()
        elif self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

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
                    identity, _, metadata_bytes = frontend.recv_multipart()
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        metadata_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
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
            logger.error("Error processing Mooncake handshake: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_ec_cache(self, meta: MooncakeECAgentMetadata):
        send_mm_hashes: list[tuple[MMHash, SendMMHashMeta]] = []
        with self.mm_hashes_need_send.lock:
            for mm_hash in meta.mm_hashes:
                send_meta = self.mm_hashes_need_send.mm_hashes.get(mm_hash)
                if send_meta is None:
                    logger.warning(
                        "Request %s not found in mm_hashes_need_send", req_id
                    )
                    return
                # Mark it as not expired. We will send it now.
                send_meta.expire_time = float("inf")
                send_mm_hashes.append((mm_hash, send_meta))

        self._send_caches(send_mm_hashes, meta)

        with self.mm_hashes_need_send.lock:
            for req_id in meta.mm_hashes:
                del self.mm_hashes_need_send.mm_hashes[req_id]

        with self.finished_sending_mm_hashes.lock:
            self.finished_sending_mm_hashes.set.update(meta.mm_hashes)

    def _send_caches(
        self,
        send_mm_hashes: list[tuple[MMHash, SendMMHashMeta]],
        agent_meta: MooncakeECAgentMetadata,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        local_base_addrs = self._ENCODER_MM_BASE_ADDRS
        remote_base_addrs = agent_meta.enc_base_addrs
        remote_token_bytes = agent_meta.enc_token_bytes
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        assert len(send_mm_hashes) == len(remote_token_bytes)
        for (mm_hash, send_meta), remote_token_byte, remote_base_addr in zip(
            send_mm_hashes, remote_token_bytes, remote_base_addrs
        ):
            send_meta.ready.wait()

            if remote_token_byte == 0:
                continue

            src_ptrs.append(local_base_addrs[mm_hash])
            dst_ptrs.append(remote_base_addr)
            lengths.append(remote_token_byte)

            logger.debug(
                "Sending ec_caches for mm_hash %s (%d bytes) to %s",
                mm_hash,
                remote_token_byte,
                remote_session,
            )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        logger.debug(
            "Sending to %s done, took %s",
            remote_session,
            time.perf_counter() - start_time,
        )

    def register_encoder_cache(self, ec_caches: TensorMemoryPool):
        """Register the EC Cache data in mooncake."""
        ret_value = self.engine.register_memory(
            ec_caches.base_address, ec_caches.max_block_size
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        logger.debug("registered tensor pool with size=%d", ec_caches.max_block_size)

        # No need to launch server for consumer node.
        if not self.is_producer:
            return

        ready_event = threading.Event()
        self._mooncake_sender_t = threading.Thread(
            target=self._mooncake_sender,
            args=(ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="ec_mooncake_sender",
        )
        self._mooncake_sender_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_mm_hashes(self) -> set[MMHash]:
        async with self.finished_recving_mm_hashes.lock:
            finished_recving_mm_hashes = self.finished_recving_mm_hashes.set
            self.finished_recving_mm_hashes.set = set()
        return finished_recving_mm_hashes

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        fut = None
        if not self.is_producer:
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_mm_hashes(), self.receiver_loop
            )

        if self.is_producer:
            with self.finished_sending_mm_hashes.lock:
                finished_sending_mm_hashes = self.finished_sending_mm_hashes.set
                self.finished_sending_mm_hashes.set = set()
        else:
            finished_sending_mm_hashes = set()

        finished_recving_mm_hashes = fut.result() if fut else set()

        if finished_sending_mm_hashes or finished_recving_mm_hashes:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_mm_hashes),
                len(finished_recving_mm_hashes),
            )

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        with self.mm_hashes_need_send.lock:
            expired_reqs = [
                req_id
                for req_id, send_meta in self.mm_hashes_need_send.mm_hashes.items()
                if send_meta.expire_time < now
            ]
            for req_id in expired_reqs:
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                del self.mm_hashes_need_send.mm_hashes[req_id]
            if expired_reqs:
                finished_sending_mm_hashes.update(expired_reqs)

        return finished_sending_mm_hashes or None, finished_recving_mm_hashes or None

    async def receive_ec(
        self, path: str, mm_hash_items: list[tuple[MMHash, MMHashMeta]]
    ):
        mm_hashes, mm_hashes_meta = map(list, zip(*mm_hash_items))
        metadata = MooncakeECAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            mm_hashes=mm_hashes,
            enc_base_addrs=[meta.mm_base_addr for meta in mm_hashes_meta],
            enc_token_bytes=[
                meta.num_encoder_tokens * self.byte_per_token for meta in mm_hashes_meta
            ],
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeAgentMetadata: %d bytes", len(encoded_data)
        )
        logger.debug("Sending ec transfer request for %s on path: %s", mm_hashes, path)

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        try:
            await sock.send(encoded_data)
            ret_msg = await sock.recv()
            if ret_msg != TRANS_DONE:
                logger.error(
                    "Error happens during tranfering kvcache for %s, see logs in prefiller.",  # noqa: E501
                    mm_hashes,
                )
                return
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error(
                "MooncakeAgentMetadata transfer failed for %s: %s", mm_hashes, e
            )
            return
        finally:
            sock.close()

        async with self.finished_recving_mm_hashes.lock:
            self.finished_recving_mm_hashes.set.update(mm_hashes)

        logger.debug("pulling ec_caches for %s finished", mm_hashes)

    def group_ec_pull(self, metadata: MooncakeECConnectorMetadata):
        ec_pulls = defaultdict(list)
        for mm_hash, meta in metadata.mm_hashes_to_recv.items():
            logger.debug(
                "start_load_ec for request %s from remote engine. "
                "Num of encoder token: %s.",
                mm_hash,
                meta.mm_hash_meta.num_encoder_tokens,
            )
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            ec_pulls[path].append((mm_hash, meta.mm_hash_meta))

        return ec_pulls

    def start_load_caches(self, metadata: MooncakeECConnectorMetadata):
        if not self.is_producer:
            ec_pulls = self.group_ec_pull(metadata)
            for path, mm_hash_items in ec_pulls.items():
                asyncio.run_coroutine_threadsafe(
                    self.receive_ec(path, mm_hash_items), self.receiver_loop
                )
        else:
            with self.mm_hashes_need_send.lock:
                for mm_hash, mm_hash_meta in metadata.mm_hashes_to_send.items():
                    if mm_hash_meta:
                        # Already gone through request_finished()
                        send_meta = self.mm_hashes_need_send.mm_hashes[mm_hash]
                        send_meta.mm_hash_meta = mm_hash_meta
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter()
                            + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                        )
                    else:
                        # From update_state_after_alloc(),
                        # but not reach request_finished() yet
                        self.mm_hashes_need_send.mm_hashes[mm_hash] = SendMMHashMeta(
                            mm_hash_meta=MMHashMeta(), ready=threading.Event()
                        )


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
