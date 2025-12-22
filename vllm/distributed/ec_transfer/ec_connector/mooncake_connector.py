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
from copy import deepcopy

import msgspec
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
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

MMHash = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)


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
    enc_base_addrs: list[int]
    enc_token_bytes: list[int]


@dataclass
class MMHashMeta:
    num_encoder_tokens: int
    mm_base_addr: int


@dataclass
class RecvMMHashMeta:
    mm_hash_meta: MMHashMeta
    remote_host: str
    remote_port: int


@dataclass
class SendMMHashMeta:
    expire_time: float = float("inf")


@dataclass
class SendMeta:
    send_reqs: dict[Key, SendMMHashMeta]
    lock: threading.Lock


@dataclass
class FinishedSendMMHashSet:
    set: set[Key]
    lock: threading.Lock


@dataclass
class FinishedReceiveMMHashSet:
    set: set[MMHash]
    finish_recv_cond: asyncio.Condition


class MooncakeECConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_hashes_to_recv: dict[Key, RecvMMHashMeta] = {}

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
            self.connector_worker = MooncakeECConnectorWorker(
                vllm_config
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
        transfer_pool: TensorMemoryPool,
    ):
        """Register encoder cache tensors with Mooncake."""
        assert self.connector_worker is not None
        # For NIXL, we register the main encoder cache tensor
        # Individual mm_hash caches are handled via recv tensors
        if (
            hasattr(self.connector_worker, "transfer_pool")
            and self.connector_worker.transfer_pool is not None
        ):
            # Already registered
            return
        # The encoder_cache will be registered when it's first set
        # via register_encoder_cache method
        self.connector_worker.register_encoder_cache(transfer_pool)

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """Start loading encoder caches from remote via Mooncake."""
        assert self.connector_worker is not None
        metadata: MooncakeECConnectorMetadata = self._get_connector_metadata()

        self.connector_worker.start_load_caches(encoder_cache, metadata)
    
    def wait_for_load(self) -> None:
        assert self.connector_worker is not None
        return self.connector_worker.wait_for_load()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Save encoder cache to remote (handled by request_finished)."""
        self.connector_worker.save_caches(encoder_cache, mm_hash)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get finished receiving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)


class MooncakeECConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)
        self.is_producer = vllm_config.ec_transfer_config.is_ec_producer

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        # Track mm_hashes that need to be loaded from remote
        self._mm_hashes_need_recv: dict[Key, tuple[Request, int]] = {}
        # self._mm_hashes_need_send: dict[Key, SendMMHashMeta] = {}

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
                    and all(p in mm_hash_params for p in ("remote_host", "remote_port"))
                )
            result.append(has_cache)

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
                num_encoder_tokens = request.get_num_encoder_tokens(index)
                self._mm_hashes_need_recv[Key(mm_hash, request.request_id)] = (
                    request,
                    num_encoder_tokens
                )
                logger.debug(
                    "Added mm_hash %s to recv queue",
                    mm_hash
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
        for key, (request, num_encoder_tokens) in self._mm_hashes_need_recv.items():
            mm_hash = key.mm_hash
            ec_transfer_params = getattr(request, "ec_transfer_params", None)
            if ec_transfer_params:
                mm_hash_params = ec_transfer_params.get(mm_hash)
                logger.debug(f"hero: mm_hash_params for {mm_hash}: {mm_hash_params}")
                if mm_hash_params:
                    meta.add_recv_req(
                        req_id=request.request_id,
                        mm_hash=mm_hash,
                        mm_hash_meta=MMHashMeta(
                            num_encoder_tokens=num_encoder_tokens,
                            mm_base_addr=0,
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

        # meta.mm_hashes_to_send = self._mm_hashes_need_send.copy()

        # Clear the lists once workers start the transfers
        self._mm_hashes_need_recv.clear()
        # self._mm_hashes_need_send.clear()

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
            return False, None

        # Build params for all mm_hashes in this request
        result_params: dict[MMHash, dict[str, Any]] = {}
        for idx, feature in enumerate(request.mm_features):
            mm_hash = feature.identifier
            req_id = request.request_id
            # self._mm_hashes_need_send[Key(mm_hash, req_id)] = SendMMHashMeta(
            #     expire_time = time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            # )
            # Return params keyed by mm_hash for proxy aggregation
            result_params[mm_hash] = {
                "do_remote_encode": True,
                "remote_host": self.side_channel_host,
                "remote_port": self.side_channel_port,
            }

        return len(result_params) > 0, result_params if result_params else None


class MooncakeECConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", "rdma",
                                           vllm_config.ec_transfer_config.ec_connector_extra_config.get("device_name"))
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
        self.dtype = vllm_config.model_config.dtype \
            if isinstance(vllm_config.model_config.dtype, torch.dtype) \
            else getattr(torch, vllm_config.model_config.dtype)
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        self.embed_size = vllm_config.model_config.get_inputs_embeds_size()
        self.byte_per_token = self.embed_size * dtype_size
        self.device_type = current_platform.device_type

        self._ENCODER_MM_BASE_ADDRS: dict[MMHash, int] = {}

        self.num_workers = vllm_config.ec_transfer_config.ec_connector_extra_config.get(
            "num_workers", 10
        )
        # self.mm_hashes_need_send: SendMeta = SendMeta(
        #     send_reqs={}, lock=threading.Lock()
        # )
        self.mm_hashes_need_recv: set[Key] = set()

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
            FinishedReceiveMMHashSet(set(), asyncio.Condition())
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
    
    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        addr = self.transfer_pool.store_tensor(encoder_cache[mm_hash])
        self._ENCODER_MM_BASE_ADDRS[mm_hash] = addr

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
        # send_mm_hashes: list[MMHash] = []
        # with self.mm_hashes_need_send.lock:
        send_mm_hashes = [mm_hash for (mm_hash, _) in meta.mm_hashes]

        self._send_caches(send_mm_hashes, meta)

        with self.finished_sending_mm_hashes.lock:
            keys: list[Key] = []
            for (mm_hash, req_ids) in meta.mm_hashes:
                keys.extend([(mm_hash, req_id) for req_id in req_ids])
            self.finished_sending_mm_hashes.set.update(keys)

    def _send_caches(
        self,
        send_mm_hashes: list[MMHash],
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
        for mm_hash, remote_token_byte, remote_base_addr in zip(
            send_mm_hashes, remote_token_bytes, remote_base_addrs
        ):
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

    def register_encoder_cache(self, transfer_pool: TensorMemoryPool):
        """Register the EC Cache data in mooncake."""
        self.transfer_pool = transfer_pool
        ret_value = self.engine.register_memory(
            transfer_pool.base_address, transfer_pool.max_block_size
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        logger.debug("registered tensor pool with size=%d", transfer_pool.max_block_size)

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
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            finished_recving_mm_hashes = self.finished_recving_mm_hashes.set
            self.finished_recving_mm_hashes.set = set()
        return finished_recving_mm_hashes

    def get_finished(
            self, finished_req_ids: set[str]
        ) -> tuple[set[str] | None, set[str] | None]:
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

        return finished_sending_mm_hashes or None, finished_recving_mm_hashes or None

    async def receive_ec(
        self,
        path: str,
        mm_hash_items: list[tuple[tuple[MMHash, list[ReqId]], MMHashMeta]],
        ec_cache: dict[str, torch.Tensor]
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

        for (mm_hash, _), addr, num_bytes in zip(metadata.mm_hashes, metadata.enc_base_addrs, metadata.enc_token_bytes):
            ec_cache[mm_hash] = self.transfer_pool.load_tensor(addr, self.dtype, (num_bytes//self.byte_per_token, self.embed_size), device=self.device_type, copy=True)

        async with self.finished_recving_mm_hashes.finish_recv_cond:
            mm_hashes = [mm_hash for (mm_hash, _) in mm_hashes]
            self.finished_recving_mm_hashes.set.update(mm_hashes)

            if self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv:
                self.finished_recving_mm_hashes.finish_recv_cond.notify_all()

        logger.debug("pulling ec_caches for %s finished", mm_hashes)

    def group_ec_pull(self, metadata: MooncakeECConnectorMetadata):
        ec_pulls: dict[str, dict[MMHash, tuple[list[ReqId], MMHashMeta]]] = defaultdict(dict)
        assert isinstance(metadata, MooncakeECConnectorMetadata)
        for key, meta in metadata.mm_hashes_to_recv.items():
            logger.debug(
                "start_load_ec for request %s from remote engine. "
                "Num of encoder token: %s.",
                key.mm_hash,
                meta.mm_hash_meta.num_encoder_tokens,
            )
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            mm_hashes_meta = ec_pulls[path]
            if key.mm_hash not in mm_hashes_meta:
                meta.mm_hash_meta.mm_base_addr = self.transfer_pool.allocate(
                    meta.mm_hash_meta.num_encoder_tokens * self.byte_per_token)
                mm_hashes_meta[key.mm_hash] = ([key.req_id], meta.mm_hash_meta)
            else:
                req_ids, _ = mm_hashes_meta[key.mm_hash]
                req_ids.append(key.req_id)

        return ec_pulls

    def start_load_caches(self, ec_cache: dict[str, torch.Tensor], metadata: MooncakeECConnectorMetadata):
        self.mm_hashes_need_recv = set([key.mm_hash for key in metadata.mm_hashes_to_recv.keys()])
        ec_pulls = self.group_ec_pull(metadata)
        for path, mm_hashes_meta in ec_pulls.items():
            mm_hash_items = [((mm_hash, req_ids), meta)
                            for mm_hash, (req_ids, meta)
                            in mm_hashes_meta.items()]
            asyncio.run_coroutine_threadsafe(
                self.receive_ec(path, mm_hash_items, ec_cache), self.receiver_loop
            )

    async def _wait_for_load(self) -> None:
        async with self.finished_recving_mm_hashes.finish_recv_cond:
            await self.finished_recving_mm_hashes.finish_recv_cond.wait_for(
                lambda: self.finished_recving_mm_hashes.set == self.mm_hashes_need_recv
            )
    
    def wait_for_load(self) -> None:
        fut = asyncio.run_coroutine_threadsafe(
            self._wait_for_load(), self.receiver_loop
        )
        fut.result()  # Block until complete


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )

#TODO: what if transfer pool is full??? lots of request at once