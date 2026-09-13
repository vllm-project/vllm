# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side of the ECZmqConnector.

Producer: `save_caches` copies the freshly computed encoder output to host
memory and hands it to a sender thread, which pushes it to every destination
rank named by the scheduler.

Consumer: a receive thread parks incoming embeddings in `EmbeddingStaging`;
`start_load_caches` moves the ones the scheduler asked for into the GPU encoder
cache, and `build_worker_meta` tells the scheduler what has landed.
"""

import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.zmq.common import (
    ECZmqConnectorMetadata,
    ECZmqOptions,
    ECZmqWorkerMetadata,
    ZmqDst,
    parse_zmq_options,
)
from vllm.distributed.ec_transfer.ec_connector.zmq.protocol import ECZmqProtocol
from vllm.distributed.ec_transfer.ec_connector.zmq.worker.staging import (
    EmbeddingStaging,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    is_valid_ipv6_address,
    make_zmq_path,
    split_zmq_path,
)
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# How long the receive loop waits for a message before re-checking whether the
# worker is shutting down.
_POLL_INTERVAL_MS = 50
# How long the receive loop waits for staging space before retrying.
_STAGING_RETRY_S = 0.005
# Bounded receive queue, so a producer outrunning this consumer is pushed back
# on rather than filling this process's memory with undelivered messages.
_RECV_HWM = 8


@dataclass
class _PendingSend:
    """One embedding waiting to go out on the wire."""

    mm_hash: str
    embedding: torch.Tensor
    endpoints: tuple[str, ...]
    # Marks the end of the device-to-host copy; None when the embedding was
    # already in host memory.
    copy_done: Any | None = None


@dataclass
class _InflightLoads:
    """Host buffers that a queued host-to-device copy still reads from."""

    event: Any
    sources: list[torch.Tensor] = field(default_factory=list)


class ECZmqWorker:
    """Worker-side delegate for the ECZmqConnector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        recv_endpoint: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            vllm_config: the engine config.
            recv_endpoint: consumer receive address; defaults to this rank's
                slice of the configured port range.
            device: where loaded embeddings land; defaults to the platform
                device.
        """
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._options: ECZmqOptions = parse_zmq_options(vllm_config)
        self._is_producer = ec_config.is_ec_producer
        self._is_consumer = ec_config.is_ec_consumer
        self._device = device or torch.device(current_platform.device_type)
        self._pin_memory = is_pin_memory_available()

        # All TP/PCP ranks hold identical encoder output, so one rank sends and
        # every consumer rank receives its own copy. DCP subdivides TP, so
        # tp_rank == 0 covers it.
        self._is_send_rank = (
            get_tensor_model_parallel_rank() == 0 and get_pcp_group().rank_in_group == 0
        )

        self._ctx = zmq.Context()
        self._stopped = threading.Event()

        # Producer state.
        self._send_queue: queue.Queue[_PendingSend | None] = queue.Queue(
            maxsize=self._options.max_inflight_sends
        )
        self._send_sockets: dict[str, zmq.Socket] = {}
        self._send_thread: threading.Thread | None = None
        self._finished_sending: set[str] = set()
        self._finished_lock = threading.Lock()
        self._send_stream: Any | None = None

        # Consumer state.
        self._staging = EmbeddingStaging(
            capacity_bytes=self._options.staging_bytes,
            ttl_s=self._options.staging_ttl_s,
        )
        self._recv_socket: zmq.Socket | None = None
        self._recv_thread: threading.Thread | None = None
        self._inflight_loads: list[_InflightLoads] = []
        self._load_stream: Any | None = None

        if self._is_producer and self._is_send_rank:
            self._send_thread = threading.Thread(
                target=self._send_loop, name="ec_zmq_send", daemon=True
            )
            self._send_thread.start()

        if self._is_consumer:
            endpoint = recv_endpoint or self._local_recv_endpoint()
            self._recv_socket = self._bind_recv_socket(endpoint)
            self._recv_thread = threading.Thread(
                target=self._recv_loop, name="ec_zmq_recv", daemon=True
            )
            self._recv_thread.start()
            logger.info("EC ZMQ: receiving embeddings on %s", endpoint)

    # ==============================
    # Producer path
    # ==============================

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECZmqConnectorMetadata,
    ) -> None:
        """Queue `encoder_cache[mm_hash]` for delivery to its destinations."""
        if not self._is_send_rank:
            return
        dsts = connector_metadata.sends.get(mm_hash)
        if not dsts:
            return

        embedding = encoder_cache.get(mm_hash)
        if embedding is None:
            logger.error("EC ZMQ: no encoder output to send for mm_hash %s", mm_hash)
            self._report_sent(mm_hash)
            return

        host_embedding, copy_done = self._copy_to_host(embedding)
        pending = _PendingSend(
            mm_hash=mm_hash,
            embedding=host_embedding,
            endpoints=tuple(_endpoints(dsts)),
            copy_done=copy_done,
        )
        try:
            self._send_queue.put(pending, timeout=self._options.send_timeout_s)
        except queue.Full:
            # Dropping is better than stalling the engine forever; the consumer
            # times out waiting for this item and fails its request.
            logger.error(
                "EC ZMQ: send queue full for %.0fs, dropping mm_hash %s",
                self._options.send_timeout_s,
                mm_hash,
            )
            self._report_sent(mm_hash)

    def get_finished(self) -> set[str] | None:
        """Return the mm_hashes whose delivery completed since the last call."""
        with self._finished_lock:
            if not self._finished_sending:
                return None
            finished = self._finished_sending
            self._finished_sending = set()
        return finished

    def _copy_to_host(self, embedding: torch.Tensor) -> tuple[torch.Tensor, Any | None]:
        """Stage `embedding` in host memory for the sender thread.

        The copy runs on a dedicated stream and is fenced by an event: the
        sender thread must not read the buffer before the encoder forward that
        produced it has landed.
        """
        if embedding.is_cpu:
            return embedding.detach(), None

        stream = self._get_send_stream()
        host = torch.empty(
            embedding.shape,
            dtype=embedding.dtype,
            device="cpu",
            pin_memory=self._pin_memory,
        )
        with current_platform.stream(stream):
            stream.wait_stream(current_platform.current_stream())
            host.copy_(embedding, non_blocking=True)
            event = current_platform.Event()
            event.record()
        return host, event

    def _get_send_stream(self) -> Any:
        if self._send_stream is None:
            self._send_stream = current_platform.Stream()
        return self._send_stream

    def _send_loop(self) -> None:
        protocol = ECZmqProtocol()
        while True:
            pending = self._send_queue.get()
            if pending is None:
                return
            try:
                if pending.copy_done is not None:
                    pending.copy_done.synchronize()
                frames = protocol.encode_embedding(pending.mm_hash, pending.embedding)
                for endpoint in pending.endpoints:
                    self._socket_for(endpoint).send_multipart(frames)
            except Exception:
                logger.exception(
                    "EC ZMQ: failed to send mm_hash %s to %s",
                    pending.mm_hash,
                    pending.endpoints,
                )
            finally:
                self._report_sent(pending.mm_hash)

    def _socket_for(self, endpoint: str) -> zmq.Socket:
        """Return the PUSH socket for `endpoint`, connecting on first use.

        Only ever called from the sender thread: ZMQ sockets are not
        thread-safe.
        """
        socket = self._send_sockets.get(endpoint)
        if socket is None:
            socket = self._ctx.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, _RECV_HWM)
            socket.setsockopt(zmq.SNDTIMEO, int(self._options.send_timeout_s * 1000))
            socket.setsockopt(zmq.LINGER, int(self._options.send_timeout_s * 1000))
            if _is_ipv6(endpoint):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(endpoint)
            self._send_sockets[endpoint] = socket
        return socket

    def _report_sent(self, mm_hash: str) -> None:
        with self._finished_lock:
            self._finished_sending.add(mm_hash)

    # ==============================
    # Consumer path
    # ==============================

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECZmqConnectorMetadata,
    ) -> None:
        """Move the embeddings named by the scheduler into the encoder cache."""
        self._retire_completed_loads()
        self._staging.expire()
        if not connector_metadata.loads:
            return

        sources: list[torch.Tensor] = []
        for mm_hash in connector_metadata.loads:
            if mm_hash in encoder_cache:
                continue
            staged = self._staging.pop(mm_hash)
            if staged is None:
                # The scheduler only asks for hashes every rank reported, so
                # this means the entry expired or was already consumed.
                logger.error(
                    "EC ZMQ: no staged embedding for mm_hash %s; the request "
                    "will fall back to local encoding",
                    mm_hash,
                )
                continue
            if staged.device == self._device:
                encoder_cache[mm_hash] = staged
                continue
            sources.append(staged)
            encoder_cache[mm_hash] = staged.to(self._device, non_blocking=True)

        if sources and self._pin_memory and self._device.type != "cpu":
            # A pinned source must outlive its async copy.
            event = current_platform.Event()
            event.record()
            self._inflight_loads.append(_InflightLoads(event=event, sources=sources))

    def build_worker_meta(self) -> ECZmqWorkerMetadata | None:
        """Report the embeddings that landed on this rank."""
        if not self._is_consumer:
            return None
        staged = self._staging.drain_arrivals()
        return ECZmqWorkerMetadata(staged=staged) if staged else None

    def _retire_completed_loads(self) -> None:
        if not self._inflight_loads:
            return
        self._inflight_loads = [
            inflight for inflight in self._inflight_loads if not inflight.event.query()
        ]

    def _local_recv_endpoint(self) -> str:
        flat_rank = (
            get_tensor_model_parallel_rank()
            + get_pcp_group().rank_in_group * get_tensor_model_parallel_world_size()
        )
        return make_zmq_path(
            "tcp",
            self._options.bind_host,
            self._options.recv_port_base + flat_rank,
        )

    def _bind_recv_socket(self, endpoint: str) -> zmq.Socket:
        socket = self._ctx.socket(zmq.PULL)
        # Set before bind: HWM only applies to connections made afterwards.
        socket.setsockopt(zmq.RCVHWM, _RECV_HWM)
        socket.setsockopt(zmq.LINGER, 0)
        if _is_ipv6(endpoint):
            socket.setsockopt(zmq.IPV6, 1)
        socket.bind(endpoint)
        return socket

    def _recv_loop(self) -> None:
        assert self._recv_socket is not None
        protocol = ECZmqProtocol()
        poller = zmq.Poller()
        poller.register(self._recv_socket, zmq.POLLIN)
        while not self._stopped.is_set():
            try:
                if not poller.poll(timeout=_POLL_INTERVAL_MS):
                    self._staging.expire()
                    continue
                frames = self._recv_socket.recv_multipart(copy=False)
                msg = protocol.decode_embedding(frames)
            except zmq.ContextTerminated:
                return
            except Exception:
                if self._stopped.is_set():
                    return
                logger.exception("EC ZMQ: dropping an undecodable message")
                continue
            self._stage(msg.mm_hash, msg.embedding)

    def _stage(self, mm_hash: str, embedding: torch.Tensor) -> None:
        nbytes = embedding.numel() * embedding.element_size()
        if nbytes > self._options.staging_bytes:
            logger.error(
                "EC ZMQ: embedding for mm_hash %s needs %d bytes but the "
                "staging budget is %d; raise ec_zmq_staging_bytes",
                mm_hash,
                nbytes,
                self._options.staging_bytes,
            )
            return
        warned = False
        while not self._stopped.is_set():
            if self._staging.try_put(mm_hash, embedding):
                return
            if not warned:
                logger.warning(
                    "EC ZMQ: staging area full (%d bytes), holding off on mm_hash %s",
                    self._options.staging_bytes,
                    mm_hash,
                )
                warned = True
            self._stopped.wait(_STAGING_RETRY_S)

    # ==============================
    # Shared
    # ==============================

    def shutdown(self) -> None:
        self._stopped.set()
        if self._send_thread is not None:
            self._send_queue.put(None)
            self._send_thread.join(timeout=self._options.send_timeout_s)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
        for socket in self._send_sockets.values():
            socket.close(linger=0)
        self._send_sockets.clear()
        if self._recv_socket is not None:
            self._recv_socket.close(linger=0)
            self._recv_socket = None
        self._ctx.term()
        self._staging.clear()
        self._inflight_loads.clear()


def _endpoints(dsts: list[ZmqDst]) -> list[str]:
    return [endpoint for dst in dsts for endpoint in dst.endpoints()]


def _is_ipv6(endpoint: str) -> bool:
    scheme, host, _ = split_zmq_path(endpoint)
    return scheme == "tcp" and is_valid_ipv6_address(host)
