# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector._nixl_import import (  # noqa: E501
    NixlWrapper,
)
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket

logger = init_logger(__name__)

GET_META_MSG = b"get_meta_msg"


@dataclass
class NixlAgentMetadata(KVConnectorHandshakeMetadata):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_lens: list[int]
    attn_backend_name: str
    kv_cache_layout: str


class HandshakeStrategy(ABC):
    """
    Abstract base class for handshake strategies.
    This class is used to abstract the handshake process for different
    communication protocols.
    """

    def __init__(
        self,
        nixl_wrapper: NixlWrapper,
        tp_rank: int,
        tp_size: int,
        side_channel_port: int,
        engine_id: str,
        handshake_lock: Optional[threading.RLock] = None,
    ):
        self.nixl_wrapper = nixl_wrapper
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.side_channel_port = side_channel_port
        self.engine_id = engine_id
        self.handshake_lock = handshake_lock

    @abstractmethod
    def initiate_handshake(
        self, host: str, port: int, remote_tp_size: int, expected_engine_id: str
    ) -> dict[int, str]:
        pass

    @abstractmethod
    def setup_listener(self, metadata: NixlAgentMetadata) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class ZmqHandshakeStrategy(HandshakeStrategy):
    """
    Handshake strategy that uses a ZMQ socket at port defined by
    VLLM_NIXL_SIDE_CHANNEL_PORT + tp_rank for communication.
    This is the default handshake strategy for NIXL, and is P2P.
    """

    def __init__(
        self,
        nixl_wrapper,
        tp_rank: int,
        tp_size: int,
        side_channel_port: int,
        engine_id: str,
        add_remote_agent_func,
        handshake_lock: Optional[threading.RLock] = None,
    ):
        super().__init__(
            nixl_wrapper, tp_rank, tp_size, side_channel_port, engine_id, handshake_lock
        )
        self.add_remote_agent_func = add_remote_agent_func
        self._listener_thread: Optional[threading.Thread] = None
        self._tp_size: dict[str, int] = {engine_id: tp_size}

    def initiate_handshake(
        self, host: str, port: int, remote_tp_size: int, expected_engine_id: str
    ) -> dict[int, str]:
        start_time = time.perf_counter()

        def handshake(path: str, rank: int) -> tuple[NixlAgentMetadata, str]:
            with self._zmq_ctx(zmq.REQ, path) as sock:
                sock.send(GET_META_MSG)
                metadata_bytes = sock.recv()
                decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                metadata = decoder.decode(metadata_bytes)
                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )
                # Register Remote agent (with locking if provided)
                if self.handshake_lock:
                    with self.handshake_lock:
                        agent_name = self.add_remote_agent_func(
                            metadata, rank, remote_tp_size
                        )
                else:
                    agent_name = self.add_remote_agent_func(
                        metadata, rank, remote_tp_size
                    )
                setup_agent_time = time.perf_counter()

                logger.debug(
                    "NIXL handshake: add agent took: %s",
                    setup_agent_time - got_metadata_time,
                )
                return metadata, agent_name

        # Handshake with remote agent-rank0 first to get the tp_size of remote
        path = make_zmq_path("tcp", host, port)
        logger.debug("Querying master rank metadata on path: %s", path)
        metadata, agent_name_0 = handshake(path, 0)

        agents = {0: agent_name_0}

        # Handshake only with the other TP remote the current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio
        if p_remote_rank > 0:
            path = make_zmq_path("tcp", host, port + p_remote_rank)
            logger.debug(
                "Querying metadata on path: %s at remote rank %s", path, p_remote_rank
            )
            _, agent_name = handshake(path, p_remote_rank)
            agents[p_remote_rank] = agent_name

        return agents

    def setup_listener(self, metadata: NixlAgentMetadata) -> None:
        ready_event = threading.Event()
        self._listener_thread = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="nixl_handshake_listener",
        )
        self._listener_thread.start()
        ready_event.wait()

    def cleanup(self) -> None:
        if self._listener_thread:
            self._listener_thread.join(timeout=0)

    @staticmethod
    def _nixl_handshake_listener(
        metadata: NixlAgentMetadata,
        ready_event: threading.Event,
        base_port: int,
        tp_rank: int,
    ):
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes", size_in_bytes)

        # Listen for new requests for metadata
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, base_port + tp_rank)
        logger.debug("Starting listening on path: %s", path)
        with ZmqHandshakeStrategy._zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data))

    @staticmethod
    @contextlib.contextmanager
    def _zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
        if socket_type not in (zmq.ROUTER, zmq.REQ):
            raise ValueError(f"Unexpected socket type: {socket_type}")

        ctx: Optional[zmq.Context] = None
        try:
            ctx = zmq.Context()
            yield make_zmq_socket(
                ctx=ctx,
                path=addr,
                socket_type=socket_type,
                bind=socket_type == zmq.ROUTER,
            )
        finally:
            if ctx is not None:
                ctx.destroy(linger=0)
