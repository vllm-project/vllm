# SPDX-License-Identifier: Apache-2.0

import copy
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from itertools import count
from queue import Queue
from typing import Any, Callable, Optional, Union

import msgspec
import zmq

from vllm.config import KVEventsConfig, ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class EventBatch(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]


class KVCacheEvent(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
        tag=True):
    """Base class for all KV cache-related events"""


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]
    data_parallel_rank: Optional[int] = None


class EventPublisher(ABC):
    """Lightweight publisher for EventBatch batches."""

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
        connect.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    """
    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        # Storage
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._endpoint = endpoint
        self._replay_endpoint = replay_endpoint
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._seq_gen = count()
        self._topic_bytes = topic.encode('utf-8')

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(target=self._publisher_thread,
                                        daemon=True,
                                        name="zmq-publisher")
        self._thread.start()

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        self._event_queue.put(events)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        self._running = False
        self._event_queue.put_nowait(None)

        start = time.time()
        pending_items = True
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Clean up ZMQ resources
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention
            if ("*" in self._endpoint or "::" in self._endpoint
                    or self._endpoint.startswith("ipc://")
                    or self._endpoint.startswith("inproc://")):
                self._pub.bind(self._endpoint)
            else:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        self._pack = msgspec.msgpack.Encoder()

        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # --- main queue (critical) ---------------------------------
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break  # Sentinel received, exit thread
            except queue.Empty:
                continue

            try:
                seq = next(self._seq_gen)

                payload = self._pack.encode(event)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart(
                    (self._topic_bytes, seq_bytes, payload))

                self._buffer.append((seq, payload))
                self._event_queue.task_done()

            except Exception as e:
                # Publishing failed;  back-off a bit to avoid a tight error loop
                logger.exception("Error in publisher thread: %s", e)
                time.sleep(0.1)

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                # (identity, empty_delim) are stripped off by the router
                # receiving payload is (seq_bytes, payload)
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf))
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str,
                           ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(cls, config: Optional[KVEventsConfig]) -> EventPublisher:
        """Create publisher from a config mapping."""
        if not config:
            return NullEventPublisher()

        config_dict = asdict(config)

        kind = config_dict.pop("publisher", "null")
        config_dict.pop("enable_kv_cache_events")
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(**config_dict)


def _offset_endpoint_port(endpoint: str, data_parallel_rank: int) -> str:
    """Helper function to offset the port in an endpoint by 
        the data parallel rank.

    Args:
        endpoint: The endpoint string 
            (e.g., "tcp://*:5557" or "inproc://cache")
        data_parallel_rank: The data parallel rank to offset by

    Returns:
        The endpoint with the port offset by data_parallel_rank 
            or suffix appended
    """
    if "inproc" in endpoint:
        return f"{endpoint}_dp{data_parallel_rank}"
    elif "tcp" in endpoint:
        if endpoint and ":" in endpoint:
            # Get everything after the last colon (the port)
            last_colon_idx = endpoint.rfind(":")
            base_addr = endpoint[:last_colon_idx]
            base_port = int(endpoint[last_colon_idx + 1:])
            new_port = base_port + data_parallel_rank
            return f"{base_addr}:{new_port}"
        return endpoint
    else:
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


def get_kv_event_publisher(
        parallel_config: ParallelConfig,
        kv_events_config: Optional[KVEventsConfig]) -> EventPublisher:
    """Create a KV event publisher for data parallel masters only.

    Only one publisher is created per data parallel group 
    to avoid duplicate events. The endpoint port is offset by the 
    data parallel rank to ensure unique publishers.

    Args:
        parallel_config: Parallel configuration containing rank information
        kv_events_config: KV events configuration, 
            if None returns NullEventPublisher

    Returns:
        EventPublisher instance 
            (either configured publisher or NullEventPublisher)
    """
    # Check if KV events are enabled
    if not kv_events_config or not kv_events_config.enable_kv_cache_events:
        return NullEventPublisher()

    # Create a KV event publisher only one for each dp group
    data_parallel_rank = parallel_config.data_parallel_rank
    data_parallel_size = parallel_config.data_parallel_size

    # If data_parallel_rank <= 1, no need to offset ports
    if data_parallel_size <= 1:
        return EventPublisherFactory.create(kv_events_config)

    # Create a modified config with port offsetting
    modified_config = copy.deepcopy(kv_events_config)

    # Apply port offsetting to the endpoint
    original_endpoint = modified_config.endpoint
    modified_config.endpoint = _offset_endpoint_port(original_endpoint,
                                                     data_parallel_rank)
    if original_endpoint != modified_config.endpoint:
        logger.info(
            "KV event publisher endpoint adjusted from %s to %s for DP rank %d",
            original_endpoint,
            modified_config.endpoint,
            data_parallel_rank,
        )

    # Apply port offsetting to the replay_endpoint if it exists
    if modified_config.replay_endpoint:
        original_replay_endpoint = modified_config.replay_endpoint
        modified_config.replay_endpoint = _offset_endpoint_port(
            original_replay_endpoint, data_parallel_rank)
        if original_replay_endpoint != modified_config.replay_endpoint:
            logger.info(
                ("KV event publisher replay_endpoint "
                 "adjusted from %s to %s for DP rank %d"),
                original_replay_endpoint,
                modified_config.replay_endpoint,
                data_parallel_rank,
            )

    return EventPublisherFactory.create(modified_config)
