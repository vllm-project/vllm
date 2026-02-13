# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict
from itertools import count
from queue import Queue
from typing import Any

import msgspec
import zmq

from vllm.config.kv_events import KVEventsConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

logger = init_logger(__name__)


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


MEDIUM_GPU = "GPU"


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int

    lora_id: int | None
    """Deprecated: use `lora_name` for KV block key hash.
    Retained for backward compatibility.
    """

    medium: str | None
    lora_name: str | None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
            )
        )


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None

    def __hash__(self) -> int:
        return hash((tuple(self.block_hashes), self.medium))


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


class KVEventAggregator:
    """
    Aggregates KV events across multiple workers.
    Tracks how many times each event appears and returns only those
    that were emitted by all workers.
    """

    __slots__ = ("_event_counter", "_num_workers")

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers: int = num_workers

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """
        Add events from a worker batch.

        :param events: List of KVCacheEvent objects.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def get_common_events(self) -> list[KVCacheEvent]:
        """
        Return events that appeared in all workers.

        :return: List of events present in all workers.
        """
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers
        ]

    def get_all_events(self) -> list[KVCacheEvent]:
        """
        Return all events for all workers.

        :return: List of events for all workers.
        """
        return list(self._event_counter.elements())

    def clear_events(self) -> None:
        """
        Clear all tracked events.
        """
        self._event_counter.clear()

    def increment_workers(self, count: int = 1) -> None:
        """
        Increment the number of workers contributing events.

        :param count: Number to increment the workers by.
        """
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def reset_workers(self) -> None:
        """
        Reset the number of workers to 1.
        """
        self._num_workers = 1

    def get_number_of_workers(self) -> int:
        """
        Return the number of workers.

        :return: int number of workers.
        """
        return self._num_workers

    def __repr__(self) -> str:
        return (
            f"<KVEventAggregator workers={self._num_workers}, "
            f"events={len(self._event_counter)}>"
        )


class KVConnectorKVEvents(ABC):
    """
    Abstract base class for KV events.
    Acts as a container for KV events from the connector.
    """

    @abstractmethod
    def add_events(self, events: list[KVCacheEvent]) -> None:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self) -> "KVConnectorKVEvents":
        raise NotImplementedError

    @abstractmethod
    def increment_workers(self, count: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_all_events(self) -> list[KVCacheEvent]:
        raise NotImplementedError

    @abstractmethod
    def get_number_of_workers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear_events(self) -> None:
        raise NotImplementedError


class EventPublisher(ABC):
    """Lightweight publisher for EventBatch batches with data parallelism
    support.

    In data parallel setups, each DP rank runs its own EventPublisher instance
    to avoid duplicate events and ensure proper event attribution:

    - Each DP rank creates a separate publisher
    - Publishers automatically annotate events with their data_parallel_rank
    - This allows consumers to distinguish events from different DP ranks

    The publisher is responsible for adding DP metadata since the scheduler
    operates independently of DP topology and shouldn't need DP awareness.
    """

    def __init__(self, data_parallel_rank: int = 0) -> None:
        self._data_parallel_rank = data_parallel_rank

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
        PUB address. Default is `tcp://*:5557`.
    bind:
        If True, bind the publisher socket to the endpoint. If False, connect.
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
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        bind: bool = True,
        replay_endpoint: str | None = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        # Storage
        super().__init__(data_parallel_rank)
        self._event_queue = Queue[EventBatch | None](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: zmq.Socket | None = None
        self._replay: zmq.Socket | None = None
        self._dp_rank = data_parallel_rank

        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._bind = bind
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._seq_gen = count()
        self._topic_bytes = topic.encode("utf-8")

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.data_parallel_rank is None:
            events.data_parallel_rank = self._data_parallel_rank
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
            if self._endpoint is not None:
                if self._bind:
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
                self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))

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
                    (client_id, b"", seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: str | None, data_parallel_rank: int
    ) -> str | None:
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
        # Do nothing if input is None or data_parallel_rank is 0
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(
        cls, config: KVEventsConfig | None, data_parallel_rank: int = 0
    ) -> EventPublisher:
        """Create publisher from a config mapping."""
        if (
            config is None
            or not config.enable_kv_cache_events
            or config.publisher == "null"
        ):
            return NullEventPublisher()

        config_dict = asdict(config)

        kind = config_dict.pop("publisher")
        config_dict.pop("enable_kv_cache_events")
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(data_parallel_rank=data_parallel_rank, **config_dict)
