# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict
from itertools import count
from queue import Queue
from typing import Any, Callable, Optional, Union

import aiohttp
import msgspec
import zmq

from vllm.config import KVEventsConfig
from vllm.logger import init_logger
from vllm.utils import get_ip

logger = init_logger(__name__)


class EventBatch(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    data_parallel_rank: Optional[int] = None


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
        data_parallel_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
    ) -> None:
        # Storage
        super().__init__(data_parallel_rank)
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._dp_rank = data_parallel_rank

        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank)
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
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention
            if (self._endpoint is not None
                    and ("*" in self._endpoint or "::" in self._endpoint
                         or self._endpoint.startswith("ipc://")
                         or self._endpoint.startswith("inproc://"))):
                self._pub.bind(self._endpoint)
            elif self._endpoint is not None:
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

    @staticmethod
    def offset_endpoint_port(endpoint: Optional[str],
                             data_parallel_rank: int) -> Optional[str]:
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
                base_port = int(endpoint[last_colon_idx + 1:])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class HttpEventPublisher(EventPublisher):
    SHUTDOWN_TIMEOUT: float = 1.0
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    def __init__(
        self,
        data_parallel_rank: int,
        endpoint: str = "",
        max_queue_size: int = 100_000,
        port: int = 8080,
        **kwargs,
    ) -> None:
        # Storage
        super().__init__(data_parallel_rank)
        self._event_queue: asyncio.Queue[Optional[EventBatch]] = asyncio.Queue(
            maxsize=max_queue_size)
        self._publisher_task: Optional[asyncio.Task] = None
        self._init_task: Optional[asyncio.Task] = None

        # HTTP endpoint metadata
        self._endpoint = endpoint
        self.port = port
        self.ip = get_ip()

        self.create_stamp = str(int(time.time()))
        self.unique_id = uuid.uuid4().hex
        self.instance_id = self.ip + ":" + str(
            self.port) + ":" + self.create_stamp + self.unique_id

        # Thread
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_async_tasks,
                                        daemon=True)
        self._thread.start()

    def _run_async_tasks(self):
        asyncio.set_event_loop(self._loop)
        self._init_task = self._loop.create_task(self._heartbeat_loop())
        self._publisher_task = self._loop.create_task(self._publisher_loop())
        self._loop.run_forever()

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")

        try:
            self._loop.call_soon_threadsafe(
                lambda: self._event_queue.put_nowait(events))
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping events")

    async def _heartbeat_loop(self):
        """Periodically register this instance to signal liveness."""
        async with aiohttp.ClientSession() as session:
            init_dict = {
                "ip": self.ip,
                "port": self.port,
                "instance_id": self.instance_id,
            }
            payload = msgspec.json.encode(init_dict)
            while self._running:
                try:
                    async with session.post(
                            self._endpoint + "/v1/kv/init",
                            data=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=5.0,
                    ) as response:
                        response.raise_for_status()
                except Exception as e:
                    logger.exception("Error in init loop: %s", e)
                await asyncio.sleep(5)

    async def _post_event(self, session, event, endpoint):
        """Publish a single event to the endpoint with retry handling."""
        event_dict = {f: getattr(event, f) for f in event.__struct_fields__}
        event_dict["instance_id"] = self.instance_id
        payload = msgspec.json.encode(event_dict)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with session.post(
                        endpoint,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=10,
                ) as response:
                    if 200 <= response.status < 300:
                        return
                    elif 500 <= response.status < 600:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status)
                    else:
                        logger.warning(
                            "Client error, not retrying (status=%d, body=%s)",
                            response.status,
                            await response.text(),
                        )
                        return

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.MAX_RETRIES:
                    logger.exception("Publish retries exhausted: %s", e)
                    return
                await asyncio.sleep(self.RETRY_DELAY)
            except Exception as e:
                logger.exception("Unexpected error while publishing event: %s",
                                 e)
                return

    async def _publisher_loop(self):
        """Background thread that processes the event queue."""
        async with aiohttp.ClientSession() as session:
            while self._running or not self._event_queue.empty():
                try:
                    events = await asyncio.wait_for(self._event_queue.get(),
                                                    timeout=0.1)
                    if events is None:
                        break
                    for event in events.events:
                        event_type = event.__class__.__name__

                        if event_type == "BlockStored":
                            endpoint = self._endpoint + "/v1/kv/create"
                        elif event_type == "BlockRemoved":
                            endpoint = self._endpoint + "/v1/kv/delete"

                        asyncio.create_task(
                            self._post_event(session, event, endpoint))

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.exception("Error in publisher loop: %s", e)
                    await asyncio.sleep(0.1)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        if not self._running:
            return

        self._running = False
        self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        remaining = self._event_queue.qsize()

        if remaining > 0:
            logger.warning("Dropping %s unprocessed events during shutdown",
                           remaining)


class EventPublisherFactory:
    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "zmq": ZmqEventPublisher,
        "http": HttpEventPublisher,
    }

    @classmethod
    def register_publisher(cls, name: str,
                           ctor: Callable[..., EventPublisher]) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(cls,
               config: Optional[KVEventsConfig],
               data_parallel_rank: int = 0) -> EventPublisher:
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
        return constructor(data_parallel_rank=data_parallel_rank,
                           **config_dict)
