# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional, Union

import msgspec
import zmq

from vllm.config import KVEventsConfig


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


class EventPublisher(ABC):
    """Lightweight publisher for EventBatch batches."""

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    def close(self) -> None:  # optional
        return


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

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
    """

    def __init__(
        self,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        topic: str = "",
    ) -> None:
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.set_hwm(hwm)
        # Heuristic: bind if wildcard / * present, else connect.
        # bind stable, connect volatile convention
        if "*" in endpoint or "::" in endpoint or endpoint.startswith(
                "ipc://") or endpoint.startswith("inproc://"):
            self._pub.bind(endpoint)
        else:
            self._pub.connect(endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        self._replay = None
        if replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            self._replay.bind(replay_endpoint)

        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)
        self._seq = 0
        self._pack = msgspec.msgpack.Encoder()

        self._topic = topic
        self._topic_bytes = topic.encode('utf-8')

    def publish(self, events: EventBatch) -> None:
        self._service_replay()
        payload = self._pack.encode(events)
        seq_bytes = self._seq.to_bytes(8, "big")

        self._pub.send_multipart((self._topic_bytes, seq_bytes, payload))

        self._buffer.append((self._seq, payload))
        self._seq += 1

    def close(self) -> None:
        try:
            self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            # Do not terminate context; other sockets may use it.
            pass

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        if self._replay is None:
            return
        if not self._replay.poll(0):  # no request waiting
            return

        client_id, start_seq_bytes = self._replay.recv_multipart()
        start_seq = int.from_bytes(start_seq_bytes, "big")
        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf))
        # Send end of sequence marker
        self._replay.send_multipart((client_id, b"", b""))


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

        config_dict = config.model_dump()

        kind = config_dict.pop("publisher", "null")
        config_dict.pop("enable_kv_cache_events", False)
        try:
            constructor = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        return constructor(**config_dict)
