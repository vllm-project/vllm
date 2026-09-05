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
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


MEDIUM_GPU = "GPU"
MEDIUM_CPU = "CPU"
MEDIUM_STORAGE = "STORAGE"


class ExtraKey(msgspec.Struct, gc=False, frozen=True, tag=True):
    """Base class for typed extra keys attached to a ``BlockStored`` event.

    ``extra_keys`` in ``BlockStored`` is the *published* form of the internal
    block-hash extra keys computed by ``generate_block_hash_extra_keys`` (see
    ``vllm/v1/core/kv_cache_utils.py``). Publishing a typed, versioned schema
    (instead of raw tuples of ``Any``) gives external consumers such as the
    llm-d router a stable contract for:

    - locating which multi-modal inputs a block references (``MultiModalKey``),
    - filtering by LoRA / ``cache_salt`` / prompt-embeds identity,
    - and validating prefix-cache routing decisions without re-inferring the
      internal block-hash representation.

    Instances are ``frozen`` and hash by field value so they can be used in
    sets / as ``Counter`` keys by ``KVEventAggregator`` after msgpack
    round-trips (structs travel over ZMQ and may be reconstructed on the
    consumer side).
    """

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def to_tuple(self) -> tuple[Any, ...]:
        """Flatten this key to a hashable tuple.

        Mirrors the internal tuple form used by
        ``generate_block_hash_extra_keys`` so consumers can round-trip between
        the published schema and the block-hash representation.
        """
        raise NotImplementedError


class LoRAKey(ExtraKey, frozen=True, tag="lora"):
    """LoRA identity the block was computed with (``lora_name``)."""

    name: str

    def to_tuple(self) -> tuple[Any, ...]:
        return (self.name,)


class CacheSaltKey(ExtraKey, frozen=True, tag="cache_salt"):
    """``cache_salt`` applied to the request (present on the first block only)."""

    salt: str

    def to_tuple(self) -> tuple[Any, ...]:
        return (self.salt,)


class PromptEmbedsKey(ExtraKey, frozen=True, tag="prompt_embeds"):
    """Stable hash of the prompt-embedding slice covered by the block."""

    hash: bytes

    def to_tuple(self) -> tuple[Any, ...]:
        return (self.hash,)


class MultiModalKey(ExtraKey, frozen=True, tag="mm"):
    """A multi-modal input referenced by the block.

    ``hash`` is the same value as ``MultiModalFeatures.mm_hashes[modality][i]``
    from the render step (see ``vllm/entrypoints/scale_out/token_in_token_out/
    protocol.py``), i.e. the identifier used for encoder-output cache lookups
    (without a LoRA prefix; the prefixed form is ``MultiModalFeatureSpec.
    identifier``). ``block_offset`` is the offset of the item's tokens relative
    to the start of this block, matching the second element of the internal
    ``(identifier, offset)`` tuple.
    """

    modality: str
    hash: str
    block_offset: int

    def to_tuple(self) -> tuple[Any, ...]:
        return (self.hash, self.block_offset)


class LegacyExtraKey(ExtraKey, frozen=True, tag="legacy"):
    """Fallback wrapper for extra-key shapes we do not yet classify.

    Preserves the original value so no information is lost when an internal
    extra key does not match the known schema (e.g. future key types added to
    ``generate_block_hash_extra_keys``). ``value`` is normalised to a hashable
    form inside ``to_tuple`` so the containing block event remains usable as a
    ``Counter`` key.

    ``__eq__``/``__hash__`` compare through the normalised view so that
    msgpack round-trips (which turn nested tuples into lists) do not break
    equality: ``LegacyExtraKey(("a", "b")) == decode(encode(...))`` holds even
    though the decoded ``value`` is a list.
    """

    value: Any

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LegacyExtraKey):
            return NotImplemented
        return self.to_tuple() == other.to_tuple()

    def to_tuple(self) -> tuple[Any, ...]:
        return (_hashable_key(self.value),)


def _hashable_key(value: Any) -> Any:
    """Normalise ``value`` to a hashable form for ``LegacyExtraKey``.

    Raw tuple extra keys are hashable already; lists and dicts are converted to
    tuples so the containing ``ExtraKey`` remains usable as a ``Counter`` key.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable_key(v)) for k, v in value.items()))
    return value


#: Explicit union over all ``ExtraKey`` subclasses. msgspec only resolves
#: tagged unions that are spelled out explicitly, so consumers must decode
#: ``BlockStored.extra_keys`` entries against this union (or against
#: ``BlockStored`` whose field type uses it) to get typed keys back.
ExtraKeyUnion = ExtraKey | LoRAKey | CacheSaltKey | PromptEmbedsKey | (
    MultiModalKey | LegacyExtraKey
)


def extra_keys_to_typed(
    extra_keys_list: list[tuple[Any, ...] | None] | None,
    request: Any,
) -> list[tuple[ExtraKeyUnion, ...] | None] | None:
    """Convert internal block-hash extra-key tuples to the published schema.

    ``generate_block_hash_extra_keys`` (``vllm/v1/core/kv_cache_utils.py``)
    produces per-block tuples whose elements are bare values: MM
    ``(identifier, offset)`` 2-tuples, LoRA names, ``cache_salt``, and
    prompt-embeds ``bytes``. This helper maps each element to a typed
    ``ExtraKey`` so external consumers get a stable, versioned schema.

    The MM ``identifier`` is resolved back to its ``modality`` and unprefixed
    ``mm_hash`` via ``request.mm_features`` (identifier is unique per MM
    item). Unknown shapes are wrapped in ``LegacyExtraKey`` so no information
    is lost.

    ``None`` entries pass through unchanged (block has no extra keys).
    """
    if extra_keys_list is None:
        return None
    if not request.mm_features:
        id_to_mm: dict[str, tuple[str, str]] = {}
    else:
        id_to_mm = {
            f.identifier: (f.modality, f.mm_hash or f.identifier)
            for f in request.mm_features
        }

    lora_name = request.lora_request.name if request.lora_request else None
    cache_salt = request.cache_salt

    converted: list[tuple[ExtraKeyUnion, ...] | None] = []
    for block_keys in extra_keys_list:
        if block_keys is None:
            converted.append(None)
            continue
        typed: list[ExtraKeyUnion] = []
        for key in block_keys:
            if isinstance(key, ExtraKey):
                typed.append(key)
                continue
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], str)
                and isinstance(key[1], int)
            ):
                # (mm_identifier, offset) from _gen_mm_extra_hash_keys.
                identifier, offset = key
                modality, mm_hash = id_to_mm.get(identifier, ("", identifier))
                typed.append(
                    MultiModalKey(
                        modality=modality,
                        hash=mm_hash,
                        block_offset=offset,
                    )
                )
            elif isinstance(key, bytes):
                # prompt-embeds block hash.
                typed.append(PromptEmbedsKey(hash=key))
            elif isinstance(key, str) and lora_name is not None and key == lora_name:
                typed.append(LoRAKey(name=key))
            elif isinstance(key, str) and cache_salt is not None and key == cache_salt:
                typed.append(CacheSaltKey(salt=key))
            else:
                typed.append(LegacyExtraKey(value=key))
        converted.append(tuple(typed))
    return converted


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int

    """Deprecated: use `lora_name` for KV block key hash.
    Retained for backward compatibility.
    """
    lora_id: int | None

    medium: str | None
    lora_name: str | None

    """Schema version of this event. Incremented whenever the semantics of a
    field (or of ``extra_keys``) change in a way consumers must branch on.
    Version 1 introduces the typed ``ExtraKey`` schema for ``extra_keys``.
    """
    event_version: int = 1

    """Extra keys used in block hash computation, one entry per block in
    block_hashes. Each entry contains typed ``ExtraKey`` items: MM
    identifiers (``MultiModalKey``), LoRA name (``LoRAKey``), cache_salt
    (``CacheSaltKey``), prompt embedding hashes (``PromptEmbedsKey``), or a
    ``LegacyExtraKey`` fallback, for that specific block. Exposed for external
    KV cache consumers to reconstruct block hashes.
    """
    extra_keys: list[tuple[ExtraKeyUnion, ...] | None] | None = None

    """Store events carry cache-spec metadata so consumers can classify and
    filter groups as they are learned. Remove events only need group_idx+hash.
    """
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    """LOCAL or REMOTE relative to the publisher; None means unspecified."""
    locality: str | None = None
    """Secondary offloading tier identifier, if generated by one."""
    ownership: str | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
                tuple(self.extra_keys) if self.extra_keys else None,
                self.group_idx,
                self.kv_cache_spec_kind,
                self.kv_cache_spec_sliding_window,
                self.locality,
                self.ownership,
                self.event_version,
            )
        )


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    """LOCAL or REMOTE relative to the publisher; None means unspecified."""
    locality: str | None = None
    """Secondary offloading tier identifier, if generated by one."""
    ownership: str | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.medium,
                self.group_idx,
                self.locality,
                self.ownership,
            )
        )


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

        Args:
            events: List of KVCacheEvent objects.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    def get_common_events(self) -> list[KVCacheEvent]:
        """
        Return events that appeared in all workers.

        Returns:
            List of events present in all workers.
        """
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers
        ]

    def get_all_events(self) -> list[KVCacheEvent]:
        """
        Return all events for all workers.

        Returns:
            List of events for all workers.
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

        Args:
            count: Number to increment the workers by.
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

        Returns:
            int number of workers.
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

    def merge(self, other: "KVConnectorKVEvents") -> "KVConnectorKVEvents":
        self.add_events(other.get_all_events())
        return self


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

    def get_publisher_config(self) -> KVEventsConfig | None:
        """Return the publisher's resolved runtime configuration."""
        return None


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
        PUB address. Use `tcp://*:5557` to bind or `tcp://host:5557` to
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
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        assert self._endpoint is not None
        self._publisher_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=self._endpoint,
            replay_endpoint=self._replay_endpoint,
            buffer_steps=buffer_steps,
            hwm=hwm,
            max_queue_size=max_queue_size,
            topic=topic,
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

    def get_publisher_config(self) -> KVEventsConfig:
        return self._publisher_config

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
            if self._endpoint is not None and (
                "*" in self._endpoint
                or "::" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
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
                # Subscriber receives (topic, seq_bytes, payload)
                self._replay.send_multipart(
                    (client_id, b"", self._topic_bytes, seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        self._replay.send_multipart((client_id, b"", b"", self.END_SEQ, b""))

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
