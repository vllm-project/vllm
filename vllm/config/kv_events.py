# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal

from vllm.config.utils import config


@config
class KVEventsConfig:
    """Configuration for KV event publishing."""

    enable_kv_cache_events: bool = False
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    publisher: Literal["null", "zmq"] = None  # type: ignore[assignment]
    """The publisher to use for publishing kv events.
    Can be "null" or "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The endpoint to use for publishing kv events.
    For ZMQ, this is the PUB endpoint.
    """

    replay_endpoint: str | None = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    prefix_cache_upload_endpoint: str | None = None
    """Optional HTTP endpoint used by prefix-aware routing workers to upload
    KV cache events to the routing master. This is a separate best-effort side
    channel and does not replace the configured ``publisher``.
    Example: http://master:8000/prefix_routing/kv_events/node0.
    """

    prefix_cache_upload_max_queue_size: int = 100_000
    """Maximum number of KV event batches buffered for prefix-cache upload."""

    prefix_cache_upload_timeout: float = 5.0
    """HTTP request timeout in seconds for prefix-cache upload."""

    def __post_init__(self):
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
