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
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
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

    snapshot_endpoint: str | None = None
    """The zmq ROUTER endpoint that serves snapshots of the current KV cache
    state, so a consumer that starts late can catch up and then follow the live
    stream. Setting it also adds a publisher identity to every live batch and
    sends an empty batch every second while idle. See the KV event snapshots
    docs for the protocol.
    """

    snapshot_max_blocks: int = 1_000_000
    """The most block records the snapshot recorder keeps, and separately the
    most live block references across all tiers. Going over either stops
    snapshots until the engine restarts.
    """

    snapshot_max_response_bytes: int = 256 * 1024 * 1024
    """The most encoded bytes in one snapshot reply. A larger snapshot stops
    snapshots until the engine restarts.
    """

    def __post_init__(self):
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
