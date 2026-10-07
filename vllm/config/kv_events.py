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
    """The number of event batches to keep for the replay endpoint. Only the
    last N batches can be replayed.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher, in batches. Once N
    batches are queued for a subscriber that is not keeping up, further
    batches are dropped for it.
    """

    max_queue_size: int = 100_000
    """The maximum number of event batches waiting to be published. When the
    queue is full, the scheduler blocks until there is room.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    def __post_init__(self):
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
