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

    tier_scoped_clear_events: bool = False
    """If True, a prefix-cache reset that clears only the GPU block pool is
    published as `TierBlocksCleared(medium="GPU")` instead of `AllBlocksCleared`,
    so consumers tracking several tiers keep records for blocks still resident
    in a KV connector. Every subscriber must decode the `TierBlocksCleared` tag
    before this is enabled: a batch decodes as one unit, so an unknown tag drops
    the whole batch. `AllBlocksCleared` is still published when an attached
    connector reports that it cleared every tier.
    """

    def __post_init__(self):
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
