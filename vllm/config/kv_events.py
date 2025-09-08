# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class KVEventsConfig:
    """Configuration for KV event publishing."""

    enable_kv_cache_events: bool = False
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: Optional[str] = None
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
