# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config.kv_events import PublisherConfig
from vllm.config.utils import config


@config
class ECEventsConfig(PublisherConfig):
    """Configuration for encoder cache (EC) event publishing."""

    enable_ec_cache_events: bool = False
    """If True, enable encoder cache events for tracking which multimodal
    inputs are cached and where. Events can be published externally by zmq
    using the event publisher config.
    """

    endpoint: str = "tcp://*:6558"
    """The zmq endpoint for EC events.

    Ports are offset by the data parallel rank like KV events. The block starts
    1000 ports above the 5557/5558 KV events block, so the two stay disjoint for
    up to 1000 DP ranks on one host, replay included.
    """

    @property
    def enable_events(self) -> bool:
        return self.enable_ec_cache_events
