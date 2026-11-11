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

    @property
    def enable_events(self) -> bool:
        return self.enable_ec_cache_events
