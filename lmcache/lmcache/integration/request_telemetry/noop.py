# SPDX-License-Identifier: Apache-2.0
"""
No-op implementation of request telemetry.
"""

# Standard
from typing import Any

# First Party
from lmcache.integration.request_telemetry.base import RequestTelemetry


class NoOpRequestTelemetry(RequestTelemetry):
    """
    A no-op implementation of RequestTelemetry.

    This implementation does nothing when events are reported.
    Use this when telemetry is disabled or not needed.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        # NoOp doesn't need any config, but accepts it for interface consistency
        pass

    def on_request_store_finished(
        self,
        request_ids_set: set[str],
        model_name: str,
        world_size: int,
        kv_rank: int,
    ) -> None:
        pass

    def close(self) -> None:
        pass
