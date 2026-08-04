# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for request telemetry.

This module provides the interface for tracking request-level events
in LMCache, such as when a request finishes and its associated async
save operations complete.
"""

# Standard
from abc import ABC, abstractmethod
from typing import Any


class RequestTelemetry(ABC):
    """
    Abstract base class for request telemetry.

    This class defines the interface for capturing request-level telemetry
    events. Implementations can log events, emit metrics, or perform other
    actions when specific request lifecycle events occur.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def on_request_store_finished(
        self,
        request_ids_set: set[str],
        model_name: str,
        world_size: int,
        kv_rank: int,
    ) -> None:
        """
        Callback when request finishes AND all its KV cache store ops completes.

        This method ensures that request_ids_set is not empty.

        Technically this function is implemented by inspecting the return value
        of `get_finished` method.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()
