# SPDX-License-Identifier: Apache-2.0
"""Generic runtime reconfiguration protocol for L2 adapters."""

# Future
from __future__ import annotations

# Standard
from typing import Optional, Protocol, TypedDict, runtime_checkable


class L2ReconfigureError(RuntimeError):
    """HTTP-mappable runtime L2 reconfiguration error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        payload: Optional[dict[str, object]] = None,
    ) -> None:
        """Create a runtime reconfiguration error.

        Args:
            status_code: HTTP status code the API should return.
            message: Human-readable error message.
            payload: Optional response body. When omitted, ``{"error": message}``
                is used.
        """
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload if payload is not None else {"error": message}


class L2ReconfigureStatus(TypedDict):
    """Standard status envelope for runtime-reconfigurable L2 adapters."""

    backend: str
    supported_operations: list[str]
    status: dict[str, object]


@runtime_checkable
class L2ReconfigurableAdapter(Protocol):
    """Protocol implemented by L2 adapters with runtime reconfiguration."""

    def reconfigure_status(self) -> L2ReconfigureStatus:
        """Return JSON-serializable runtime reconfiguration status."""
        ...

    def reconfigure(
        self,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        """Apply an adapter-specific runtime reconfiguration operation.

        Args:
            operation: Adapter-specific operation name.
            payload: Adapter-specific operation payload.

        Returns:
            JSON-serializable operation result.
        """
        ...
