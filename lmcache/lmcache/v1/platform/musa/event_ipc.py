# SPDX-License-Identifier: Apache-2.0
"""MUSA device-event IPC backend.

The MUSA capability gate and TorchMUSA module loading stay in this platform
package. Generic multiprocess code depends only on
:class:`lmcache.v1.platform.base.event_ipc.EventIPCBackend`.
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    EventIPCBackend,
)


class MusaEventIPCBackend:
    """Event IPC backend backed by TorchMUSA interprocess events."""

    device_type: str = "musa"

    def __init__(self, ipc_module: Any | None = None) -> None:
        """Create a MUSA event IPC backend.

        Args:
            ipc_module: Optional adapter exposing the MUSA capability and module
                helpers from ``platform.musa.ipc_wrapper``. It is injectable for
                tests.
        """
        self._ipc_module = ipc_module
        self._default_backend_cache: DefaultEventIPCBackend | None = None

    def _ipc(self) -> Any:
        """Return the MUSA IPC wrapper module, importing it lazily."""
        if self._ipc_module is None:
            # First Party
            from lmcache.v1.platform.musa import ipc_wrapper

            self._ipc_module = ipc_wrapper
        return self._ipc_module

    def _default_backend(self) -> DefaultEventIPCBackend:
        """Return a default event adapter bound to the TorchMUSA module.

        Returns:
            A CUDA-style event adapter using the current TorchMUSA event API.

        Raises:
            RuntimeError: If TorchMUSA cannot be loaded.
        """
        backend = self._default_backend_cache
        if backend is not None:
            return backend

        get_torch_musa_module = getattr(self._ipc(), "get_torch_musa_module", None)
        if not callable(get_torch_musa_module):
            raise RuntimeError("MUSA event IPC requires the TorchMUSA module")
        torch_musa = get_torch_musa_module()
        if torch_musa is None:
            raise RuntimeError("MUSA event IPC requires the TorchMUSA module")
        backend = DefaultEventIPCBackend(
            event_module=torch_musa,
            device_type=self.device_type,
        )
        self._default_backend_cache = backend
        return backend

    def check_event_support(self, device: object) -> None:
        """Fail closed unless the opt-in MUSA event IPC API is available.

        Args:
            device: Target MUSA device.

        Raises:
            RuntimeError: If the TorchMUSA event IPC API is unavailable.
        """
        is_available = getattr(self._ipc(), "is_musa_event_ipc_available", None)
        if not callable(is_available) or not is_available():
            raise RuntimeError(
                "Device backend 'musa' does not support interprocess events: "
                "TorchMUSA event IPC is unavailable."
            )
        try:
            self._default_backend()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Device backend 'musa' does not support interprocess events: {exc}"
            ) from exc

    def create_event(self, device: object) -> object:
        """Create a TorchMUSA interprocess event."""
        return self._default_backend().create_event(device)

    def export_event(self, event: object, device: object) -> bytes:
        """Serialize a TorchMUSA event for another process."""
        return self._default_backend().export_event(event, device)

    def import_event(self, handle: bytes, device: object) -> object:
        """Import a TorchMUSA event handle on ``device``."""
        return self._default_backend().import_event(handle, device)

    def record_event(self, event: object, stream: object) -> None:
        """Record an event on a TorchMUSA stream."""
        self._default_backend().record_event(event, stream)

    def wait_event(self, event: object, stream: object) -> None:
        """Make a TorchMUSA stream wait for an event."""
        self._default_backend().wait_event(event, stream)

    def query_event(self, event: object) -> bool:
        """Return whether a TorchMUSA event has completed."""
        return self._default_backend().query_event(event)

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until a TorchMUSA event completes."""
        self._default_backend().synchronize_event(event, device)


assert isinstance(MusaEventIPCBackend(), EventIPCBackend)
