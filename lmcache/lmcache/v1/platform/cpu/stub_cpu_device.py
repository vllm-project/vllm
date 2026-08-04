# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from contextlib import nullcontext
from typing import Any

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class StubDeviceProperties:
    """Stub for torch_dev.get_device_properties() return value."""

    def __init__(self) -> None:
        self.name = "StubCPU"
        self.major = 0
        self.minor = 0
        self.total_memory = 0
        self.multi_processor_count = 0
        self.uuid = "stub-0000-0000-0000-000000000000"

    def __repr__(self) -> str:
        return f"StubDeviceProperties(name={self.name!r})"


class StubEvent:
    """Stub for a CUDA event, used in CPU-only test environments."""

    def __init__(
        self,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess
        self._recorded = False
        self._handle = b"stub_ipc_handle"

    def record(self, stream: Any = None) -> None:
        """Mark this event as recorded on the given stream.

        Args:
            stream: The stream to record on. If None, uses the
                current stream.
        """
        self._recorded = True

    def wait(self, stream: Any = None) -> None:
        """Make the given stream wait until this event completes.

        Args:
            stream: The stream that should wait. If None, uses the
                current stream.
        """
        return None

    def query(self) -> bool:
        """Check whether the event has completed.

        Returns:
            True always, since the stub has no real work.
        """
        return True

    def synchronize(self) -> None:
        """Block the host until this event completes.

        No-op in the stub implementation.
        """
        return None

    def elapsed_time(self, end_event: "StubEvent") -> float:
        """Return elapsed time in milliseconds between this event and *end_event*.

        Args:
            end_event: The ending event to measure against.

        Returns:
            Elapsed time in milliseconds. Always 0.0 for the stub.
        """
        return 0.0

    def ipc_handle(self) -> bytes:
        """Return an IPC handle for cross-process sharing.

        Returns:
            A bytes object representing the IPC handle.
        """
        return self._handle

    @classmethod
    def from_ipc_handle(cls, device: Any, handle: bytes) -> "StubEvent":
        """Reconstruct a StubEvent from an IPC handle.

        Args:
            device: The device to associate with the event.
            handle: The IPC handle bytes obtained from
                :meth:`ipc_handle`.

        Returns:
            A new StubEvent with interprocess=True and the given handle.
        """
        ev = cls(interprocess=True)
        ev._handle = handle
        return ev

    def __repr__(self) -> str:
        return f"StubEvent(interprocess={self.interprocess}, recorded={self._recorded})"


class StubStream:
    """Stub for a CUDA stream, used in CPU-only test environments."""

    def __init__(self, device: Any = "cpu", priority: int = 0, **kwargs: Any) -> None:
        self.device = device
        self.priority = priority
        self.cuda_stream = 0
        # Mirrors the ``ptr`` attribute exposed by ``cupy.cuda.Stream``
        # so callers (e.g. ``mp_observability.event_bus``) that pass a
        # raw stream pointer to native recorders accept this stub
        # without an isinstance check.
        self.ptr = 0

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        """Run ``callback(arg)`` synchronously.

        ``cupy.cuda.Stream.launch_host_func`` schedules the callback
        on the GPU stream's host-side completion queue; with no real
        stream there's nothing to wait for, so we just invoke it
        immediately. Exceptions are swallowed to mirror the cupy
        contract (callbacks are best-effort and must not propagate).
        """
        try:
            callback(arg)
        except Exception as e:  # noqa: BLE001
            logger.warning("launch_host_func callback raised: %s", e)

    def synchronize(self) -> None:
        """Block the host until all kernels on this stream complete.

        No-op in the stub implementation.
        """
        return None

    def wait_event(self, event: StubEvent) -> None:
        """Make this stream wait until *event* completes.

        Args:
            event: The event this stream should wait for.
        """
        return None

    def wait_stream(self, stream: "StubStream") -> None:
        """Make this stream wait until all kernels on *stream* complete.

        Args:
            stream: The stream whose pending work must finish before
                this stream continues.
        """
        return None

    def record_event(self, event: StubEvent | None = None) -> StubEvent:
        """Record an event on this stream and return it.

        Args:
            event: An existing event to record. If None, a new
                StubEvent is created.

        Returns:
            The recorded StubEvent.
        """
        event = event or StubEvent()
        event.record(self)
        return event

    def query(self) -> bool:
        """Check whether all kernels on this stream have completed.

        Returns:
            True always, since the stub has no real work.
        """
        return True

    @staticmethod
    def priority_range() -> tuple[int, int]:
        """Return the range of stream priorities.

        Returns:
            A tuple of (lowest_priority, highest_priority). Always
            (0, 0) for the stub.
        """
        return (0, 0)

    def __repr__(self) -> str:
        return f"StubStream(device={self.device}, priority={self.priority})"


class StubCPUDevice:
    """Stub stand-in for torch_dev in CPU-only test environments."""

    def __init__(self, device_type: str = "cpu") -> None:
        self._device_type = device_type
        self._stream = StubStream(device=device_type)

        self.Event = StubEvent
        self.Stream = StubStream

    def is_available(self) -> bool:
        """Check whether the device backend is available.

        Returns:
            False always, since this is a CPU-only stub.
        """
        return False

    def init(self) -> None:
        """Initialize the device backend.

        No-op in the stub implementation.
        """
        return None

    def device(self, device: Any = None) -> Any:
        """Return a context manager that sets the current device.

        Args:
            device: The device to select. Ignored in the stub.

        Returns:
            A no-op context manager.
        """
        return nullcontext()

    def current_stream(self, device: Any = None) -> StubStream:
        """Return the current stream for the given device.

        Args:
            device: The device to query. Ignored in the stub.

        Returns:
            The current StubStream.
        """
        return self._stream

    def default_stream(self, device: Any = None) -> StubStream:
        """Return the default stream for the given device.

        Args:
            device: The device to query. Ignored in the stub.

        Returns:
            The default StubStream.
        """
        return self._stream

    def stream(self, stream: StubStream | None = None) -> Any:
        """Return a context manager that sets the active stream.

        Args:
            stream: The stream to activate. If None, uses the
                current stream.

        Returns:
            A context manager yielding the active StubStream.
        """
        return nullcontext(stream or self._stream)

    def synchronize(self, device: Any = None) -> None:
        """Wait for all streams on the given device to complete.

        Args:
            device: The device to synchronize. Ignored in the stub.
        """
        return None

    def set_stream(self, stream: StubStream) -> None:
        """Set the current stream.

        Args:
            stream: The stream to make current.
        """
        self._stream = stream

    def device_count(self) -> int:
        """Return the number of available devices.

        Returns:
            1 always for the stub.
        """
        return 1

    def current_device(self) -> int:
        """Return the index of the currently selected device.

        Returns:
            0 always for the stub.
        """
        return 0

    def set_device(self, device: Any) -> None:
        """Select the given device.

        Args:
            device: The device index or identifier to select.
                Ignored in the stub.
        """
        return None

    def get_device_properties(self, device: Any = 0) -> StubDeviceProperties:
        """Return device properties for the given device.

        Args:
            device: The device index or identifier to query.
                Ignored in the stub.

        Returns:
            A StubDeviceProperties instance with default values.
        """
        return StubDeviceProperties()

    def empty_cache(self) -> None:
        """Release all unoccupied cached memory.

        No-op in the stub implementation.
        """
        return None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"StubCPUDevice does not implement '{name}'")

    def __repr__(self) -> str:
        return f"StubCPUDevice(device_type={self._device_type})"
