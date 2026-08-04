# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.musa import MusaDeviceSpec, ipc_wrapper
from lmcache.v1.platform.musa.event_ipc import MusaEventIPCBackend


class _UnavailableIPC:
    def is_musa_event_ipc_available(self) -> bool:
        return False


class _AvailableIPC:
    """Fake MUSA IPC wrapper exposing a TorchMUSA-style Event API."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.torch_musa_module_calls = 0

    def is_musa_event_ipc_available(self) -> bool:
        return True

    def get_torch_musa_module(self) -> object:
        self.torch_musa_module_calls += 1
        outer = self

        class _Event:
            def __init__(
                self,
                interprocess: bool = False,
                *,
                imported: bool = False,
            ) -> None:
                if not imported:
                    outer.calls.append(("create", interprocess))

            @classmethod
            def from_ipc_handle(cls, device: object, handle: bytes) -> "_Event":
                outer.calls.append(("import", device, handle))
                return cls(imported=True)

            def ipc_handle(self) -> bytes:
                outer.calls.append(("export", self))
                return b"musa-handle"

            def record(self, stream: object | None = None) -> None:
                outer.calls.append(("record", self, stream))

            def wait(self, stream: object | None = None) -> None:
                outer.calls.append(("wait", self, stream))

            def query(self) -> bool:
                outer.calls.append(("query", self))
                return True

            def synchronize(self) -> None:
                outer.calls.append(("synchronize", self))

        class _MusaModule:
            Event = _Event

        return _MusaModule()


class _Device:
    def __init__(self, type_: str, index: int | None = None) -> None:
        self.type = type_
        self.index = index


def test_musa_backend_is_event_ipc_backend() -> None:
    """MUSA exposes the generic event IPC contract."""
    assert isinstance(
        MusaEventIPCBackend(ipc_module=_UnavailableIPC()), EventIPCBackend
    )
    assert MusaEventIPCBackend(ipc_module=_UnavailableIPC()).device_type == "musa"


def test_check_event_support_fails_closed_when_unavailable() -> None:
    """Unavailable MUSA event IPC fails before event creation."""
    backend = MusaEventIPCBackend(ipc_module=_UnavailableIPC())
    with pytest.raises(RuntimeError, match="musa"):
        backend.check_event_support(_Device("musa", 0))


def test_check_event_support_fails_closed_when_module_missing() -> None:
    """A missing TorchMUSA module fails the capability check."""

    class _AvailableWithoutModule:
        def is_musa_event_ipc_available(self) -> bool:
            return True

    backend = MusaEventIPCBackend(ipc_module=_AvailableWithoutModule())
    with pytest.raises(RuntimeError, match="TorchMUSA"):
        backend.check_event_support(_Device("musa", 0))


def test_event_operations_delegate_to_torch_musa() -> None:
    """MUSA operations use the TorchMUSA interprocess Event API."""
    ipc = _AvailableIPC()
    backend = MusaEventIPCBackend(ipc_module=ipc)
    device = _Device("musa", 1)

    backend.check_event_support(device)
    event = backend.create_event(device)
    assert backend.export_event(event, device) == b"musa-handle"
    remote = backend.import_event(b"musa-handle", device)
    backend.record_event(event, "STREAM")
    backend.wait_event(remote, "STREAM")
    assert backend.query_event(remote) is True
    backend.synchronize_event(remote, device)

    assert [call[0] for call in ipc.calls] == [
        "create",
        "export",
        "import",
        "record",
        "wait",
        "query",
        "synchronize",
    ]
    assert ("import", device, b"musa-handle") in ipc.calls
    assert ipc.torch_musa_module_calls == 1


def test_default_backend_loads_ipc_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production backend resolves the current MUSA IPC wrapper module."""
    ipc = _AvailableIPC()
    monkeypatch.setattr(
        ipc_wrapper,
        "is_musa_event_ipc_available",
        ipc.is_musa_event_ipc_available,
    )
    monkeypatch.setattr(
        ipc_wrapper,
        "get_torch_musa_module",
        ipc.get_torch_musa_module,
    )

    backend = MusaEventIPCBackend()
    backend.check_event_support(_Device("musa", 0))
    event = backend.create_event(_Device("musa", 0))

    assert backend.export_event(event, _Device("musa", 0)) == b"musa-handle"
    assert ipc.torch_musa_module_calls == 1


def test_musa_device_spec_returns_musa_backend() -> None:
    """The MUSA device specification owns its event backend."""
    backend = MusaDeviceSpec().event_ipc_backend
    assert isinstance(backend, MusaEventIPCBackend)
