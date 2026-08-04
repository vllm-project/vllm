# SPDX-License-Identifier: Apache-2.0
# Standard
import inspect

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cpu import CpuDeviceSpec
from lmcache.v1.platform.cuda import CudaDeviceSpec
import lmcache.v1.platform as platform


class _FakeEvent:
    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess
        self.calls: list[tuple] = []

    def ipc_handle(self) -> bytes:
        return b"handle-bytes"

    @classmethod
    def from_ipc_handle(cls, device, handle):
        ev = cls(interprocess=True)
        ev.calls.append(("from_ipc_handle", device, handle))
        return ev

    def record(self, stream=None):
        self.calls.append(("record", stream))

    def wait(self, stream=None):
        self.calls.append(("wait", stream))

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))


class _FakeEventModule:
    Event = _FakeEvent


class _NoInterprocessEvent:
    def __init__(self) -> None:  # no interprocess parameter
        pass

    @classmethod
    def from_ipc_handle(cls, device, handle):
        return cls()


class _NoInterprocessModule:
    Event = _NoInterprocessEvent


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


class _FakeDeviceSpec:
    def __init__(self, backend: EventIPCBackend | None) -> None:
        self.event_ipc_backend = backend


def test_default_backend_create_export_import_delegate():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    event = backend.create_event(_Device("fake"))
    assert isinstance(event, _FakeEvent) and event.interprocess is True
    assert backend.export_event(event, _Device("fake")) == b"handle-bytes"
    imported = backend.import_event(b"h", _Device("fake"))
    assert (
        "from_ipc_handle",
        _is_device := imported.calls[0][1],
        b"h",
    ) == imported.calls[0]


def test_default_backend_record_wait_query_synchronize_delegate():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    event = backend.create_event(_Device("fake"))
    backend.record_event(event, "STREAM")
    backend.wait_event(event, "STREAM")
    assert backend.query_event(event) is True
    backend.synchronize_event(event, _Device("fake"))
    assert ("record", "STREAM") in event.calls
    assert ("wait", "STREAM") in event.calls
    assert ("synchronize",) in event.calls


def test_check_event_support_raises_with_device_name_when_no_interprocess():
    backend = DefaultEventIPCBackend(
        event_module=_NoInterprocessModule(), device_type="weirddev"
    )
    with pytest.raises(RuntimeError, match="weirddev"):
        backend.check_event_support(_Device("weirddev"))


def test_check_event_support_passes_for_capable_module():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    backend.check_event_support(_Device("fake"))  # must not raise


def test_get_event_ipc_backend_resolves_cpu_to_default():
    backend = get_event_ipc_backend(_Device("cpu"))
    assert isinstance(backend, EventIPCBackend)
    assert isinstance(backend, DefaultEventIPCBackend)


def test_get_event_ipc_backend_requires_device() -> None:
    signature = inspect.signature(get_event_ipc_backend)
    assert signature.parameters["device"].default is inspect.Parameter.empty


def test_get_event_ipc_backend_accepts_string_device_type():
    assert isinstance(get_event_ipc_backend("cpu"), DefaultEventIPCBackend)


@pytest.mark.parametrize(
    ("device", "expected_device_type"),
    [
        pytest.param(0, torch_device_type, id="integer-index"),
        pytest.param(2, torch_device_type, id="nonzero-integer-index"),
        pytest.param("cuda:0", "cuda", id="indexed-cuda-string"),
        pytest.param("musa:3", "musa", id="indexed-musa-string"),
    ],
)
def test_get_event_ipc_backend_normalizes_indexed_devices(
    monkeypatch: pytest.MonkeyPatch,
    device: object,
    expected_device_type: str,
) -> None:
    """Backend lookup uses the base type for integer and indexed devices."""
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    resolved_device_types: list[str] = []

    def fake_get_device_spec(device_type: str) -> _FakeDeviceSpec:
        resolved_device_types.append(device_type)
        return _FakeDeviceSpec(backend)

    monkeypatch.setattr(platform, "get_device_spec", fake_get_device_spec)

    assert get_event_ipc_backend(device) is backend
    assert resolved_device_types == [expected_device_type]


def test_get_event_ipc_backend_rejects_unregistered_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered device is a platform configuration error."""
    monkeypatch.setattr(platform, "get_device_spec", lambda device_type: None)

    with pytest.raises(
        RuntimeError,
        match="No DeviceSpec registered for device type 'unregistered'",
    ):
        get_event_ipc_backend("unregistered:0")


def test_get_event_ipc_backend_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered device must explicitly provide event IPC support."""
    monkeypatch.setattr(
        platform,
        "get_device_spec",
        lambda device_type: _FakeDeviceSpec(None),
    )

    with pytest.raises(
        RuntimeError,
        match="Device type 'unsupported' does not support event IPC",
    ):
        get_event_ipc_backend("unsupported")


def test_get_event_ipc_backend_cpu_uses_stub_when_accelerator_is_active() -> None:
    """Explicit CPU lookups do not use the active accelerator event module."""
    backend = get_event_ipc_backend("cpu")
    backend.check_event_support("cpu")
    event = backend.create_event("cpu")

    assert backend.export_event(event, "cpu") == b"stub_ipc_handle"


def test_device_spec_does_not_expose_event_backend() -> None:
    spec = DeviceSpec()
    assert spec.event_ipc_backend is None


def test_cpu_device_spec_exposes_cached_default_event_backend() -> None:
    spec = CpuDeviceSpec()
    assert isinstance(spec.event_ipc_backend, DefaultEventIPCBackend)
    assert spec.event_ipc_backend is spec.event_ipc_backend


def test_cuda_device_spec_exposes_cached_default_event_backend() -> None:
    spec = CudaDeviceSpec()
    assert isinstance(spec.event_ipc_backend, DefaultEventIPCBackend)
    assert spec.event_ipc_backend.device_type == "cuda"
    assert spec.event_ipc_backend is spec.event_ipc_backend


def test_default_backend_stub_end_to_end():
    # Uses the real active torch_dev (StubCPUDevice/StubEvent) via the default.
    backend = get_event_ipc_backend("cpu")
    backend.check_event_support("cpu")
    event = backend.create_event("cpu")
    handle = backend.export_event(event, "cpu")
    assert isinstance(handle, bytes)
    remote = backend.import_event(handle, "cpu")
    backend.record_event(event, None)
    assert backend.query_event(remote) is True
    backend.synchronize_event(remote, "cpu")


def test_protocol_signatures_match_default_backend():
    proto_methods = {
        "check_event_support",
        "create_event",
        "export_event",
        "import_event",
        "record_event",
        "wait_event",
        "query_event",
        "synchronize_event",
    }
    for name in proto_methods:
        assert callable(getattr(DefaultEventIPCBackend, name))
        assert inspect.signature(getattr(DefaultEventIPCBackend, name)) is not None
