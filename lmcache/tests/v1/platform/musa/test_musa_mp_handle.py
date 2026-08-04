# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import NoReturn, cast
import importlib

# Third Party
import pytest
import torch


class _FakeMusaEvent:
    """Minimal TorchMUSA Event facade for capability-gate tests."""

    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess

    @classmethod
    def from_ipc_handle(cls, _device: object, _handle: bytes) -> "_FakeMusaEvent":
        """Reconstruct an event from an IPC handle."""
        return cls(interprocess=True)

    def ipc_handle(self) -> bytes:
        """Return a fake process-portable event handle."""
        return b"event"

    def record(self, _stream: object | None = None) -> None:
        """Record the fake event."""

    def wait(self, _stream: object | None = None) -> None:
        """Wait for the fake event."""

    def query(self) -> bool:
        """Return the fake event completion state."""
        return True

    def synchronize(self) -> None:
        """Synchronize the fake event."""


class _FakeDevice:
    """Minimal device object exposing only ``type`` for factory routing tests."""

    type = "musa"


class _FakeMusaTensor:
    """Minimal tensor-like object exposing a MUSA device for routing tests."""

    device = _FakeDevice()


def _fake_musa_kv_caches() -> dict[str, torch.Tensor]:
    """Return a typed KV cache mapping backed by a minimal MUSA tensor facade."""
    return cast(dict[str, torch.Tensor], {"layer_0": _FakeMusaTensor()})


def test_musa_handle_transfer_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUSA handle transfer is opt-in and unavailable by default."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    monkeypatch.delenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, raising=False)

    assert ipc_wrapper.is_musa_handle_transfer_enabled() is False
    assert ipc_wrapper.is_musa_handle_transfer_available() is False


def test_musa_handle_transfer_requires_torch_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in alone is not enough without a visible TorchMUSA runtime."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: False)

    assert ipc_wrapper.is_musa_handle_transfer_available() is False


def test_get_torch_musa_module_imports_torch_musa_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TorchMUSA may register ``torch.musa`` only after ``torch_musa`` import."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    fake_torch = SimpleNamespace()
    torch_musa = SimpleNamespace(is_available=lambda: True)

    def import_module(name: str) -> object:
        assert name == "torch_musa"
        fake_torch.musa = torch_musa
        return SimpleNamespace()

    monkeypatch.setattr(ipc_wrapper, "torch", fake_torch)
    monkeypatch.setattr(ipc_wrapper.importlib, "import_module", import_module)

    assert ipc_wrapper.get_torch_musa_module() is torch_musa
    assert ipc_wrapper.is_torch_musa_available() is True


def test_get_torch_musa_module_import_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TorchMUSA native-extension load failure is treated as unavailable."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    def raise_loader_error(_name: str) -> object:
        raise OSError("missing TorchMUSA shared object")

    monkeypatch.setattr(ipc_wrapper, "torch", SimpleNamespace())
    monkeypatch.setattr(ipc_wrapper.importlib, "import_module", raise_loader_error)

    assert ipc_wrapper.get_torch_musa_module() is None


def test_musa_memory_ipc_does_not_enable_handle_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory IPC alone is not enough to enable the MUSA handle path."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(
            export_tensor=lambda _tensor: object(),
            open_tensor=lambda _handle, **_kwargs: object(),
        ),
    )

    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc_wrapper, "get_torch_musa_module", lambda: torch_musa)

    assert ipc_wrapper.check_torch_musa_ipc_support(torch_musa) is True
    assert ipc_wrapper.check_torch_musa_event_support(torch_musa) is False
    assert ipc_wrapper.is_musa_memory_ipc_available() is True
    assert ipc_wrapper.is_musa_event_ipc_available() is False
    assert ipc_wrapper.is_musa_handle_transfer_available() is False


def test_musa_ipc_components_do_not_require_block_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory and event IPC remain independently testable."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(
            export_tensor=lambda _tensor: object(),
            open_tensor=lambda _handle, **_kwargs: object(),
        ),
        Event=_FakeMusaEvent,
    )

    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc_wrapper, "get_torch_musa_module", lambda: torch_musa)
    monkeypatch.setattr(ipc_wrapper, "is_musa_block_transfer_available", lambda: False)

    assert ipc_wrapper.is_musa_memory_ipc_available() is True
    assert ipc_wrapper.is_musa_event_ipc_available() is True
    assert ipc_wrapper.is_musa_handle_transfer_available() is False


def test_musa_handle_transfer_available_when_all_capabilities_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced MUSA handle mode is available only after all gates pass."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(
            export_tensor=lambda _tensor: object(),
            open_tensor=lambda _handle, **_kwargs: object(),
        ),
        Event=_FakeMusaEvent,
    )

    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc_wrapper, "get_torch_musa_module", lambda: torch_musa)
    monkeypatch.setattr(ipc_wrapper, "is_musa_block_transfer_available", lambda: True)

    assert ipc_wrapper.check_torch_musa_ipc_support(torch_musa) is True
    assert ipc_wrapper.check_torch_musa_event_support(torch_musa) is True
    assert ipc_wrapper.is_musa_handle_transfer_available() is True


def test_musa_handle_transfer_rejects_incomplete_torch_musa_runtime() -> None:
    """Missing TorchMUSA IPC symbols keep MUSA handle mode disabled."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(export_tensor=lambda _tensor: object())
    )

    assert ipc_wrapper.check_torch_musa_ipc_support(torch_musa) is False


def test_musa_handle_transfer_rejects_incomplete_event_runtime() -> None:
    """Missing event operations keep MUSA handle mode disabled."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    incomplete_event = type(
        "IncompleteMusaEvent",
        (_FakeMusaEvent,),
        {"query": None},
    )

    assert (
        ipc_wrapper.check_torch_musa_event_support(
            SimpleNamespace(Event=incomplete_event)
        )
        is False
    )


def test_musa_event_capability_probe_handles_opaque_event_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque Event binding is checked by probing its public constructor."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    def raise_signature(_obj: object) -> object:
        raise ValueError("opaque binding")

    monkeypatch.setattr(
        ipc_wrapper.inspect,
        "signature",
        raise_signature,
    )

    assert ipc_wrapper.check_torch_musa_event_support(
        SimpleNamespace(Event=_FakeMusaEvent)
    )


def test_musa_platform_discovers_factory_and_registers_capability_predicate() -> None:
    """The MUSA wrapper is auto-discovered while availability stays explicit."""
    # First Party
    from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
    import lmcache.v1.platform.musa as musa_platform

    importlib.reload(musa_platform)

    factory = resolve_kv_wrapper_factory("musa")

    device_spec = get_device_spec("musa")
    assert device_spec is not None
    assert device_spec.is_handle_transfer_available() is False
    assert callable(factory)
    with pytest.raises(ValueError, match="expected a MUSA tensor"):
        factory(torch.empty(1))


def test_create_transfer_context_auto_keeps_musa_on_data_path() -> None:
    """MUSA auto mode remains on the engine-driven data path."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context(_fake_musa_kv_caches())

    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_musa_handle_requires_capability() -> None:
    """Forced MUSA handle mode fails closed when its capability is absent."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context

    with pytest.raises(ValueError, match="not available"):
        create_transfer_context(_fake_musa_kv_caches(), mode="lmcache_driven")


def test_create_transfer_context_musa_handle_allowed_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced MUSA handle mode is allowed once the platform reports capability."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.platform import get_device_spec

    musa_spec = get_device_spec("musa")
    assert musa_spec is not None
    # Force the transport-capability gate open without touching
    # ``ipc_wrapper_cls`` -- the real MusaIPCWrapper already binds it.
    monkeypatch.setattr(musa_spec, "is_handle_transfer_available", lambda: True)

    context = create_transfer_context(
        _fake_musa_kv_caches(),
        mode="lmcache_driven",
    )

    assert isinstance(context, LMCacheDrivenTransferContext)


def test_musa_ipc_wrapper_rejects_non_musa_tensor() -> None:
    """The MUSA IPC wrapper never accepts CPU tensors by accident."""
    # First Party
    from lmcache.v1.platform.musa.ipc_wrapper import MusaIPCWrapper

    with pytest.raises(ValueError, match="expected a MUSA tensor"):
        MusaIPCWrapper(torch.empty(4))


def test_musa_ipc_wrapper_uses_device_agnostic_wire_base() -> None:
    """MUSA handles use the device-agnostic KVCache wire base."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper, KVCache
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper
    from lmcache.v1.platform.musa.ipc_wrapper import MusaIPCWrapper

    assert issubclass(MusaIPCWrapper, DeviceIPCWrapper)
    assert not issubclass(MusaIPCWrapper, CudaIPCWrapper)
    assert MusaIPCWrapper.device_type == "musa"
    assert KVCache == list[DeviceIPCWrapper]


def test_musa_ipc_wrapper_does_not_serialize_receiver_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire payload excludes receiver-local TorchMUSA owner state."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
    from lmcache.v1.platform.musa import ipc_wrapper

    class _Owner:
        tensor = torch.empty(1)

        def close(self) -> None:
            pass

        def __reduce__(self) -> NoReturn:
            raise TypeError("receiver owner is process-local")

    open_calls = 0

    def open_tensor(_handle: object, **_kwargs: object) -> _Owner:
        nonlocal open_calls
        open_calls += 1
        return _Owner()

    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(
            export_tensor=lambda _tensor: object(),
            open_tensor=open_tensor,
        )
    )
    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc_wrapper, "get_torch_musa_module", lambda: torch_musa)

    wrapper = object.__new__(ipc_wrapper.MusaIPCWrapper)
    wrapper.__setstate__(
        {
            "handle": SimpleNamespace(token="handle"),
            "dtype": torch.float32,
            "shape": (1,),
            "stride": (1,),
            "storage_offset": 0,
            "device_uuid": "musa:0",
        }
    )
    assert wrapper.to_tensor() is _Owner.tensor

    restored = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(wrapper))

    assert restored == wrapper
    assert isinstance(restored, ipc_wrapper.MusaIPCWrapper)
    assert restored.to_tensor() is _Owner.tensor
    assert open_calls == 2


def test_musa_ipc_wrapper_closes_receiver_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the wrapper releases its receiver-side TorchMUSA owner."""
    # First Party
    from lmcache.v1.platform.musa import ipc_wrapper

    class _Owner:
        def __init__(self) -> None:
            self.tensor = torch.empty(1)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    owner = _Owner()
    torch_musa = SimpleNamespace(
        ipc=SimpleNamespace(
            export_tensor=lambda _tensor: object(),
            open_tensor=lambda _handle, **_kwargs: owner,
        )
    )
    monkeypatch.setenv(ipc_wrapper.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc_wrapper, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc_wrapper, "get_torch_musa_module", lambda: torch_musa)

    wrapper = object.__new__(ipc_wrapper.MusaIPCWrapper)
    wrapper.__setstate__({"handle": object()})

    assert wrapper.to_tensor() is owner.tensor
    wrapper.close()
    wrapper.close()

    assert owner.close_calls == 1
