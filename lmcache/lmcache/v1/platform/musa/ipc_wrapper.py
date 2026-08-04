# SPDX-License-Identifier: Apache-2.0
"""Optional TorchMUSA IPC wrapper for MP handle transfer.

The MUSA handle path is deliberately fail-closed. This module exposes a
``DeviceIPCWrapper`` subclass for platform discovery, but it only constructs
wrappers when the user explicitly enables the path and TorchMUSA exposes the
required memory IPC APIs. Event IPC and server-side block transfer are checked
independently because they belong to the later full transfer path.
"""

# Standard
from typing import ClassVar, Protocol, cast
import importlib
import inspect
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.contiguity import (
    attempt_permute_to_contiguous_view,
)
from lmcache.v1.gpu_connector.utils import assert_contiguous
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

ENV_MUSA_HANDLE_TRANSFER = "LMCACHE_MUSA_HANDLE_TRANSFER"

_REQUIRED_TORCH_MUSA_IPC_SYMBOLS = (
    "export_tensor",
    "open_tensor",
)

_REQUIRED_TORCH_MUSA_EVENT_SYMBOLS = (
    "ipc_handle",
    "record",
    "wait",
    "query",
    "synchronize",
)


class _TorchMusaIPCOwner(Protocol):
    """Receiver-side owner returned by ``torch.musa.ipc.open_tensor``."""

    @property
    def tensor(self) -> torch.Tensor:
        """Return the imported tensor while the owner is open."""
        ...

    def close(self) -> None:
        """Release the receiver-side IPC mapping."""
        ...


class _TorchMusaIPCModule(Protocol):
    """Public ``torch.musa.ipc`` surface required by LMCache."""

    def export_tensor(self, tensor: torch.Tensor) -> object:
        """Export a MUSA tensor as a process-portable IPC handle."""
        ...

    def open_tensor(
        self,
        handle: object,
        *,
        device: object | None = None,
        stream: object | None = None,
    ) -> _TorchMusaIPCOwner:
        """Open a MUSA tensor handle in the current process."""
        ...


def is_musa_handle_transfer_enabled() -> bool:
    """Return whether users opted into the experimental MUSA handle path."""
    return os.environ.get(ENV_MUSA_HANDLE_TRANSFER, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_torch_musa_module() -> object | None:
    """Return the TorchMUSA module when this PyTorch build exposes it.

    Returns:
        ``torch.musa`` when present, otherwise ``None``.
    """
    if not hasattr(torch, "musa"):
        # torch_musa registers torch.musa as an import side effect.
        try:
            importlib.import_module("torch_musa")
        except Exception:
            # Optional native dependencies can fail with ImportError, OSError,
            # or a backend-specific initialization exception.
            pass
    if not hasattr(torch, "musa"):
        return None
    return torch.musa  # type: ignore[attr-defined]


def is_torch_musa_available() -> bool:
    """Return whether TorchMUSA is importable and reports a visible device."""
    torch_musa = get_torch_musa_module()
    if torch_musa is None:
        return False
    is_available = getattr(torch_musa, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except Exception:
        return False


def check_torch_musa_ipc_support(torch_musa: object) -> bool:
    """Return whether TorchMUSA exposes the required memory IPC API.

    Args:
        torch_musa: Candidate ``torch.musa`` module.

    Returns:
        ``True`` when ``torch.musa.ipc`` provides tensor export and import.
    """
    ipc_module = getattr(torch_musa, "ipc", None)
    return all(
        callable(getattr(ipc_module, symbol, None))
        for symbol in _REQUIRED_TORCH_MUSA_IPC_SYMBOLS
    )


def check_torch_musa_event_support(torch_musa: object) -> bool:
    """Return whether TorchMUSA exposes the required event IPC API.

    Args:
        torch_musa: Candidate ``torch.musa`` module.

    Returns:
        ``True`` when TorchMUSA events support interprocess construction,
        import, export, record, wait, query, and synchronization.
    """
    event_cls = getattr(torch_musa, "Event", None)
    if event_cls is None or not callable(getattr(event_cls, "from_ipc_handle", None)):
        return False
    supports_interprocess = _has_interprocess_parameter(
        event_cls
    ) or _has_interprocess_parameter(getattr(event_cls, "__new__", None))
    if not supports_interprocess:
        supports_interprocess = _can_create_interprocess_event(event_cls)
    return supports_interprocess and all(
        callable(getattr(event_cls, symbol, None))
        for symbol in _REQUIRED_TORCH_MUSA_EVENT_SYMBOLS
    )


def is_musa_block_transfer_available() -> bool:
    """Return whether server-side MUSA block transfer is production-ready.

    Memory and event IPC can be validated independently. Forced MUSA handle
    mode remains unavailable until the server-side MUSA block-transfer
    primitive is implemented and validated.

    Returns:
        ``False`` while the server-side block-transfer path is unavailable.
    """
    return False


def is_musa_memory_ipc_available() -> bool:
    """Return whether the opt-in TorchMUSA memory IPC API is available.

    Returns:
        ``True`` when the feature is enabled and TorchMUSA provides tensor
        export and import.
    """
    if not is_musa_handle_transfer_enabled() or not is_torch_musa_available():
        return False
    torch_musa = get_torch_musa_module()
    return torch_musa is not None and check_torch_musa_ipc_support(torch_musa)


def is_musa_event_ipc_available() -> bool:
    """Return whether the opt-in TorchMUSA event IPC API is available.

    Returns:
        ``True`` when the feature is enabled and TorchMUSA provides all event
        operations required by :class:`MusaEventIPCBackend`.
    """
    if not is_musa_handle_transfer_enabled() or not is_torch_musa_available():
        return False
    torch_musa = get_torch_musa_module()
    return torch_musa is not None and check_torch_musa_event_support(torch_musa)


def is_musa_handle_transfer_available() -> bool:
    """Return whether forced MUSA MP handle mode may be selected.

    Returns:
        ``True`` only when memory IPC, event IPC, and server-side block
        transfer are all available.
    """
    return (
        is_musa_memory_ipc_available()
        and is_musa_event_ipc_available()
        and is_musa_block_transfer_available()
    )


class MusaIPCWrapper(DeviceIPCWrapper):
    """Wire-compatible wrapper for TorchMUSA memory IPC handles."""

    device_type: ClassVar[str] = "musa"

    _opened_handle: _TorchMusaIPCOwner | None

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "MusaIPCWrapper":
        """Create a wire wrapper for a MUSA tensor.

        Args:
            tensor: MUSA tensor to export.

        Returns:
            A wrapper carrying the TorchMUSA tensor IPC handle.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        """Export a contiguous MUSA tensor through TorchMUSA IPC.

        Args:
            tensor: A contiguous MUSA tensor to export.

        Raises:
            ValueError: If ``tensor`` is not a MUSA tensor or cannot be
                represented as a contiguous view.
            RuntimeError: If the opt-in TorchMUSA memory IPC capability is
                unavailable.
        """
        tensor_view = attempt_permute_to_contiguous_view(tensor)
        if (
            not isinstance(tensor_view, torch.Tensor)
            or tensor_view.device.type != "musa"
        ):
            raise ValueError("expected a MUSA tensor for MusaIPCWrapper")
        tensor = tensor_view
        assert_contiguous(tensor)

        ipc_module = _torch_musa_ipc_module_if_ready()
        if ipc_module is None:
            raise RuntimeError(
                "MUSA memory IPC is not available. Set "
                f"{ENV_MUSA_HANDLE_TRANSFER}=1 with compatible TorchMUSA "
                "memory IPC support, or use MP transfer mode "
                "'engine_driven' or 'auto'."
            )

        self.handle = ipc_module.export_tensor(tensor)
        self._opened_handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(
            0 if device_index is None else device_index
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the MUSA tensor in the current process.

        Returns:
            The imported tensor owned by this wrapper.

        Raises:
            RuntimeError: If the opt-in TorchMUSA memory IPC capability is
                unavailable in the receiver.
        """
        owner = self._opened_handle
        if owner is None:
            ipc_module = _torch_musa_ipc_module_if_ready()
            if ipc_module is None:
                raise RuntimeError(
                    "MUSA memory IPC is not available in the receiver process."
                )
            # The TorchMUSA handle carries its producer ordinal and UUID;
            # open_tensor resolves them against receiver-visible devices.
            owner = ipc_module.open_tensor(self.handle)
            self._opened_handle = owner
        return owner.tensor

    def close(self) -> None:
        """Release the receiver-side TorchMUSA tensor owner, if opened.

        Calling this method more than once has no effect.
        """
        owner = self._opened_handle
        if owner is None:
            return
        owner.close()
        self._opened_handle = None

    def __getstate__(self) -> dict[str, object]:
        """Return transport state without receiver-local owner state."""
        state = self.__dict__.copy()
        state.pop("_opened_handle", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore transport state with no receiver-local open owner."""
        self.__dict__.update(state)
        self._opened_handle = None

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Return a stable MUSA device identifier for ``device_index``."""
        props = torch.musa.get_device_properties(device_index)  # type: ignore[attr-defined]
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            return str(uuid)
        pci_bus_id = getattr(props, "pci_bus_id", None)
        if pci_bus_id is not None:
            return str(pci_bus_id)
        name = getattr(props, "name", "musa")
        return f"{name}:{device_index}"


def _torch_musa_ipc_module_if_ready() -> _TorchMusaIPCModule | None:
    """Return ``torch.musa.ipc`` when memory IPC is available."""
    if not is_musa_memory_ipc_available():
        return None
    module = get_torch_musa_module()
    if module is None:
        return None
    ipc_module = getattr(module, "ipc", None)
    if ipc_module is None:
        return None
    return cast(_TorchMusaIPCModule, ipc_module)


def _has_interprocess_parameter(obj: object) -> bool:
    """Return whether ``obj`` accepts the ``interprocess`` parameter."""
    if not callable(obj):
        return False
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    return "interprocess" in signature.parameters


def _can_create_interprocess_event(event_cls: object) -> bool:
    """Return whether an opaque Event binding accepts interprocess events.

    Some C/pybind-backed classes do not expose a useful signature even though
    their public constructor supports ``interprocess=True``. The probe keeps
    the capability check aligned with the operation used by the backend.
    """
    if not callable(event_cls):
        return False
    try:
        event_cls(interprocess=True)  # type: ignore[operator]
    except Exception:
        return False
    return True
