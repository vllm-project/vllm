# SPDX-License-Identifier: Apache-2.0
"""Base class for platform device specification.

Each accelerator sub-package (``platform/cuda``, ``platform/musa``, ...)
provides a concrete :class:`DeviceSpec` subclass that describes how to
detect the device and which ops backend to load.

The :mod:`~lmcache.v1.platform` module discovers these
subclasses automatically at import time via ``pkgutil.iter_modules``:
it imports each sub-package, inspects its module namespace for
:class:`DeviceSpec` subclasses, instantiates them, and uses the
resulting objects for device detection and backend selection.

No manual registration (e.g. ``DEVICE_SPEC = ...``) is required --
simply defining the subclass in the sub-package's ``__init__.py`` is
enough.

:class:`DeviceSpec` itself is instantiable and doubles as the fallback
implementation used when no accelerator sub-package matches the
detected device: all capabilities default to a safe "no-op / False"
behaviour, and ``device_type`` / ``torch_module_name`` default to an
empty string (concrete backends -- including CPU via
:class:`~lmcache.v1.platform.cpu.CpuDeviceSpec` -- override them).
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class DeviceSpec:
    """Description of a hardware accelerator backend.

    Subclasses override the properties / methods below to describe a
    concrete accelerator.  Defining a concrete subclass in a platform
    sub-package's ``__init__.py`` is sufficient for auto-discovery.

    Instantiating :class:`DeviceSpec` directly yields the fallback
    implementation with "no-op / all False" semantics -- this is the
    behaviour used for CPU-only or unknown device types.
    """

    # Cached pin-memory backend instance (lazy-initialized).
    _pin_backend_cache: PinMemoryBackend | None = None
    # Cached DeviceOps singleton instance (lazy-initialized).
    _ops_cache: DeviceOps | None = None

    @property
    def device_type(self) -> str:
        """Device type string (e.g. ``"cuda"``, ``"musa"``, ``"mlu"``).

        Concrete backends override this; the base returns an empty
        string so a bare ``DeviceSpec()`` instance is never mistaken
        for a real accelerator (CPU is represented by
        :class:`~lmcache.v1.platform.cpu.CpuDeviceSpec`).
        """
        return ""

    @property
    def torch_module_name(self) -> str:
        """Attribute name on the ``torch`` package for the device module.

        For example, ``"cuda"`` corresponds to ``torch.cuda``.  The
        base returns an empty string; concrete backends (including
        :class:`~lmcache.v1.platform.cpu.CpuDeviceSpec`) override it.
        """
        return ""

    @property
    def ops_cls(self) -> type[DeviceOps]:
        """DeviceOps subclass providing the ``lmcache.c_ops`` surface.

        Lazy by design: the import happens on *access*, not at class-definition
        or DeviceSpec-discovery time, so resolving a spec never drags the torch
        baseline (or a native .so) into the platform package's import graph.
        The base returns the torch/CPU baseline; accelerator specs override.

        Returns:
            type[DeviceOps]: The base DeviceOps torch/CPU baseline class for the
            fallback spec. Accelerator subclasses override this property to
            return their backend-specific DeviceOps subclass.
        """
        # First Party
        from lmcache.v1.platform.base.device_ops import DeviceOps

        return DeviceOps

    def get_ops(self) -> DeviceOps:
        """Return the cached :class:`DeviceOps` singleton for this spec.

        Lazy-initialized on first access.  Calls :meth:`ensure_native`
        so native ops are bound before the instance is used.  The same
        instance is reused process-wide for a given spec.
        """
        ops = self._ops_cache
        if ops is None:
            ops = self.ops_cls()
            ops.ensure_native()
            self._ops_cache = ops
        return ops

    def is_available(self) -> bool:
        """Return ``True`` when the device is usable on this system.

        This method must NOT import from ``lmcache.__init__`` to avoid
        circular dependencies.  Use ``import torch`` directly instead.
        The fallback implementation always returns ``False`` so that
        auto-detection never picks it up.
        """
        return False

    def is_handle_transfer_available(self) -> bool:
        """Return ``True`` when the device is usable for handle transfer."""
        # TODO(chunxiaozheng): implement on subclasses
        return True

    @property
    def event_ipc_backend(self) -> "EventIPCBackend | None":
        """Return the device-event IPC backend for this device, if supported.

        Concrete device specifications must explicitly provide an event IPC
        backend when they support cross-process event synchronization.

        Returns:
            The device's ``EventIPCBackend``, or ``None`` when unsupported.
        """
        return None

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """PinMemoryBackend subclass for this device, or None for default.

        Subclasses that support host-memory pinning should override this
        property and return the appropriate backend class.  Use a lazy
        import inside the property body to avoid heavy imports at class
        definition time.
        """
        return None

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """:class:`DeviceIPCWrapper` subclass used to ship KV tensors across
        the multiprocess wire for this device, or ``None`` when the
        device has no IPC wrapper support.

        Subclasses that participate in the LMCache multiprocess KV
        transfer path override this property and return their concrete
        wrapper class.  Use a lazy import inside the property body to
        avoid dragging accelerator-specific modules into the platform
        base import graph.

        The wrapper class is expected to expose a ``wrap(tensor)``
        classmethod that returns a serializable
        :class:`DeviceIPCWrapper` instance ready for the wire.
        """
        return None

    def _get_pin_backend(self) -> PinMemoryBackend:
        """Return the cached pin-memory backend, instantiating on first use."""
        backend = self._pin_backend_cache
        if backend is None:
            backend_cls = self.pin_memory_backend or PinMemoryBackend
            backend = backend_cls()
            self._pin_backend_cache = backend
        return backend

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region for DMA access.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Platform-specific registration flags (e.g.
                ``cudaHostRegisterDefault = 0``).

        Returns:
            True if pinning succeeded, False otherwise.
        """
        return self._get_pin_backend().pin_memory(ptr, size, flags)

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return self._get_pin_backend().unpin_memory(ptr)

    @property
    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning."""
        return self._get_pin_backend().is_pin_supported

    # ------------------------------------------------------------------
    # Cache context factory
    # ------------------------------------------------------------------

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Instantiate the ``BaseCacheContext`` implementation for this device.

        Subclasses that ship a ``cache_context`` module (e.g. ``cuda``,
        ``cpu``) must override this hook: perform a lazy import of the
        concrete subclass and forward ``*args`` / ``**kwargs`` verbatim
        so the call site in
        :func:`lmcache.v1.platform.cache_context.create_cache_context`
        stays backend-agnostic.

        The default implementation raises :class:`NotImplementedError`
        so a missing override surfaces loudly instead of silently
        falling back to the CPU path.
        """
        raise NotImplementedError(
            "DeviceSpec for device_type=%r does not provide a "
            "BaseCacheContext implementation." % self.device_type
        )
