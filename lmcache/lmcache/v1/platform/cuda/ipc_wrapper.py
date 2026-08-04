# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC wrapper implementations.

:class:`CudaIPCWrapper` handles tensors backed by PyTorch's caching
allocator (vLLM default).  :class:`RawCudaIPCWrapper` handles tensors
allocated outside PyTorch (e.g. TRT-LLM's ``cudaMalloc``'d pool).

:class:`CudaIPCWrapper` is bound to ``device_type="cuda"`` via
:attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`, so the
multiprocess adapter dispatches to it via
:func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.
:class:`RawCudaIPCWrapper` is not exposed on the spec -- callers (the
TRT-LLM adapter) instantiate it directly.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class CudaIPCWrapper(DeviceIPCWrapper):
    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "CudaIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: A CUDA tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`CudaIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        # Permute any non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride we encode across IPC reflects the physical layout.
        # Offset is preserved by the wrapper's storage_offset field.
        tensor = attempt_permute_to_contiguous_view(tensor)

        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()

        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """
        Note:
            This function may break if the accelerator is not initialized.
            We should call ``torch_dev.init()`` before using this function
            (guarded by hasattr since not all backends expose init()).
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)

        storage = torch.UntypedStorage._new_shared_cuda(  # noqa: SLF001
            device_index, *self.handle[1:]
        )

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t


class RawCudaIPCWrapper(DeviceIPCWrapper):
    """IPC wrapper for CUDA tensors allocated outside PyTorch's caching
    allocator.

    PyTorch's ``UntypedStorage._share_cuda_()`` only works for tensors
    backed by its own caching allocator. TRT-LLM publishes its KV pool
    via ``at::for_blob`` over a ``cudaMalloc``'d buffer, which raises in
    ``_share_cuda_()``. This subclass bypasses that path: it calls
    ``cudaIpcGetMemHandle`` on the raw data pointer, then reconstructs
    the tensor on the receiving side via ``cudaIpcOpenMemHandle`` plus
    a CuPy ``UnownedMemory`` → DLPack → ``torch`` round-trip.

    Sharing the ``DeviceIPCWrapper`` base (rather than introducing a
    parallel class with its own msgspec ext code) is load-bearing —
    msgspec does not support unions of custom ext-encoded types. With a
    common base, ``KVCache = list[DeviceIPCWrapper]`` type-checks, the
    single ext code 1 round-trips every wrapper, and pickle preserves
    the concrete subclass identity through the wire so ``to_tensor``
    dispatches correctly.
    """

    #: Same ``torch.device.type`` as ``CudaIPCWrapper``, but not exposed
    #: on :attr:`~lmcache.v1.platform.cuda.CudaDeviceSpec.ipc_wrapper_cls`
    #: -- callers (TRT-LLM adapter) instantiate it directly.
    device_type: ClassVar[str] = "cuda"

    def __init__(self, tensor: torch.Tensor) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        assert_contiguous(tensor)

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        data_ptr = tensor.data_ptr()
        err, ipc_handle = cudart.cudaIpcGetMemHandle(data_ptr)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed: {err} (ptr=0x{data_ptr:x})"
            )

        # Store only what's needed for reconstruction.
        self._ipc_handle_reserved = bytes(ipc_handle.reserved)
        self._nbytes = tensor.untyped_storage().nbytes()

        # DeviceIPCWrapper interface fields. ``handle`` is unused —
        # ``to_tensor`` is overridden to bypass it — but kept (None) so
        # the base-class equality check has a value to compare.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via raw CUDA IPC."""
        # Third Party
        import cupy

        try:
            # Third Party
            from cuda.bindings import runtime as cudart
        except ImportError:
            # Third Party
            from cuda import cudart

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        handle = cudart.cudaIpcMemHandle_t()
        handle.reserved = self._ipc_handle_reserved
        err, ptr = cudart.cudaIpcOpenMemHandle(
            handle, cudart.cudaIpcMemLazyEnablePeerAccess
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed: {err}")

        # Wrap as a flat ``uint8`` CuPy array, DLPack to torch, then view
        # as the original dtype/shape. ``uint8`` avoids dtype-conversion
        # gaps (bfloat16, fp8 have no direct CuPy/NumPy equivalent without
        # ml_dtypes).
        with cupy.cuda.Device(device_index):
            mem = cupy.cuda.UnownedMemory(ptr, self._nbytes, owner=self)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            cp_flat = cupy.ndarray(self._nbytes, dtype=cupy.uint8, memptr=memptr)

        raw = torch.from_dlpack(cp_flat)
        return raw.view(self.dtype).reshape(self.shape)
