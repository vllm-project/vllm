# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module

_EXPORT_TO_MODULE = {
    "AdHocMemoryAllocator": "ad_hoc_memory_allocator",
    "BufferAllocator": "buffer_allocator",
    "CuFileMemoryAllocator": "cu_file_memory_allocator",
    "DevDaxMemoryAllocator": "devdax_memory_allocator",
    "GPUMemoryAllocator": "gpu_memory_allocator",
    "HipFileMemoryAllocator": "hip_file_memory_allocator",
    "HostMemoryAllocator": "host_memory_allocator",
    "LazyMemoryAllocator": "lazy_memory_allocator",
    "MixedMemoryAllocator": "mixed_memory_allocator",
    "PagedCpuGpuMemoryAllocator": "paged_cpu_gpu_memory_allocator",
    "PagedTensorMemoryAllocator": "paged_tensor_memory_allocator",
    "PinMemoryAllocator": "pin_memory_allocator",
    "TensorMemoryAllocator": "tensor_memory_allocator",
}


def __getattr__(name: str) -> object:
    """Lazily re-export allocator classes without eager package imports.

    Args:
        name: Exported allocator or helper name.

    Returns:
        The requested object from its concrete allocator module.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_TO_MODULE)
