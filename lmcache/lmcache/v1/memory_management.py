# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, wraps
from typing import Any, List, Optional, Tuple, Union
import abc
import ctypes
import os
import threading

# Third Party
from sortedcontainers import SortedList
import torch

# First Party
from lmcache import torch_dev
from lmcache import torch_device_type as torch_device_type  # noqa: F401
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.platform import current_device_spec as current_device_spec  # noqa: F401
from lmcache.v1.system_detection import NUMAMapping
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


# Cache for ctypes ubyte-array types keyed by length.
#
# ctypes does not cache `(c_ubyte * N)` array types -- each call to the `*`
# operator builds a fresh heap type via PyCArrayType_from_ctype. The heap
# type metadata stays alive forever (held by the type system), so calling
# `(ctypes.c_ubyte * N).from_address(...)` on every TensorMemoryObj.byte_array
# access leaks ~1-2 kB per call. Under long-running remote-backend put/get
# workloads this is the dominant source of monotonic anonymous-memory growth
# (see https://github.com/LMCache/LMCache/issues/3767).
#
# Caching by length is safe: the `from_address(addr)` instance never owns the
# underlying buffer, only the metadata (length, item-type), and that metadata
# depends solely on `N`. ``functools.cache`` provides a thread-safe unbounded
# memoization primitive, so concurrent first-time accesses for the same `N`
# cannot race to create distinct heap types.
@cache
def _get_cached_ubyte_array_type(num_bytes: int) -> type[ctypes.Array[ctypes.c_ubyte]]:
    """Return a cached ``ctypes.c_ubyte * num_bytes`` array type.

    Args:
        num_bytes: The length of the array type in bytes.

    Returns:
        The cached ``ctypes.Array`` subclass for the given length. Subsequent
        calls with the same ``num_bytes`` return the same type object.
    """
    return ctypes.c_ubyte * num_bytes


# Helper functions for thread safety
def synchronized(lock_attr_name):
    """
    Decorator to make a method thread-safe by acquiring the lock
    specified by lock_attr_name on the instance.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr_name)
            with lock:
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


class MemoryFormat(Enum):
    UNDEFINED = 0
    """[2, num_layers, num_tokens, hidden_dim]
    """
    # KV_BLOB = 1
    KV_2LTD = auto()
    """[num_tokens, 2, hidden_dim]
    """
    # LAYER_KV_BLOB = 2
    KV_T2D = auto()
    """[2, num_tokens, hidden_dim]
    """

    KV_2TD = auto()
    """Compressed binary array format
    """
    BINARY = auto()

    BINARY_BUFFER = auto()

    KV_MLA_FMT = auto()
    """[1, num_layers, num_tokens, aligned_head_size]
    """

    # This is for the encoder cache (EC) tensor format
    EC_TD = auto()
    """[num_tokens, hidden_dim]
    """

    # Hidden-state store (HS) tensor format. Same logical shape as EC_TD
    # ([num_tokens, hidden_dim]) but tagged separately so the allocator and
    # any future mp/serialization paths can distinguish encoder-cache entries
    # from hidden-state entries.
    HS_TD = auto()
    """[num_tokens, hidden_dim]
    """

    def token_dim(self) -> int:
        if self == MemoryFormat.KV_2LTD:
            return 2
        elif self == MemoryFormat.KV_T2D:
            return 1
        elif self == MemoryFormat.KV_2TD:
            return 0
        elif self == MemoryFormat.BINARY:
            return 0
        elif self == MemoryFormat.BINARY_BUFFER:
            return 0
        elif self == MemoryFormat.KV_MLA_FMT:
            return 2
        elif self == MemoryFormat.EC_TD:
            return 0
        elif self == MemoryFormat.HS_TD:
            return 0
        return 0


@dataclass
class FreeBlock:
    """Metadata class used by the memory allocators"""

    start: int
    size: int

    def can_be_coalesced(self, succ: "FreeBlock") -> bool:
        return self.start + self.size == succ.start


@dataclass
class MemoryObjMetadata:
    # TODO(chunxiaozheng): use shapes and dtypes to replace shape and dtype
    # The 'logical' shape of the tensor
    shape: torch.Size

    # The 'logical' dtype of the tensor
    dtype: Optional[torch.dtype]

    # The 'physical address' of the tensor
    address: int

    # The 'physical size' in bytes of the allocated memory
    phy_size: int

    # Reference count
    ref_count: int

    # Whether the object is pinned and cannot be evicted
    # lookup pins are temporary
    # cache controller pins are persistent
    pin_count: int = 0

    # The 'logical' format of the tensor
    fmt: MemoryFormat = MemoryFormat.UNDEFINED

    # Positions when the cache is stored
    cached_positions: Optional[torch.Tensor] = None

    # shapes and dtypes should be used in the future
    shapes: Optional[list[torch.Size]] = None
    dtypes: Optional[list[torch.dtype]] = None

    def to_dict(self):
        # Note(Kuntai): this is used for serializing MemoryObjMetadata via
        # msgpack.
        return {
            "__type__": "MemoryObjMetadata",
            "shape": list(self.shape),  # torch.Size -> list
            "dtype": str(self.dtype) if self.dtype else None,
            "address": self.address,
            "phy_size": self.phy_size,
            "ref_count": self.ref_count,
            "fmt": self.fmt.value,
            "shapes": [list(shape) for shape in self.shapes] if self.shapes else None,
            "dtypes": [str(dtype) for dtype in self.dtypes] if self.dtypes else None,
        }

    @staticmethod
    def from_dict(d):
        dtype_str = d["dtype"]
        dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None
        shapes_list = d["shapes"]
        shapes = [torch.Size(s) for s in shapes_list] if shapes_list else None
        dtypes_list = d["dtypes"]
        dtypes = (
            [getattr(torch, d_str.replace("torch.", "")) for d_str in dtypes_list]
            if dtypes_list
            else None
        )
        return MemoryObjMetadata(
            shape=torch.Size(d["shape"]),
            dtype=dtype,
            address=d["address"],
            phy_size=d["phy_size"],
            ref_count=d["ref_count"],
            fmt=MemoryFormat(d["fmt"]),
            shapes=shapes,
            dtypes=dtypes,
        )

    def get_size(self) -> int:
        if self.shapes is not None and self.dtypes is not None:
            return get_size_bytes(self.shapes, self.dtypes)
        return self.shape.numel() * self.dtype.itemsize  # type: ignore


class MemoryObj(metaclass=abc.ABCMeta):
    """
    MemoryObj interface.
    """

    # subclasses should expose raw_data differently
    raw_data: Any

    def __init__(self, metadata: MemoryObjMetadata):
        self.meta = metadata

    @abc.abstractmethod
    def invalidate(self):
        """
        Invalidate the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self):
        """
        Check if the MemoryObj is valid.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the MemoryObj in bytes.
        Note that this number could be smaller than the physical size.
        The physical size is aligned to the allocator's alignment.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        """
        Get the shape of the MemoryObj.
        """
        raise NotImplementedError

    def get_dtype(self) -> Optional[torch.dtype]:
        """
        Get the dtype of the MemoryObj.
        """
        return None

    @abc.abstractmethod
    def get_shapes(self) -> list[torch.Size]:
        """
        Get the shapes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dtypes(self) -> list[torch.dtype]:
        """
        Get the dtypes of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_memory_format(self) -> MemoryFormat:
        """
        Get the memory format of the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_physical_size(self) -> int:
        """
        Get the physical size of the MemoryObj in bytes.
        """
        raise NotImplementedError

    def set_used_size(self, n: int) -> None:  # noqa: B027
        """Narrow this buffer's logical size to the first ``n`` bytes.

        Optional hook for callers that have just written ``n`` bytes
        into a buffer originally allocated with an upper-bound size
        (e.g. the async serde processor, where the destination is sized
        from ``estimate_serialized_size`` but ``serialize`` returns the
        actual ``n`` it wrote). After this call, ``get_size()`` /
        ``byte_array`` / any downstream L2 adapter that reads the
        logical size will see exactly ``n`` bytes.

        Default is a no-op so subclasses without a "used vs allocated"
        distinction (e.g. :class:`BytesBufferMemoryObj`, where the raw
        bytes already are the actual contents) keep working unchanged.

        Args:
            n: bytes actually used in this buffer. Subclasses that
                implement this must validate ``n`` and raise
                ``ValueError`` on out-of-range or unsupported layouts.
        """
        pass

    @abc.abstractmethod
    def pin(self) -> bool:
        """
        Pin the memory obj so that it will not be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_up(self):
        """
        Increase ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unpin(self) -> bool:
        """
        Unpin the memory obj so that it can be evicted.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_count_down(self):
        """
        Decrease ref count for the given MemoryObj by one.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ref_count(self) -> int:
        """
        Get ref count for the given MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_tokens(self) -> int:
        """
        Get token number for the given MemoryObj.
        """
        raise NotImplementedError

    @property
    def shm_offset(self) -> int:
        """Return the byte offset of this object inside the SHM pool."""
        return self.meta.address

    @property
    def shm_byte_length(self) -> int:
        """Return the byte length of this object inside the SHM pool."""
        return self.get_size()

    @property
    @abc.abstractmethod
    def metadata(self) -> MemoryObjMetadata:
        """
        Get the metada of the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor(self) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def byte_array(self) -> bytes:
        """
        Get the byte array from the MemoryObj.
        The size is will be the physical size instead of the unaligned size.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_ptr(self) -> int:
        """
        Get the data pointer of the MemoryObj.
        This is used to access the raw data in the memory.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_pinned(self) -> bool:
        """
        Check whether the memory obj is pinned.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def raw_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the raw tensor from the MemoryObj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        """
        Get the tensor from the MemoryObj at the given index(group).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        """
        Get the allocator that allocates this memory object
        """
        raise NotImplementedError


@dataclass
class PinnedAllocFree:
    """Resolved alloc/free function pair for pinned CPU memory."""

    alloc_fn: Any
    alloc_args: tuple
    free_fn: Any
    free_args: tuple

    def alloc(self) -> int:
        """Allocate pinned memory and return the raw pointer."""
        return self.alloc_fn(*self.alloc_args)

    def free(self, ptr: int) -> None:
        """Free a previously allocated pinned-memory pointer."""
        self.free_fn(ptr, *self.free_args)


def _resolve_pinned_alloc_free(
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    size: Optional[int] = None,
    use_hugepages: bool = False,
) -> PinnedAllocFree:
    """Resolve the alloc/free function pair based on memory type.

    Returns:
        A PinnedAllocFree with the resolved functions and their extra
        arguments.  Call ``ptr = resolved.alloc()`` and ``resolved.free(ptr)``.
    """
    if shm_name:
        if use_hugepages:
            raise ValueError("Hugepages are not supported with shared memory (shm)")
        return PinnedAllocFree(
            alloc_fn=lmc_ops.alloc_shm_pinned_ptr,
            alloc_args=(size, shm_name),
            free_fn=lmc_ops.free_shm_pinned_ptr,
            free_args=(size, shm_name),
        )
    elif numa_mapping:
        if torch_dev.is_available():
            current_device_id = torch_dev.current_device()
        else:
            current_device_id = 0
        gpu_to_numa_mapping = numa_mapping.gpu_to_numa_mapping
        assert current_device_id in gpu_to_numa_mapping, (
            f"Current device {current_device_id} is not in the GPU NUMA mapping."
        )
        numa_id = gpu_to_numa_mapping[current_device_id]
        if use_hugepages:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_hugepage_pinned_numa_ptr,
                alloc_args=(size, numa_id),
                free_fn=lmc_ops.free_hugepage_pinned_numa_ptr,
                free_args=(size,),
            )
        else:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_pinned_numa_ptr,
                alloc_args=(size, numa_id),
                free_fn=lmc_ops.free_pinned_numa_ptr,
                free_args=(size,),
            )
    else:
        flags = 0
        if use_hugepages:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_hugepage_pinned_ptr,
                alloc_args=(size, flags),
                free_fn=lmc_ops.free_hugepage_pinned_ptr,
                free_args=(size,),
            )
        else:
            return PinnedAllocFree(
                alloc_fn=lmc_ops.alloc_pinned_ptr,
                alloc_args=(size, flags),
                free_fn=lmc_ops.free_pinned_ptr,
                free_args=(),
            )


def _read_hugepage_info() -> Optional[Tuple[int, int, int]]:
    """Read hugepage pool stats from sysfs.

    NOTE: We only use 2 MiB hugepages, so the pool stats are taken from
    the 2 MiB pool directly rather than the system default pool reported in
    ``/proc/meminfo`` (which can be 1 GiB on some hosts).

    Returns:
        ``(nr_hugepages, free_hugepages, page_size_mb)`` for the hugepage
        pool, or ``None`` if the sysfs entries are unavailable.
    """
    base = "/sys/kernel/mm/hugepages/hugepages-2048kB"
    try:
        with open(f"{base}/nr_hugepages") as f:
            total = int(f.read().strip())
        with open(f"{base}/free_hugepages") as f:
            free = int(f.read().strip())
        return total, free, 2
    except (OSError, ValueError):
        return None


def _allocate_cpu_memory(
    size: int,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    use_hugepages: bool = False,
) -> torch.Tensor:
    if size == 0:
        return torch.empty(0, dtype=torch.uint8)

    resolved = _resolve_pinned_alloc_free(
        numa_mapping,
        shm_name,
        size,
        use_hugepages,
    )

    try:
        ptr = resolved.alloc()
    except RuntimeError as e:
        if use_hugepages and "mmap failed" in str(e):
            diag = _read_hugepage_info()
            if diag is not None:
                total, free, page_mb = diag
                page_bytes = page_mb * 1024 * 1024
                needed = (size + page_bytes - 1) // page_bytes
                logger.error(
                    "Failed to allocate huge pages. "
                    "Pool has %d pages (%d free, each %d MiB). "
                    "Requested %d bytes (%d pages). "
                    "Please grow the %d MiB hugepage pool.",
                    total,
                    free,
                    page_mb,
                    size,
                    needed,
                    page_mb,
                )
            else:
                logger.error(
                    "Failed to allocate huge pages. "
                    "Please grow the 2 MiB hugepage pool."
                )
        raise

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_address(ptr)
    buffer = torch.frombuffer(buf, dtype=torch.uint8)

    return buffer


def _free_cpu_memory(
    buffer: torch.Tensor,
    size: int | None = None,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
    use_hugepages: bool = False,
) -> None:
    if torch_dev.is_available():
        torch_dev.synchronize()

    resolved = _resolve_pinned_alloc_free(
        numa_mapping,
        shm_name,
        size,
        use_hugepages,
    )
    resolved.free(buffer.data_ptr())


def _allocate_gpu_memory(
    size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    page_size = os.sysconf("SC_PAGESIZE")

    # Over-allocate
    base_buffer = torch.empty(size + page_size, dtype=torch.uint8, device=device)
    offset = -base_buffer.data_ptr() % page_size

    # Make aligned view
    aligned_buffer = base_buffer[offset : offset + size]

    # Need to return the base buffer as well in order to prevent GC
    return base_buffer, aligned_buffer


class TensorMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    monitor = LMCStatsMonitor.GetOrCreate()

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        parent_allocator: Optional["MemoryAllocatorInterface"],
    ):
        assert metadata.dtype is not None, "dtype must be specified for TensorMemoryObj"
        super().__init__(metadata)
        self.raw_data = raw_data
        self.valid = True
        self.lock = threading.Lock()
        self.parent_allocator = parent_allocator
        # ``None`` means "use the layout-derived size from
        # group_prefix_sum"; a non-None value narrows the logical view to
        # exactly that many bytes (see set_used_size).  Allocator reuse
        # paths must reset this to None along with the rest of the
        # per-allocation metadata.
        self._used_size_override: Optional[int] = None
        # Calculate the prefix sum of the group sizes
        # If there are two groups, the prefix sum will be
        # [0, size_of_group_1, size_of_group_1 + size_of_group_2]
        self.group_prefix_sum = [0]
        if self.meta.shapes is not None and self.meta.dtypes is not None:
            size_in_bytes = 0
            for shape, dtype in zip(self.meta.shapes, self.meta.dtypes, strict=True):
                size_in_bytes += shape.numel() * dtype.itemsize
                self.group_prefix_sum.append(size_in_bytes)
        else:
            self.group_prefix_sum.append(self.meta.get_size())

    def __del__(self):
        """
        Destructor to ensure memory is released when the object is garbage collected.
        This acts as a safety net to prevent memory leaks if ref_count_down() is not
        called properly somewhere in the code path.
        """
        if self.parent_allocator is not None and self.is_valid():
            if self.meta.ref_count > 0 or self.meta.pin_count > 0:
                logger.warning(
                    "MemoryObj at %s is being garbage collected "
                    "with ref_count=%d, pin_count=%d. "
                    "This indicates ref_count_down()/unpin() was not called properly.",
                    self.meta.address,
                    self.meta.ref_count,
                    self.meta.pin_count,
                )
            self.parent_allocator.free(self)

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        if self._used_size_override is not None:
            return self._used_size_override
        return self.group_prefix_sum[-1]

    def set_used_size(self, n: int) -> None:
        """Narrow the logical size to ``n`` bytes after a write.

        After this call, ``get_size()`` returns ``n`` and ``byte_array``
        exposes exactly ``n`` bytes from the start of ``raw_data``.  The
        physical allocation (``get_physical_size``) and ``raw_data``
        buffer are unchanged.  Allocator reuse resets this override to
        ``None`` so a recycled block returns to its layout-derived size.

        Note: the ``tensor`` property still derives its shape from
        ``meta.shape``, so accessing ``.tensor`` on a buffer narrowed
        below its layout size will fail to reshape.  Use ``byte_array``
        (or read ``raw_data[: get_size()]`` directly) for downstream
        I/O that must honor the narrowed size.

        Args:
            n: bytes actually written.  Must satisfy
                ``0 <= n <= get_physical_size()``.

        Raises:
            ValueError: if ``n`` is outside the allowed range.
        """
        if n < 0 or n > self.meta.phy_size:
            raise ValueError(
                f"set_used_size: n={n} out of range [0, {self.meta.phy_size}]"
            )
        with self.lock:
            self._used_size_override = n

    # TODO(chunxiaozheng): use get_shapes and get_dtypes to replace
    #  get_shape and get_dtype
    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        assert self.meta.dtype is not None
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        assert self.meta.shapes is not None
        return self.meta.shapes

    def get_dtypes(self) -> list[torch.dtype]:
        assert self.meta.dtypes is not None
        return self.meta.dtypes

    def get_memory_format(self) -> MemoryFormat:
        with self.lock:
            return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        with self.lock:
            self.meta.ref_count += 1

    def ref_count_down(self):
        with self.lock:
            self.meta.ref_count -= 1
            if self.meta.ref_count < 0:
                logger.warning(
                    f"Ref count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.ref_count}."
                    "Double free occurred somewhere."
                    "Setting ref count back to 0 as a hack but please find the bug."
                )
                self.meta.ref_count = 0
            if (
                self.meta.ref_count == 0
                and self.parent_allocator is not None
                and self.meta.pin_count == 0
            ):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count

    def get_num_tokens(self) -> int:
        with self.lock:
            token_dim = self.meta.fmt.token_dim()
            return self.meta.shape[token_dim]

    def pin(self) -> bool:
        with self.lock:
            # if pin_count is 0, indicates that the object is pinned for the first time
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(1)

            self.meta.pin_count += 1

            # Register/update with PinMonitor for timeout tracking on every pin
            pin_monitor = PinMonitor.GetOrCreate()
            pin_monitor.on_pin(self)
            return True

    def unpin(self) -> bool:
        with self.lock:
            self.meta.pin_count -= 1

            # if pin_count is 0, indicates that the object is unpinned
            if self.meta.pin_count == 0:
                TensorMemoryObj.monitor.update_pinned_memory_objs_count(-1)
                # Unregister from PinMonitor when fully unpinned
                pin_monitor = PinMonitor.GetOrCreate()
                pin_monitor.on_unpin(self)

            if self.meta.pin_count <= 0 and self.meta.ref_count <= 0:
                if self.parent_allocator is None:
                    logger.error(
                        "Parent allocator is None when trying to free MemoryObj."
                        "This could cause memory leak"
                    )
                else:
                    self.parent_allocator.free(self)

            if self.meta.pin_count < 0:
                logger.warning(
                    f"Pin count of MemoryObj {self.meta.address}"
                    f"is negative: {self.meta.pin_count}."
                    "Double unpin occurred somewhere."
                    "Setting pin count back to 0 as a hack but please find the bug."
                )
                self.meta.pin_count = 0
            return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        with self.lock:
            return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.dtype is not None
        if self._used_size_override is not None:
            # Narrowed byte buffer (see set_used_size): expose exactly
            # the used bytes as a flat uint8 view.  Reshaping to the
            # original meta.shape would raise -- fewer than shape-many
            # bytes are logically present -- so keep the view consistent
            # with get_size()/byte_array/shm_byte_length, which all
            # report the narrowed length.  Consumers that build SHM
            # transport slots from ``tensor.shape`` and
            # ``shm_byte_length`` then stay self-consistent.
            return self.raw_data[: self._used_size_override].view(torch.uint8)
        # TODO(Jiayi): consider caching the `get_size()`
        return (
            self.raw_data[: self.get_size()].view(self.meta.dtype).view(self.meta.shape)
        )

    @property
    def byte_array(self) -> memoryview:
        # TODO: consider using one of the alternatives

        # Alternative 1:
        # # PyTorch tensors support buffer protocol directly for CPU tensors
        # return memoryview(self.raw_data)

        # Alternative 2:
        # assert self.raw_data.device.type == 'cpu',
        #   "byte_array only works with CPU tensors"
        # return memoryview(self.raw_data.contiguous().numpy())

        # Use logical size (get_size) rather than raw_data physical size.
        # The raw_data buffer may include alignment padding (e.g. from
        # batched_allocate) that must not be exposed to callers such as
        # remote-backend put/get which rely on byte_array length matching
        # the metadata length.
        num_bytes = self.get_size()
        ptr = self.raw_data.data_ptr()
        # ctypes does not cache (c_ubyte * N) array types -- each `*` builds a
        # fresh heap type. With this property accessed once per remote put/get,
        # uncached creation leaks ~1-2 kB per call (heap-type metadata is held
        # by the type system and never reclaimed). Cache the array type per
        # size so steady-state usage reuses a fixed set of types. See
        # https://github.com/LMCache/LMCache/issues/3767.
        arr_type = _get_cached_ubyte_array_type(num_bytes)
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = arr_type.from_address(ctypes.addressof(ubyte_ptr.contents))
        return memoryview(byte_array)

    @property
    def data_ptr(self) -> int:
        return self.raw_data.data_ptr()

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A memory obj can be evicted if it is not pinned and ref_count=1.
        """
        return not self.is_pinned and self.get_ref_count() == 1

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return self.raw_data

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.shapes is not None
        assert self.meta.dtypes is not None
        begin = self.group_prefix_sum[index]
        end = self.group_prefix_sum[index + 1]
        return (
            self.raw_data[begin:end]
            .view(self.meta.dtypes[index])
            .view(self.meta.shapes[index])
        )

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        return self.parent_allocator


class BytesBufferMemoryObj(MemoryObj):
    """
    Wraps a raw flat tensor with some metadata
    """

    def __init__(self, raw_bytes: bytes, metadata: Optional[MemoryObjMetadata] = None):
        self.raw_data = raw_bytes
        if metadata is None:
            bytes_shape = torch.Size([len(self.raw_data), 0, 0, 0])
            metadata = MemoryObjMetadata(
                shape=bytes_shape,
                dtype=None,
                address=0,
                phy_size=0,
                ref_count=1,
                pin_count=0,
                fmt=MemoryFormat.BINARY_BUFFER,
            )
        super().__init__(metadata)
        self.valid = True

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return len(self.raw_data)

    def get_shape(self) -> torch.Size:
        return torch.Size([len(self.raw_data), 0, 0, 0])

    def get_dtype(self) -> Optional[torch.dtype]:
        return None

    def get_shapes(self) -> list[torch.Size]:
        return [self.get_shape()]

    def get_dtypes(self) -> list[torch.dtype]:
        return []

    def get_memory_format(self) -> MemoryFormat:
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return self.metadata.phy_size

    def pin(self) -> bool:
        self.metadata.pin_count += 1
        return True

    def unpin(self) -> bool:
        self.metadata.pin_count -= 1
        if self.metadata.pin_count < 0:
            logger.warning(
                f"Pin count of MemoryObj {self.meta.address}"
                f"is negative: {self.meta.pin_count}."
                "Double unpin occurred somewhere."
                "Setting pin count back to 0 as a hack but please find the bug."
            )
            self.metadata.pin_count = 0
        return True

    def ref_count_up(self):
        pass

    def ref_count_down(self):
        pass

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        # TODO(Jiayi): record the number of tokens somehow
        return 1

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    @property
    def byte_array(self) -> bytes:
        return self.raw_data

    @property
    def data_ptr(self) -> int:
        mv = memoryview(self.raw_data)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
        return addr

    @property
    def is_pinned(self) -> bool:
        return self.metadata.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """
        Check whether the memory obj can be evicted.
        A buffer memory obj can be evicted if it is not pinned.
        """
        return not self.is_pinned

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        # NOTE: BytesBufferMemoryObj may not be allocated by any allocator,
        # so just return None here
        return None


class GDSMemoryObject(MemoryObj):
    """A slab-anchored ``MemoryObj`` for the GDS L1 tier.

    The bytes live in the GDS slab file, not in host or device memory, so
    this object carries only the slab ``(offset, size)`` (in ``meta.address``
    / ``meta.phy_size``) and is otherwise a placeholder: ``tensor`` is always
    ``None`` and ``byte_array`` / ``data_ptr`` raise.
    """

    def __init__(self, metadata: MemoryObjMetadata) -> None:
        super().__init__(metadata)
        self.valid = True

    @property
    def slab_offset(self) -> int:
        """Byte offset of this chunk within the slab file (== ``meta.address``)."""
        return self.meta.address

    def invalidate(self) -> None:
        self.valid = False

    def is_valid(self) -> bool:
        return self.valid

    def get_size(self) -> int:
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        raise NotImplementedError(
            "GDSMemoryObject.get_shapes: per-group shapes are not tracked on "
            "the GDS path (only the singular meta.shape is); use get_shape()"
        )

    def get_dtypes(self) -> list[torch.dtype]:
        raise NotImplementedError(
            "GDSMemoryObject.get_dtypes: per-group dtypes are not tracked on "
            "the GDS path (only the singular meta.dtype is); use get_dtype()"
        )

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self) -> None:
        raise NotImplementedError(
            "GDSMemoryObject.ref_count_up: not used on the GDS path"
        )

    def ref_count_down(self) -> None:
        raise NotImplementedError(
            "GDSMemoryObject.ref_count_down: not used on the GDS path"
        )

    def get_ref_count(self) -> int:
        raise NotImplementedError(
            "GDSMemoryObject.get_ref_count: not used on the GDS path"
        )

    def get_num_tokens(self) -> int:
        raise NotImplementedError(
            "GDSMemoryObject.get_num_tokens: not used on the GDS path"
        )

    def pin(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.pin: not used on the GDS path")

    def unpin(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.unpin: not used on the GDS path")

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return None

    @property
    def byte_array(self) -> bytes:
        raise NotImplementedError(
            f"GDSMemoryObject(slab_offset={self.slab_offset}).byte_array is not "
            "supported; bytes live in the GDS slab file and the staging buffer "
            "is registered VRAM (no buffer protocol)."
        )

    @property
    def data_ptr(self) -> int:
        raise NotImplementedError(
            f"GDSMemoryObject(slab_offset={self.slab_offset}).data_ptr is not "
            "supported; GDS reads/writes use gpu_buffer.data_ptr() via the "
            "gpu_ops dispatch, never the MemoryObj's data_ptr."
        )

    @property
    def is_pinned(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.is_pinned: not used on the GDS path")

    @property
    def can_evict(self) -> bool:
        raise NotImplementedError("GDSMemoryObject.can_evict: not used on the GDS path")

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        # The GDS slab is not a MemoryAllocatorInterface; dispatch in gpu_ops
        # keys off the GDSMemoryObject type, not the parent allocator.
        return None


class MemoryAllocatorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Allocates the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A MemoryObj wrapping the allocated memory. Returns
            None if the allocation failed.

        :rtype: Optional[MemoryObj]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """
        Batched allocate the memory to hold a tensor of the given shape.

        :param torch.Size shapes: The shape of the tensor to allocate.
        :param torch.dtype dtypes: The dtype of the tensor to allocate.
        :param int batch_size: The number of tensors to allocate.
        :param MemoryFormat fmt: The format of the memory to allocate.

        :return: A list of MemoryObjs wrapping the allocated memory.
            Returns None if the allocation failed.

        :rtype: Optional[List[MemoryObj]]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        """
        Frees the memory allocated for the given MemoryObj.
        Note that this function shouldn't be explicitly called.
        Instead, use `ref_count_down` to decrease ref count.

        :param MemoryObj memory_obj: The MemoryObj to free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        """
        Frees the memory allocated for the given list of MemoryObjs.

        :param List[MemoryObj] memory_objs: The list of MemoryObjs
            to free.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the memory allocator.
        This is called when the LMCacheEngine is closed.
        """
        return

    def memcheck(self) -> bool:
        """
        Checks the memory allocator for consistency.

        Returns:
            True if everything is fine otherwise False
        """
        return True

    # TODO(chunxiaozheng): remove if after all params replaced by shapes/dtypes
    def _adapt_shapes_and_dtypes(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> Tuple[list[torch.Size], list[torch.dtype]]:
        if isinstance(shapes, torch.Size):
            shapes = [shapes]

        if isinstance(dtypes, torch.dtype):
            dtypes = [dtypes]

        assert len(shapes) == len(dtypes), (
            f"shapes and dtypes must have the same length, "
            f"got {len(shapes)} and {len(dtypes)}, "
            f"shapes: {shapes}, dtypes: {dtypes}"
        )
        return shapes, dtypes


class AddressManager:
    """
    Manages a virtual address space starting from 0 for memory allocation.

    Key interfaces:
    - allocate(size): Allocate a block of memory of the given size. The starting
      address and the actual allocated size will be aligned.

    - free(address, size): Free a previously allocated region. Note that if the
      region is not "allocated" before, it may have internal errors.

    - sbrk(size): Expand the virtual address space by the given size. The size
      will be aligned internally.

    Core assumptions:
    - The allocated size should be aligned with ALIGN_BYTES.
    """

    ALIGN_BYTES = 4096

    def __init__(self, size: int, align_bytes: int = ALIGN_BYTES):
        """
        Initializes the AddressManager with a given size.

        Args:
            size: The initial size of the virtual address space.
            align_bytes: The alignment requirement for allocations.
        """
        self._size = size
        self._align = align_bytes

        # Current implementation: explicit list
        self._explicit_list: SortedList[FreeBlock] = SortedList(key=lambda x: x.start)
        self._explicit_list.add(FreeBlock(start=0, size=size))

        # thread safe lock
        self._lock = threading.Lock()

        # For debugging purposes
        self.total_allocated_size = 0

    def compute_aligned_size(self, raw_size: int) -> int:
        """
        Helper function to compute the aligned size for a given raw size.

        Args:
            raw_size: The raw size to be aligned.

        Returns:
            The aligned size.
        """
        return (raw_size + self._align - 1) & ~(self._align - 1)

    def _can_merge_with_prev(
        self, curr_block: FreeBlock, prev_block: FreeBlock
    ) -> bool:
        """Hook: Check if curr_block can merge with prev_block."""
        return prev_block.can_be_coalesced(curr_block)

    def _can_merge_with_succ(
        self, curr_block: FreeBlock, succ_block: FreeBlock
    ) -> bool:
        """Hook: Check if curr_block can merge with succ_block."""
        return curr_block.can_be_coalesced(succ_block)

    @_lmcache_nvtx_annotate
    def _coalesce(
        self,
        curr_block: FreeBlock,
        prev_block: Optional[FreeBlock],
        succ_block: Optional[FreeBlock],
    ):
        """
        Coalesces the current block with the previous and/or successor block.
        This assumes the curr_block is NOT in self._explicit_list

        Returns True if the current block was coalesced, otherwise False.
        """
        merge_prev = prev_block is not None and self._can_merge_with_prev(
            curr_block, prev_block
        )
        merge_succ = succ_block is not None and self._can_merge_with_succ(
            curr_block, succ_block
        )

        if merge_prev and merge_succ:
            prev_block.size += curr_block.size + succ_block.size  # type: ignore
            self._explicit_list.remove(succ_block)
        elif merge_prev:
            prev_block.size += curr_block.size  # type: ignore
        elif merge_succ:
            # NOTE: logically, this won't change the order of the succ_block,
            #       so we don't need to do a "remove" and "reinsert" here
            self._explicit_list.remove(succ_block)
            succ_block.start -= curr_block.size  # type: ignore
            succ_block.size += curr_block.size  # type: ignore
            self._explicit_list.add(succ_block)

        return merge_prev or merge_succ

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def allocate(self, size: int) -> tuple[int, int]:
        """
        Allocate a block of memory from the virtual address space of a given
        size. The actual allocated size could be larger than the requested size
        in order to satisfy alignment requirements.

        Args:
            size: The requested size of the memory block. Should be greater
                than 0.

        Returns:
            A tuple (address, allocated_size) where address is the starting
            address of the allocated block and allocated_size is the actual
            size of the allocated block.

        Raises:
            RuntimeError: If no memory is available to allocate.
        """
        aligned_size = self.compute_aligned_size(size)
        for block in self._explicit_list:
            if block.size >= aligned_size:
                break
        else:
            logger.warning(
                "Failed to allocate memory block of size %d "
                "because no memory is available",
                size,
            )
            raise RuntimeError(
                f"Failed to allocate memory block of size {size} "
                "because no memory is available"
            )

        self._explicit_list.remove(block)
        if block.size > aligned_size:
            self._explicit_list.add(
                FreeBlock(
                    start=block.start + aligned_size,
                    size=block.size - aligned_size,
                )
            )

        # For debug
        self.total_allocated_size += aligned_size

        return block.start, aligned_size

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def batched_allocate(self, size: int, batch_size: int) -> list[tuple[int, int]]:
        """
        Allocate blocks of memory from the virtual address space of a given
        size and batch size. The actual allocated size could be larger than
        the requested size in order to satisfy alignment requirements.

        Args:
            size: The requested size of the memory block. Should be greater
                than 0.
            batch_size: The number of memory blocks to allocate.

        Returns:
            A list of tuple (address, allocated_size) where address is the starting
            address of the allocated block and allocated_size is the actual size of
            the allocated block.
            Note: the length of the return list is the same as the batch_size.

        Raises:
            RuntimeError: If no memory is available to allocate.
        """
        aligned_size = self.compute_aligned_size(size)
        remaining = batch_size
        allocate_result: list[tuple[int, int]] = []

        blocks_to_remove: list[FreeBlock] = []
        blocks_to_add: list[FreeBlock] = []

        for block in self._explicit_list:
            if remaining <= 0:
                break
            if block.size < aligned_size:
                continue

            # Greedily carve out as many aligned_size chunks as possible
            num_from_block = min(remaining, block.size // aligned_size)
            start = block.start
            for i in range(num_from_block):
                allocate_result.append((start + i * aligned_size, aligned_size))
            remaining -= num_from_block

            # Mark the original block for removal
            blocks_to_remove.append(block)

            # Keep the remaining tail as a new free block if any space is left
            used = num_from_block * aligned_size
            if block.size > used:
                blocks_to_add.append(
                    FreeBlock(start=block.start + used, size=block.size - used)
                )

        if remaining > 0:
            # Not enough memory; free list is untouched, no rollback needed
            logger.warning(
                "Failed to batched allocate %d memory blocks of size %d "
                "because no enough memory is available (short by %d blocks)",
                batch_size,
                size,
                remaining,
            )
            raise RuntimeError(
                f"Failed to batched allocate {batch_size} memory blocks "
                f"of size {size} because no enough memory is available"
            )
        if len(allocate_result) != batch_size:
            # The length of allocate_result is not equal to batch_size;
            # free list is untouched, no rollback needed
            logger.warning(
                "Failed to batched allocate %d memory blocks of size %d "
                "because the length of allocate_result %d is not equal to batch_size",
                batch_size,
                size,
                len(allocate_result),
            )
            raise RuntimeError(
                f"Failed to batched allocate {batch_size} memory blocks "
                f"of size {size} because the length of allocate_result "
                f"{len(allocate_result)} is not equal to batch_size"
            )

        # Allocation succeeded; batch-update the free list
        for block in blocks_to_remove:
            self._explicit_list.remove(block)
        for block in blocks_to_add:
            self._explicit_list.add(block)

        # Update debug statistics
        total_allocated = aligned_size * batch_size
        self.total_allocated_size += total_allocated

        return allocate_result

    @_lmcache_nvtx_annotate
    @synchronized("_lock")
    def free(self, address: int, size: int):
        """
        Free a previously allocated block of memory.

        Args:
            address: The starting address of the block to free.
            size: The size of the block to free. Should be greater than 0.
        """
        new_free_block = FreeBlock(start=address, size=size)
        index = self._explicit_list.bisect_left(new_free_block)
        prev_block = self._explicit_list[index - 1] if index > 0 else None
        succ_block = (
            self._explicit_list[index] if index < len(self._explicit_list) else None
        )

        coalesced = self._coalesce(new_free_block, prev_block, succ_block)
        if not coalesced:
            self._explicit_list.add(new_free_block)

        # For debug
        self.total_allocated_size -= size

    @synchronized("_lock")
    def sbrk(self, size: int):
        """
        Expand the virtual address space by a given size.

        Args:
            size: The size to expand the address space. Will be aligned internally
                with the ALIGN_BYTES
        """
        size = self.compute_aligned_size(size)
        new_block = FreeBlock(start=self._size, size=size)
        prev_block = self._explicit_list[-1] if len(self._explicit_list) > 0 else None
        succ_block = None
        coalesced = self._coalesce(new_block, prev_block, succ_block)
        if not coalesced:
            self._explicit_list.add(new_block)

        self._size += size

    def get_heap_size(self) -> int:
        """
        Get the total size of the address space.

        Returns:
            The total size in bytes.
        """
        return self._size

    def get_free_size(self) -> int:
        """
        Get the total free size in the address space.

        Returns:
            The total free size in bytes.
        """
        return self._size - self.total_allocated_size

    def check_consistency(self) -> bool:
        """
        Check if the address manager is consistent.

        Returns:
            True if consistent, False otherwise.
        """
        # Check if free blocks are properly coalesced
        for prev, succ in zip(
            self._explicit_list[:-1], self._explicit_list[1:], strict=False
        ):
            if prev.can_be_coalesced(succ):
                return False

        # Check if total size matches
        total_free_size = sum(block.size for block in self._explicit_list)
        if total_free_size + self.total_allocated_size != self._size:
            return False

        return True


_CORE_EXPORTS = [
    "AddressManager",
    "BytesBufferMemoryObj",
    "FreeBlock",
    "GDSMemoryObject",
    "MemoryAllocatorInterface",
    "MemoryFormat",
    "MemoryObj",
    "MemoryObjMetadata",
    "TensorMemoryObj",
    "torch_device_type",
]

__all__ = list(_CORE_EXPORTS)
