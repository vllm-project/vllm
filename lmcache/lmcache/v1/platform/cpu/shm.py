# SPDX-License-Identifier: Apache-2.0
"""CPU-only KV-cache IPC wrapper backed by POSIX shared memory.

Mirrors the GPU-mode CUDA-IPC zero-copy semantics for hosts without an
accelerator: client and LMCache mp server map the **same** physical
pages so transfers are pointer-shuffles rather than memcpys.

Bound to ``device_type="cpu"`` via
:attr:`~lmcache.v1.platform.cpu.CpuDeviceSpec.ipc_wrapper_cls`, so the
multiprocess adapter can dispatch by ``tensor.device.type`` without
any if/elif chain.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes
import itertools
import os
import threading
import weakref

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_map_readwrite,
    shm_munmap,
    shm_unlink,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)

# Re-export POSIX-SHM primitives so existing callers keep working.
# The canonical home is :mod:`lmcache.v1.multiprocess.posix_shm`; new
# code (e.g. the MP non-GPU SHM transport) should import from there.
__all__ = [
    "CpuShmTensorWrapper",
    "inject_stale_cache_entry_for_test",
    "migrate_to_shm_and_wrap",
    "shm_create_readwrite",
    "shm_map_readwrite",
    "shm_munmap",
    "shm_unlink",
]

# ---------------------------------------------------------------------------
# Wrapper class                                                             #
# ---------------------------------------------------------------------------


class CpuShmTensorWrapper(DeviceIPCWrapper):
    """IPC wrapper for CPU tensors backed by POSIX shared memory.

    Used by the ``lmcache bench kvcache --mode cpu`` path and engine
    CPU integrations so that the client and the LMCache mp server
    map the **same** physical pages for the KV cache, mirroring the
    GPU-mode CUDA-IPC zero-copy semantics.

    Subclassing :class:`DeviceIPCWrapper` is load-bearing for the same
    reason :class:`RawCudaIPCWrapper` does it: msgspec does not
    support unions of custom ext-encoded types, so all wire-level
    KV-cache wrappers must share the single ext code (1) registered
    for ``DeviceIPCWrapper``. Pickle preserves the subclass identity
    so ``to_tensor`` dispatches correctly on both sides.
    """

    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding
    #: without instantiating the wrapper.
    device_type: ClassVar[str] = "cpu"

    # POSIX shared-memory name (``/lmcache_...``) -- leading ``/`` is
    # required by ``shm_open(3)`` on both Linux and macOS.
    SHM_NAME_PREFIX = "/lmcache_kv_"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "CpuShmTensorWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Delegates to :func:`migrate_to_shm_and_wrap`, which migrates the
        tensor's storage to a POSIX SHM segment so the LMCache mp server
        can map the same physical pages.

        Args:
            tensor: A contiguous CPU tensor to migrate and wrap.

        Returns:
            A new :class:`CpuShmTensorWrapper` referencing the SHM
            segment that now backs ``tensor``.
        """
        return migrate_to_shm_and_wrap(tensor)

    def __init__(self, tensor: torch.Tensor, shm_name: str) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(
                "CpuShmTensorWrapper requires a CPU tensor, got %s" % tensor.device
            )
        if not tensor.is_contiguous():
            raise ValueError("CpuShmTensorWrapper requires a contiguous tensor")

        self.shm_name = shm_name
        # ``numel * element_size`` is the correct logical byte size; the
        # underlying storage may be larger when the tensor is a view.
        self.nbytes = tensor.numel() * tensor.element_size()

        # DeviceIPCWrapper interface fields. ``handle`` / ``device_uuid``
        # are unused on the CPU path but kept to satisfy the base
        # contract used by equality checks.
        self.handle = None
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        self.device_uuid = "cpu"

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor by mapping the same SHM segment.

        The returned tensor owns the mmap: a ``weakref.finalize`` hook
        runs ``munmap`` once the tensor (and any views derived from it)
        is garbage-collected, so the per-process virtual address space
        does not leak across repeated ``to_tensor`` calls.

        We rebuild the view through ``as_strided`` so the original
        memory layout (stride / storage_offset / memory_format) is
        replayed faithfully on the receiving side; reshape would
        silently re-coalesce strides and lose, e.g., channels_last.
        """
        # Empty tensors carry no SHM segment (mmap with length 0 is
        # undefined / EINVAL on POSIX); rebuild the empty view in-process.
        if self.nbytes == 0:
            return torch.empty(self.shape, dtype=self.dtype)
        addr = shm_map_readwrite(self.shm_name, self.nbytes)
        # ``torch.frombuffer`` requires a writable buffer; build one
        # via ctypes so the resulting torch tensor shares storage
        # with the SHM mapping (zero copy across processes).
        buf_type = ctypes.c_uint8 * self.nbytes
        buf = buf_type.from_address(addr)
        flat = torch.frombuffer(buf, dtype=torch.uint8)
        typed = flat.view(self.dtype)
        out = torch.as_strided(typed, self.shape, self.stride, self.storage_offset)
        # Pin the mmap to the *storage*, not the outer tensor: views
        # (reshape / slicing) create new tensor objects that share the
        # storage but do not inherit Python attributes, so a finalizer
        # attached to ``out`` would munmap as soon as ``out`` is GC'd
        # even when a view is still reading the SHM segment.
        # ``UntypedStorage`` is shared across views, so finalizing on it
        # only fires once every view is also dropped.
        storage = out.untyped_storage()
        _CPU_SHM_KEEP_ALIVE[id(storage)] = flat
        weakref.finalize(storage, _release_shm_segment, id(storage), addr, self.nbytes)
        return out


# ---------------------------------------------------------------------------
# Migrate-and-wrap factory (used by the multiprocess adapter)              #
# ---------------------------------------------------------------------------

# Per-process registry of SHM segments we have created, so the same
# tensor object is only migrated to SHM once even if the factory is
# called multiple times.
#
# Keyed by ``id(tensor)`` for cheap O(1) lookup, but each entry also
# holds a ``weakref.ref`` to the original tensor and we *verify the
# referent is still that exact object* before reusing the cached SHM
# name. CPython recycles object IDs, so a fresh tensor allocated at
# the same address as a previously migrated (now garbage-collected)
# one would otherwise inherit a stale name -- and because
# :func:`shm_create_readwrite` uses ``O_EXCL``, the next migration
# would crash with ``EEXIST`` ("File exists"). The weakref-validated
# lookup below makes that race impossible: a stale entry can only
# point at a dead referent, which we treat as a miss.
_CPU_SHM_NAMES: dict[int, tuple["weakref.ReferenceType[torch.Tensor]", str]] = {}
_CPU_SHM_LOCK = threading.Lock()
_CPU_SHM_COUNTER = itertools.count()


# Process-level registry that pins the base ``flat`` buffer of every live
# ``to_tensor()`` mmap until its storage is finalized. Keyed by ``id(storage)``,
# which is stable across views because PyTorch caches the storage Python
# wrapper (so reshape / slicing returns the same ``UntypedStorage`` object).
_CPU_SHM_KEEP_ALIVE: dict[int, torch.Tensor] = {}


def _release_shm_segment(storage_id: int, addr: int, nbytes: int) -> None:
    """Drop the pinned base buffer and ``munmap`` the mapping.

    Invoked by ``weakref.finalize`` on the tensor's ``UntypedStorage`` once
    every view of the mapping is gone, so views (e.g. ``reshape`` returning
    a new tensor without ``_lmcache_shm_buf``) cannot trigger a premature
    unmap that would turn into a use-after-free in the next read.
    """
    _CPU_SHM_KEEP_ALIVE.pop(storage_id, None)
    shm_munmap(addr, nbytes)


def _cleanup_shm_segment(tid: int, shm_name: str, addr: int, nbytes: int) -> None:
    """Release the mmap, unlink, and forget the cached SHM name."""
    with _CPU_SHM_LOCK:
        # Only drop the entry if it still points at *this* segment;
        # a future tensor reusing ``tid`` may already have replaced it.
        cached = _CPU_SHM_NAMES.get(tid)
        if cached is not None and cached[1] == shm_name:
            _CPU_SHM_NAMES.pop(tid, None)
    shm_munmap(addr, nbytes)
    shm_unlink(shm_name)


def migrate_to_shm_and_wrap(tensor: torch.Tensor) -> CpuShmTensorWrapper:
    """Re-point ``tensor``'s storage at a POSIX SHM segment, then wrap.

    Used as the registered ``"cpu"`` KV-wrapper factory: the LMCache mp
    server can mmap the same physical pages on the receiving side.
    Idempotent per tensor identity (validated via a stored weakref so
    Python's id-recycling cannot produce a stale-name hit). The SHM
    segment is released (``munmap`` + ``shm_unlink``) automatically
    when the migrated tensor is garbage-collected.
    """
    # First Party
    from lmcache.v1.gpu_connector.kv_format.contiguity import (
        attempt_permute_to_contiguous_view,
    )

    # Validate and normalise the tensor *before* touching the registry
    # or mutating storage, so a bad input never leaves things half-done.
    normalized = attempt_permute_to_contiguous_view(tensor)
    assert isinstance(normalized, torch.Tensor)
    if tensor.device.type != "cpu":
        raise ValueError(
            "migrate_to_shm_and_wrap requires a CPU tensor, got %s" % tensor.device
        )
    if not normalized.is_contiguous():
        raise ValueError("migrate_to_shm_and_wrap requires a contiguous tensor")

    tid = id(tensor)

    # Fast path: check the registry under the lock, return early if the
    # tensor has already been migrated.
    with _CPU_SHM_LOCK:
        cached = _CPU_SHM_NAMES.get(tid)
        if cached is not None:
            ref, cached_name = cached
            if ref() is tensor:
                return CpuShmTensorWrapper(tensor, cached_name)
            # Stale entry from a GC'd tensor whose id has been
            # reused; drop it and fall through to allocate fresh.
        _CPU_SHM_NAMES.pop(tid, None)

    nbytes = tensor.numel() * tensor.element_size()
    assert tensor.storage_offset() == 0, (
        "migrate_to_shm_and_wrap: SHM segment is sized to "
        "numel*elem_size; a nonzero storage_offset would cause "
        "OOB access. Got offset=%d" % tensor.storage_offset()
    )
    if nbytes == 0:
        # No SHM segment for empty tensors: ``mmap`` with length 0
        # is undefined / EINVAL on POSIX. ``to_tensor`` rebuilds an
        # empty view directly when ``shm_name`` is empty.
        return CpuShmTensorWrapper(tensor, "")

    shm_name = "%s%d_%d" % (
        CpuShmTensorWrapper.SHM_NAME_PREFIX,
        os.getpid(),
        next(_CPU_SHM_COUNTER),
    )
    # Perform the heavy work (syscall + tensor mutation) outside the lock
    # to keep the critical section small.
    addr = shm_create_readwrite(shm_name, nbytes)
    try:
        buf_type = ctypes.c_uint8 * nbytes
        buf = buf_type.from_address(addr)
        shm_storage = torch.frombuffer(buf, dtype=torch.uint8).untyped_storage()
        tensor.set_(
            shm_storage,
            normalized.storage_offset(),
            normalized.shape,
            normalized.stride(),
        )
    except Exception:
        # Make sure the SHM resources don't leak if migration fails
        # part-way (e.g. ``set_`` rejects an unusual stride).
        shm_munmap(addr, nbytes)
        shm_unlink(shm_name)
        raise

    with _CPU_SHM_LOCK:
        _CPU_SHM_NAMES[tid] = (weakref.ref(tensor), shm_name)
    weakref.finalize(tensor, _cleanup_shm_segment, tid, shm_name, addr, nbytes)
    logger.info(
        "Migrated CPU KV cache tensor (nbytes=%d) to SHM %s",
        nbytes,
        shm_name,
    )
    return CpuShmTensorWrapper(tensor, shm_name)


def inject_stale_cache_entry_for_test(
    tensor: torch.Tensor,
    dead_ref: "weakref.ReferenceType[torch.Tensor]",
    stale_shm_name: str,
) -> None:
    """Test-only hook: pre-seed the registry with a stale entry.

    Lets unit tests reproduce the CPython id-reuse race -- where a
    fresh tensor lands on the same id as a previously migrated and
    garbage-collected one -- without the per-test global-state
    surgery that would otherwise have to reach into the module's
    private dict / lock.
    """
    with _CPU_SHM_LOCK:
        _CPU_SHM_NAMES[id(tensor)] = (dead_ref, stale_shm_name)
