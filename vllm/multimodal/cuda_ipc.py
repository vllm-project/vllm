# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled CUDA-IPC transport for multimodal feature tensors (TP-aware).

This is the ``mm_tensor_ipc="cuda_ipc"`` implementation of the out-of-band
tensor hooks in :mod:`vllm.v1.serial_utils` (``OOBTensorConsumer`` /
``OOBTensorProvider``). It transports GPU-resident multimodal feature tensors
(e.g. ``pixel_values`` produced by a GPU image processor) from the API-server
(frontend) process to every tensor-parallel worker **without a host round
trip**, and unlike ``torch_shm`` it supports ``tensor_parallel_size > 1``.

Design:

- Sender (frontend): a single persistent GPU pool is allocated once and its CUDA
  IPC handle exported once. Each request copies its feature tensor into a free
  slice (device-to-device) and the OOB consumer returns only a small metadata
  proxy (pool handle + offset). No per-request ``cudaIpcGetMemHandle``.

- Deferred decode: the frontend->engine OOB provider does NOT open the handle;
  it returns the picklable :class:`CudaIpcPoolProxy` so it survives engine-core
  scheduling and the engine-core->worker broadcast as metadata (a few dozen
  bytes), never as a CUDA tensor. Each worker materializes the tensor itself.

- Receiver (every TP worker): the pool handle is opened once per process and
  cached, then each request slices the cached mapping and copies its tensor out
  on the worker's own device. rank 0 (same device as the pool) is a plain
  device-to-device copy; other ranks read over NVLink/P2P. No per-request
  ``cudaIpcOpenMemHandle`` / ``cudaIpcCloseMemHandle``.

- Recycling: each in-flight slice owns a row of lock-free per-consumer flag
  cells in a shared-memory segment. Every worker that reads the slice sets its
  own cell (indexed by device); the sender reclaims the slice once
  ``tensor_parallel_size`` distinct cells are set. A distinct cell per consumer
  needs no cross-process lock.

Requirements: the sender and all TP workers must share the same GPUs or have
peer access enabled (true on NVLink systems; the v1 ``MultiprocExecutor`` does
not isolate ``CUDA_VISIBLE_DEVICES``). CPU tensors are rejected (the encoder
falls back to the regular serialization path), so this only engages for
GPU-preprocessed multimodal features (e.g. ``mm_processor_kwargs={"device":
"cuda"}``).
"""

import atexit
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.serial_utils import OOBTensorConsumer

logger = init_logger(__name__)

# Byte alignment for pool slices.
_ALIGN = 512
# Number of concurrent in-flight slices (flag rows).
_NUM_SLOTS = 8192
# Max tensor-parallel width supported (flag columns / per-consumer cells).
_MAX_TP = 16


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


class _SenderPool:
    """Persistent IPC-exported GPU pool (frontend side).

    A free-list allocator hands out slices; lock-free per-consumer cells drive
    recycling once every worker has read a slice.
    """

    def __init__(self, device_index: int, pool_bytes: int, tp_size: int):
        self.device_index = device_index
        self.pool_bytes = pool_bytes
        self.tp_size = max(1, tp_size)
        if self.tp_size > _MAX_TP:
            # Wider worlds would alias per-device recycling cells; cuda_ipc
            # targets single-node TP (shared GPUs / P2P).
            raise ValueError(
                f"mm_tensor_ipc='cuda_ipc' supports up to tensor_parallel_size="
                f"{_MAX_TP} (got {self.tp_size}); use a host-based transport for "
                "wider or multi-node parallelism."
            )
        self.buf = torch.empty(
            pool_bytes, dtype=torch.uint8, device=f"cuda:{device_index}"
        )
        # Exported once; the same handle is reused for every request so each
        # consumer only maps it a single time.
        self.handle = self.buf.untyped_storage()._share_cuda_()

        self._free: list[tuple[int, int]] = [(0, pool_bytes)]
        self._occupied: dict[int, tuple[int, int]] = {}
        self._lock = threading.Lock()

        # Lock-free recycling cells: flags[slot, device] == 1 once the worker on
        # ``device`` has copied the slice out. A slice is reclaimed once
        # ``tp_size`` distinct cells in its row are set.
        self.flags_shm = shared_memory.SharedMemory(
            create=True, size=_NUM_SLOTS * _MAX_TP * 4
        )
        self.flags = np.ndarray(
            (_NUM_SLOTS, _MAX_TP), dtype=np.int32, buffer=self.flags_shm.buf
        )
        self.flags[:] = 0
        self._free_slots: list[int] = list(range(_NUM_SLOTS))

        self._stop = False
        self._recycler = threading.Thread(
            target=self._recycle_loop, name="MmCudaIpcRecycler", daemon=True
        )
        self._recycler.start()
        atexit.register(self.close)
        logger.info(
            "Multimodal CUDA-IPC sender pool ready: %d MiB on cuda:%d, "
            "tp_size=%d (persistent handle exported once).",
            pool_bytes // (1024 * 1024), device_index, self.tp_size,
        )

    def _alloc_region(self, need: int) -> tuple[int, int] | None:
        aligned = (need + _ALIGN - 1) // _ALIGN * _ALIGN
        for i, (s, e) in enumerate(self._free):
            if e - s >= aligned:
                if e - s == aligned:
                    self._free.pop(i)
                else:
                    self._free[i] = (s + aligned, e)
                return s, s + aligned
        return None

    def try_put(self, src_u8: torch.Tensor) -> tuple[int, int, int] | None:
        """Copy ``src_u8`` (1-D contiguous uint8 CUDA tensor) into the pool.

        Returns ``(offset, nbytes, slot)`` or ``None`` if the pool is full.
        """
        n = src_u8.numel()
        with self._lock:
            if not self._free_slots:
                return None
            reg = self._alloc_region(n)
            if reg is None:
                return None
            slot = self._free_slots.pop()
            start, _ = reg
            self._occupied[slot] = reg
            self.flags[slot, :] = 0
        self.buf[start:start + n].copy_(src_u8)
        # Block until the async copy lands so a consumer never reads a
        # half-written slice (no cross-process stream ordering); mirrors the
        # GPU-decode path in vllm/multimodal/video.py.
        torch.cuda.current_stream(self.device_index).synchronize()
        return start, n, slot

    def _recycle_loop(self):
        while not self._stop:
            try:
                with self._lock:
                    done = [
                        slot for slot in self._occupied
                        if int(self.flags[slot].sum()) >= self.tp_size
                    ]
                    for slot in done:
                        self._free.append(self._occupied.pop(slot))
                        self._free_slots.append(slot)
                        self.flags[slot, :] = 0
                    if done:
                        self._coalesce()
            except Exception as e:  # noqa: BLE001
                logger.warning("Multimodal CUDA-IPC recycler error: %s", e)
            time.sleep(0.02)

    def _coalesce(self):
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for s, e in self._free:
            if merged and merged[-1][1] == s:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        self._free = merged

    def close(self):
        self._stop = True
        try:
            self.flags_shm.close()
            self.flags_shm.unlink()
        except Exception:  # noqa: BLE001
            pass


# Per-worker caches so the pool handle and flags segment are mapped once.
_recv_storage_cache: dict[tuple, Any] = {}
_recv_flags_cache: dict[str, Any] = {}
_recv_lock = threading.Lock()


def _open_pool_storage(handle: tuple, device_index: int):
    # Redirect the storage device index to the consumer's device so the CUDA
    # guard inside ``_new_shared_cuda`` stays on the right GPU. The physical
    # memory lives on the sender's GPU; for a different device this maps it via
    # peer access (P2P).
    redirected = (device_index,) + tuple(handle)[1:]
    key = (device_index, tuple(handle))
    st = _recv_storage_cache.get(key)
    if st is None:
        with _recv_lock:
            st = _recv_storage_cache.get(key)
            if st is None:
                with torch.cuda.device(device_index):
                    st = torch.UntypedStorage._new_shared_cuda(*redirected)
                _recv_storage_cache[key] = st
                logger.info(
                    "Multimodal CUDA-IPC receiver mapped pool handle once on "
                    "cuda:%d (cached for all subsequent requests).", device_index)
    return st


def _open_flags(name: str):
    fl = _recv_flags_cache.get(name)
    if fl is None:
        with _recv_lock:
            fl = _recv_flags_cache.get(name)
            if fl is None:
                shm = shared_memory.SharedMemory(name=name)
                arr = np.ndarray(
                    (_NUM_SLOTS, _MAX_TP), dtype=np.int32, buffer=shm.buf
                )
                fl = (shm, arr)
                _recv_flags_cache[name] = fl
    return fl[1]


def _signal_consumed(flags_shm_name: str, slot: int, device_index: int) -> None:
    """Mark this consumer's cell for ``slot`` as read (lock-free, idempotent)."""
    cell = device_index if 0 <= device_index < _MAX_TP else 0
    try:
        _open_flags(flags_shm_name)[slot, cell] = 1
    except FileNotFoundError:
        pass  # Flags segment gone (sender shutting down).


class CudaIpcPoolProxy:
    """Metadata to reopen a pooled slice on the consumer's device.

    Holds no CUDA tensor, so it pickles cheaply across the engine-core->worker
    boundary. :meth:`reconstruct` opens the (cached) pool handle and copies the
    slice out on the target device.
    """

    __slots__ = (
        "pool_handle", "flags_shm", "slot", "offset", "nbytes", "shape", "dtype",
    )

    def __init__(self, pool_handle, flags_shm, slot, offset, nbytes, shape,
                 dtype):
        self.pool_handle = pool_handle
        self.flags_shm = flags_shm
        self.slot = slot
        self.offset = offset
        self.nbytes = nbytes
        self.shape = shape
        self.dtype = dtype

    def reconstruct(self, device: int | None = None) -> torch.Tensor:
        dev = torch.cuda.current_device() if device is None else device
        storage = _open_pool_storage(self.pool_handle, dev)
        dtype = getattr(torch, self.dtype)
        slice_storage = storage[self.offset:self.offset + self.nbytes]
        view = torch.empty(0, dtype=dtype, device=f"cuda:{dev}")
        view.set_(slice_storage, 0, self.shape, _contiguous_stride(self.shape))
        # Copy out so we own the memory; the pool slice can then be recycled.
        # For a different device this is a P2P (device-to-device) copy.
        out = view.clone()
        try:
            _signal_consumed(self.flags_shm, self.slot, dev)
        except Exception as e:  # noqa: BLE001
            logger.warning("Multimodal CUDA-IPC recycle-signal failed: %s", e)
        return out


def is_cuda_ipc_proxy(obj: Any) -> bool:
    return isinstance(obj, CudaIpcPoolProxy)


class CudaIpcTensorSender(OOBTensorConsumer):
    """OOB consumer: copy a CUDA tensor into the pool and return proxy metadata.

    Returns ``None`` (encoder falls back to the host-copy path) for CPU tensors
    or when the pool is unavailable/full.
    """

    def __init__(self, pool_bytes: int, tp_size: int):
        self._pool_bytes = pool_bytes
        self._tp_size = tp_size
        self._pool: _SenderPool | None = None
        self._pool_lock = threading.Lock()

    def _get_pool(self, device_index: int) -> _SenderPool | None:
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is None:
                try:
                    self._pool = _SenderPool(
                        device_index, self._pool_bytes, self._tp_size
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to create multimodal CUDA-IPC pool (%s); "
                        "falling back to host-copy transport.", e)
                    return None
            return self._pool

    def __call__(self, tensor: torch.Tensor) -> dict | None:
        if not tensor.is_cuda:
            return None
        pool = self._get_pool(tensor.get_device())
        if pool is None:
            return None
        src = tensor.detach().contiguous()
        res = pool.try_put(src.reshape(-1).view(torch.uint8))
        if res is None:
            return None
        offset, nbytes, slot = res
        # Only plain scalars/bytes go into the message (msgpack-native, no
        # pickle): the CUDA IPC handle is a tuple of ints/bytes/bool.
        return {
            "cuda_ipc": {
                "handle": list(pool.handle),
                "flags_shm": pool.flags_shm.name,
                "slot": slot,
                "offset": offset,
                "nbytes": nbytes,
            }
        }

    def new_message(self) -> None:
        pass


def cuda_ipc_provider(dtype: str, shape: tuple[int, ...], meta: dict) -> Any:
    """OOB provider: deferred decode.

    Rebuilds the lightweight :class:`CudaIpcPoolProxy` from plain metadata
    WITHOUT opening the handle, so it can travel through engine-core and the
    engine-core->worker broadcast as metadata; each worker materializes the
    on-device tensor itself via :meth:`CudaIpcPoolProxy.reconstruct` (see
    ``reduce_data`` in ``inputs.py``).
    """
    m = meta["cuda_ipc"]
    return CudaIpcPoolProxy(
        pool_handle=tuple(m["handle"]),
        flags_shm=m["flags_shm"],
        slot=m["slot"],
        offset=m["offset"],
        nbytes=m["nbytes"],
        shape=tuple(shape),
        dtype=dtype,
    )
