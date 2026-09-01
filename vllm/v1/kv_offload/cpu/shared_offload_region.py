# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import errno
import mmap
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.host_memory import (
    HostRegisterError,
    host_register_chunked,
    host_unregister,
    madvise,
)

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Goes through ctypes rather than `mmap.madvise()` because CPython holds
    # the GIL across that syscall, which would stall the engine when
    # population runs on a background thread (see AsyncHostBuffer).
    base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mmap_obj))
    madvise(base_ptr + offset, length, _MADV_POPULATE_WRITE)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap.  When a barrier is
    given, the path is unlinked once every worker has mapped the file, so
    the kernel reclaims the memory when the last worker exits, no matter
    how it exits; mappings taken before the unlink stay valid.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        populate: bool = True,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        # Set before anything that can raise, so cleanup() is safe to call on a
        # partially built region.
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None
        self._base: torch.Tensor | None = None
        self._views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self._registered_ptrs: list[int] = []
        self.is_pinned: bool = False

        try:
            try:
                self.fd = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                try:
                    _wait_for_file_size(self.fd, self.total_size_bytes)
                except (TimeoutError, OSError):
                    os.close(self.fd)
                    raise
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                try:
                    check_shm_free_space(self.total_size_bytes)
                    os.ftruncate(self.fd, self.total_size_bytes)
                except (RuntimeError, OSError):
                    os.unlink(self.mmap_path)
                    os.close(self.fd)
                    raise
                self._creator = True
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )
            self.mmap_obj = mmap.mmap(
                self.fd,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        except BaseException:
            if self._creator:
                os.unlink(self.mmap_path)
                self._creator = False
            # Peers block inside the barrier until the collective times out if
            # we die before reaching it.  Arrive anyway so every worker calls
            # barrier() exactly once and they fail on their own errors instead
            # of hanging; a failure here must not replace ours.
            if barrier is not None:
                try:
                    barrier()
                except Exception:
                    logger.warning(
                        "Failed to release peers waiting at the mmap barrier",
                        exc_info=True,
                    )
            # This mapping can be hundreds of GB; never strand it.
            self.cleanup()
            raise

        if barrier is not None:
            # Every worker has mapped the file once the barrier releases, so
            # its name is no longer needed and dropping it here means no exit
            # path — including SIGKILL — can leak the file.
            try:
                barrier()
            except BaseException:
                if self._creator:
                    os.unlink(self.mmap_path)
                    self._creator = False
                self.cleanup()
                raise
            if self._creator:
                os.unlink(self.mmap_path)
                self._creator = False
                logger.info("Unlinked mmap file %s", self.mmap_path)

        if populate:
            self.populate(cpu_page_size)

    def populate(self, cpu_page_size: int) -> None:
        """Fault in writable pages up front, so transfers do not pay for them.

        This does not CUDA-pin anything; see `pin`. Uses MADV_POPULATE_WRITE
        where supported, falling back to per-page writes on older kernels
        (see `_get_populate_write_fn`).
        """
        mmap_obj = self.mmap_obj
        assert mmap_obj is not None
        populate_write_fn = _get_populate_write_fn(mmap_obj)
        _t0 = time.perf_counter()

        if self.rank is None:
            populate_write_fn(mmap_obj, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )
            return

        # Populate only this worker's strided slot in each block row.
        worker_offset = self.rank * cpu_page_size
        page_size = self.page_size
        for block in range(self.num_blocks):
            raw_offset = block * self._row_stride + worker_offset
            aligned_offset = (raw_offset // page_size) * page_size
            aligned_length = raw_offset + cpu_page_size - aligned_offset
            populate_write_fn(mmap_obj, aligned_offset, aligned_length)
        logger.debug(
            "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
            self.num_blocks,
            time.perf_counter() - _t0,
        )

    def pin(self, chunk_size_bytes: int | None = None) -> None:
        """Page-lock the region so DMA transfers skip the bounce buffer.

        Idempotent, and paired with the cudaHostUnregister in `cleanup`. Failure
        is a warning: unpinned transfers still work, they are just slower.

        Args:
            chunk_size_bytes: When set, register the region incrementally and
                yield between chunks so inference can submit CUDA work.
        """
        if self.is_pinned:
            return
        if not current_platform.is_cuda_alike():
            logger.info(
                "Skipping mmap host registration on %s; cudaHostRegister is only "
                "available on CUDA/ROCm.",
                current_platform.device_name,
            )
            return

        assert self._base is not None
        try:
            self._registered_ptrs = host_register_chunked(
                self._base.data_ptr(), self.total_size_bytes, chunk_size_bytes
            )
        except HostRegisterError as e:
            logger.warning(
                "%s for rank=%d — transfers will still work but may be "
                "slower (unpinned DMA)",
                e,
                self.rank,
            )
            return
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB in %d range(s)",
            self.rank,
            self.total_size_bytes / 1e9,
            len(self._registered_ptrs),
        )
        self.is_pinned = True

    def _unregister_pinned_ranges(self) -> None:
        if current_platform.is_cuda_alike():
            host_unregister(self._registered_ptrs)
        self._registered_ptrs.clear()
        self.is_pinned = False

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        assert self._base is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view shared by all workers for one
        canonical tensor (canonical layout).

        Must be called once per canonical tensor, instead of
        create_next_worker_view. The full mmap layout is:

            |<-------- canonical area ------->|<-------- unused ------->|
            |  all workers share this area    |                         |
            |                                 |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            ^                ^
            _canonical_offset=0, then advances by each tensor's size

        Each canonical_t{i} cell is that tensor's canonical page for the
        block. Canonical areas are carved consecutively from the start of
        each block row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per block for this tensor.
        """
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        assert self._base is not None
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self._registered_ptrs:
            self._unregister_pinned_ranges()
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
