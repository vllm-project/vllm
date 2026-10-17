# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ctypes
import errno
import functools
import mmap
import os
import shutil
import time
from collections.abc import Callable, Iterator
from ctypes import c_int, c_long, c_size_t, c_void_p
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_offload.cpu.host_register import host_unregister

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_MAP_FIXED = 0x10
# Page-cache inserts are serialized per shm file, capping pre-faulting of one
# file near 10 GiB/s; smaller segments are not worth a file.
_MIN_SEGMENT_BYTES = 1 << 30


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if os.fstat(fd).st_nlink == 0:
            raise RuntimeError(
                "Shared offload region creator failed during initialization."
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def check_local_engines_offload_shm(
    vllm_config: "VllmConfig", local_engine_count: int
) -> None:
    """Refuse CPU offload sizes that the engines about to start on this host
    cannot hold in /dev/shm together.

    Each engine checks free space only for its own region, and ftruncate
    reserves nothing, so the combined size is compared with /dev/shm's total
    size where the engines are launched. Specs that keep their CPU buffer out
    of /dev/shm are skipped.
    """
    if local_engine_count < 2 or not os.path.isdir(SHM_PATH):
        return
    ktc = vllm_config.kv_transfer_config
    if ktc is None or ktc.kv_connector != "OffloadingConnector":
        return
    extra_config = ktc.kv_connector_extra_config
    cpu_bytes = int(extra_config.get("cpu_bytes_to_use") or 0)
    if not cpu_bytes:
        return

    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    spec_cls = OffloadingSpecFactory.get_spec_cls(extra_config)
    uses_shared_region = getattr(spec_cls, "_uses_shared_region", None)
    if uses_shared_region is None or not uses_shared_region():
        return

    total = shutil.disk_usage(SHM_PATH).total
    needed = cpu_bytes * local_engine_count
    if needed > total:
        gib = 1 << 30
        raise ValueError(
            f"{SHM_PATH} cannot hold the CPU KV offload buffers: "
            f"{local_engine_count} data-parallel engines on this host each need "
            f"{cpu_bytes / gib:.1f} GiB ({needed / gib:.1f} GiB in total), but "
            f"{SHM_PATH} is {total / gib:.1f} GiB. Lower --kv-offloading-size or "
            f"enlarge {SHM_PATH} (e.g. --shm-size for containers)."
        )


@functools.cache
def _libc_mmap() -> Callable[..., int | None]:
    mmap_fn = ctypes.CDLL(None, use_errno=True).mmap
    mmap_fn.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_long]
    mmap_fn.restype = c_void_p
    return mmap_fn


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


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
    """Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap.  When a barrier is
    given, the path is unlinked once every worker has mapped the file, so
    the kernel reclaims the memory when the last worker exits, no matter
    how it exits; mappings taken before the unlink stay valid.

    ``populate_shard=(index, count)`` pre-faults only this worker's contiguous
    share, for layouts where every worker maps the same bytes. The barrier
    also waits for all shards to finish pre-faulting before returning.
    ``num_segments`` splits backing files beneath the contiguous mapping.
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_chunks: int,
        rank: int | None,
        kv_bytes_per_chunk: int,
        cpu_page_size: int,
        barrier: Callable[[], None] | None = None,
        *,
        creator_memory_check: Callable[[int], None] | None = None,
        populate_shard: tuple[int, int] | None = None,
        num_segments: int = 1,
        shm_allocation_name: str = "CPU KV offload shared region",
        shm_size_option: str = "--kv-offloading-size",
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_chunk % self.page_size == 0

        self.num_chunks = num_chunks
        self._row_stride = kv_bytes_per_chunk
        self.total_size_bytes = self.num_chunks * self._row_stride
        self.cpu_page_size = cpu_page_size
        self._shm_allocation_name = shm_allocation_name
        self._shm_size_option = shm_size_option

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        num_segments = min(num_segments, self.total_size_bytes // _MIN_SEGMENT_BYTES)
        rows = max(1, cdiv(num_chunks, max(1, num_segments)))
        # (path, offset, length) per file.
        self._segments = [
            (
                f"{self.mmap_path}.{i}" if i else self.mmap_path,
                row * self._row_stride,
                min(rows, num_chunks - row) * self._row_stride,
            )
            for i, row in enumerate(range(0, max(num_chunks, 1), rows))
        ]
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each chunk row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        barriers_remaining = 2 if populate_shard is not None else 1
        try:
            try:
                self.fd: int | None = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                # Joiner path — another worker won O_EXCL. Reopen and wait
                # for the file to reach expected size.
                self.fd = os.open(self.mmap_path, os.O_RDWR)
                _wait_for_file_size(self.fd, self._segments[0][2])
                logger.info("Opened existing mmap file %s", self.mmap_path)
            else:
                # Creator path. We won O_EXCL, so we own the file: any
                # failure here must clean up so concurrent joiners don't
                # land on a 0-byte stub and spin in _wait_for_file_size
                # for the full 30 s timeout.
                self._creator = True
                if creator_memory_check is not None:
                    creator_memory_check(self.total_size_bytes)
                check_shm_free_space(
                    self.total_size_bytes,
                    allocation_name=f"{shm_allocation_name} in /dev/shm",
                )
                for path, _, length in self._segments[1:]:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    os.ftruncate(fd, length)
                    os.close(fd)
                # Joiners wait for the first file, so size it last.
                os.ftruncate(self.fd, self._segments[0][2])
                logger.info(
                    "Created mmap file %s (%.2f GB)",
                    self.mmap_path,
                    self.total_size_bytes / 1e9,
                )

            single = len(self._segments) == 1
            self.mmap_obj: mmap.mmap | None = mmap.mmap(
                self.fd if single else -1,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED if single else mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            if not single:
                self._map_segments()
            if barrier is not None:
                barriers_remaining -= 1
                barrier()
                if self._creator:
                    self._unlink()
                    self._creator = False
                    logger.info("Unlinked mmap file %s", self.mmap_path)
            if populate_shard is not None:
                with self._shm_exhaustion_as_error():
                    populate_write_fn = _get_populate_write_fn(self.mmap_obj)
                    index, count = populate_shard
                    share = round_up(cdiv(self.total_size_bytes, count), self.page_size)
                    start = min(index * share, self.total_size_bytes)
                    if start < self.total_size_bytes:
                        length = min(share, self.total_size_bytes - start)
                        populate_write_fn(self.mmap_obj, start, length)
                if barrier is not None:
                    barriers_remaining -= 1
                    barrier()
        except Exception:
            if self._creator:
                self._unlink()
                self._creator = False
            if hasattr(self, "mmap_obj") and self.mmap_obj is not None:
                # The prefault traceback may retain an exported buffer.
                with contextlib.suppress(BufferError):
                    self.mmap_obj.close()
                self.mmap_obj = None
            if hasattr(self, "fd") and self.fd is not None:
                os.close(self.fd)
                self.fd = None
            # Join remaining barriers so peers do not wait until the collective
            # times out. A barrier failure must not replace the setup error.
            if barrier is not None:
                for _ in range(barriers_remaining):
                    try:
                        barrier()
                    except Exception:
                        logger.warning(
                            "Failed to release peers waiting at the mmap barrier",
                            exc_info=True,
                        )
            raise

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self.is_pinned: bool = False
        self.pinned_addresses: list[int] = []

        if populate_shard is not None:
            return
        with self._shm_exhaustion_as_error():
            self._populate()

    def _populate(self) -> None:
        mmap_obj = self.mmap_obj
        assert mmap_obj is not None
        populate_write_fn = _get_populate_write_fn(mmap_obj)

        if self.rank is not None:
            # Populate only this worker's pages (one slot per chunk row).
            _t0 = time.perf_counter()
            for offset, length in self.iter_worker_slot_ranges():
                populate_write_fn(mmap_obj, offset, length)
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d chunks in %.3f s",
                self.num_chunks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            populate_write_fn(mmap_obj, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

    @contextlib.contextmanager
    def _shm_exhaustion_as_error(self) -> Iterator[None]:
        """MADV_POPULATE_WRITE reports a full tmpfs as EFAULT ("Bad address"),
        which says nothing about /dev/shm."""
        try:
            yield
        except OSError as e:
            if e.errno != errno.EFAULT:
                raise
            gib = 1 << 30
            where = ""
            if os.path.isdir(SHM_PATH):
                usage = shutil.disk_usage(SHM_PATH)
                where = f"; {SHM_PATH} has {usage.free / gib:.1f} GiB free"
                where += f" of {usage.total / gib:.1f} GiB"
            raise RuntimeError(
                f"Ran out of {SHM_PATH} while pre-faulting the "
                f"{self._shm_allocation_name} "
                f"({self.total_size_bytes / gib:.1f} GiB{where}). {SHM_PATH} is "
                f"shared by every process on this host, including other "
                f"data-parallel engines. Lower {self._shm_size_option} or "
                f"enlarge {SHM_PATH}."
            ) from e

    def _map_segments(self) -> None:
        """Replace the anonymous mapping with the segment files, back to back."""
        assert self.mmap_obj is not None and self.fd is not None
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap_obj))
        for path, offset, length in self._segments:
            fd = self.fd if offset == 0 else os.open(path, os.O_RDWR)
            address = _libc_mmap()(
                base + offset,
                length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                fd,
                0,
            )
            if fd != self.fd:
                os.close(fd)
            if address != base + offset:
                raise OSError(ctypes.get_errno(), f"Failed to map {path}")

    def _unlink(self) -> None:
        for path, _, _ in self._segments:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    def iter_worker_slot_ranges(self) -> Iterator[tuple[int, int]]:
        """Yield this worker's slots as page-aligned (offset, length) ranges."""
        assert self.rank is not None
        worker_offset = self.rank * self.cpu_page_size
        first = 0
        if len(self._segments) > 1:
            # Start each worker on a different file.
            first = (
                self.rank * self.num_chunks // (self._row_stride // self.cpu_page_size)
            )
        for i in range(self.num_chunks):
            row = (first + i) % self.num_chunks
            start = row * self._row_stride + worker_offset
            aligned = start - start % self.page_size
            end = round_up(start + self.cpu_page_size, self.page_size)
            yield aligned, end - aligned

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("Shared offload region has been released.")
        return self._base

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_chunk0 | worker1_chunk0 | ... | worker{M-1}_chunk0
            worker0_chunk1 | worker1_chunk1 | ... | worker{M-1}_chunk1
            ...

        Each worker_chunk cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and chunk concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_chunks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per chunk for this tensor.

        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
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
        chunk. Canonical areas are carved consecutively from the start of
        each chunk row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per chunk for this tensor.

        """
        new_offset = self._canonical_offset + tensor_page_size
        assert new_offset <= self._row_stride
        view = torch.as_strided(
            self._base,
            size=(self.num_chunks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_chunks, row_stride_bytes). Secondary tiers address
        chunk *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_chunks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            base_ptr = self._base.data_ptr()
            addresses = self.pinned_addresses or [base_ptr]
            for address in reversed(addresses):
                host_unregister(address)
            self.pinned_addresses.clear()
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            retained = False
            try:
                self.mmap_obj.close()
            except BufferError:
                # A native user still exports this mapping, so unmapping would
                # let the NIC write into reused address space. Keep it mapped;
                # dropping the fd and the pathname below is still safe.
                retained = True
                logger.warning(
                    "Keeping %s mapped: buffers are still exported to native users",
                    self.mmap_path,
                )
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            if not retained:
                self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                self._unlink()
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False
