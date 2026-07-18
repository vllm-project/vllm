# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import glob
import json
import mmap
import os
import time
from typing import cast

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

Metadata = dict[str, int | bool]
_LAYOUT_METADATA_FIELDS = (
    "num_blocks",
    "row_stride",
    "slot_bytes",
    "replicated_layout",
)
_METADATA_TIMEOUT = 30.0


def _stale_files_hint(mmap_path: str, metadata_path: str) -> str:
    return (
        "if no other vLLM instance on this host is using either path, "
        f"remove only the exact stale files {mmap_path} and {metadata_path}, "
        "then restart"
    )


def _unlink_path(path: str, description: str) -> None:
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
            logger.info("Removed %s %s", description, path)
    except Exception:
        logger.warning("Failed to unlink %s %s", description, path, exc_info=True)


def _cleanup_region_files(mmap_path: str, metadata_path: str) -> None:
    _unlink_path(metadata_path, "metadata sidecar")
    for tmp_path in glob.glob(f"{metadata_path}.*.tmp"):
        _unlink_path(tmp_path, "metadata tmp")
    _unlink_path(mmap_path, "mmap file")


def _wait_for_file_size(
    fd: int,
    expected_size: int,
    timeout: float = 30.0,
    mmap_path: str | None = None,
    metadata_path: str | None = None,
) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            path_text = f" for mmap {mmap_path}" if mmap_path else ""
            hint = (
                f"; {_stale_files_hint(mmap_path, metadata_path)}"
                if mmap_path and metadata_path
                else ""
            )
            raise TimeoutError(
                f"Timed out waiting for mmap file{path_text} to reach "
                f"{expected_size} bytes{hint}"
            )
        time.sleep(0.005)


def _wait_for_metadata(paths: tuple[str, str], deadline: float) -> Metadata:
    """Spin-wait until the region metadata sidecar is visible."""
    metadata_path, mmap_path = paths
    while True:
        try:
            with open(metadata_path) as f:
                return cast(Metadata, json.load(f))
        except FileNotFoundError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for mmap metadata {metadata_path} "
                    f"for mmap {mmap_path}; "
                    f"{_stale_files_hint(mmap_path, metadata_path)}"
                ) from None
            time.sleep(0.005)
        except json.JSONDecodeError as e:
            if time.monotonic() > deadline:
                raise ValueError(
                    f"Invalid mmap metadata sidecar {metadata_path} for mmap "
                    f"{mmap_path}: {e}; "
                    f"{_stale_files_hint(mmap_path, metadata_path)}"
                ) from e
            time.sleep(0.005)


def _write_metadata(metadata_path: str, metadata: Metadata) -> None:
    """Atomically write region metadata to a JSON sidecar."""
    tmp_path = f"{metadata_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, metadata_path)


def _check_metadata(paths: tuple[str, str], fd: int, expected: Metadata) -> None:
    """Validate region metadata written by the region creator."""
    metadata_path, mmap_path = paths
    deadline = time.monotonic() + _METADATA_TIMEOUT
    mmap_inode = os.fstat(fd).st_ino
    while True:
        actual = _wait_for_metadata(paths, deadline)
        sidecar_inode = actual.get("mmap_inode")
        if sidecar_inode == mmap_inode:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap metadata {metadata_path} to match "
                f"mmap {mmap_path} inode {mmap_inode}; "
                f"sidecar mmap_inode={sidecar_inode!r}; "
                f"{_stale_files_hint(mmap_path, metadata_path)}"
            ) from None
        time.sleep(0.005)

    for field in _LAYOUT_METADATA_FIELDS:
        expected_value = expected[field]
        actual_value = actual.get(field)
        if actual_value != expected_value:
            raise ValueError(
                "Shared offload region metadata mismatch for "
                f"{field}: local={expected_value!r}, sidecar={actual_value!r} "
                f"at {metadata_path} for mmap {mmap_path}; "
                f"{_stale_files_hint(mmap_path, metadata_path)}"
            )


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
        replicated_layout: bool = False,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride
        self._replicated_layout = replicated_layout

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self.metadata_path = f"/dev/shm/vllm_offload_{engine_id}.meta.json"
        metadata: Metadata = {
            "num_blocks": num_blocks,
            "row_stride": kv_bytes_per_block,
            "slot_bytes": cpu_page_size,
            "replicated_layout": replicated_layout,
        }
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        try:
            # Exclusive create — only one worker succeeds.
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            if self._replicated_layout:
                try:
                    _check_metadata(
                        (self.metadata_path, self.mmap_path), self.fd, metadata
                    )
                    _wait_for_file_size(
                        self.fd,
                        self.total_size_bytes,
                        mmap_path=self.mmap_path,
                        metadata_path=self.metadata_path,
                    )
                except Exception:
                    os.close(self.fd)
                    self.fd = None
                    raise
            else:
                _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.info("Opened existing mmap file %s", self.mmap_path)
        else:
            try:
                os.ftruncate(self.fd, self.total_size_bytes)
                if self._replicated_layout:
                    creator_metadata = {
                        **metadata,
                        "mmap_inode": os.fstat(self.fd).st_ino,
                    }
                    _write_metadata(self.metadata_path, creator_metadata)
            except Exception:
                if self.fd is not None:
                    try:
                        os.close(self.fd)
                    except Exception:
                        logger.warning("Failed to close fd %s", self.fd, exc_info=True)
                    self.fd = None
                if self._replicated_layout:
                    _cleanup_region_files(self.mmap_path, self.metadata_path)
                else:
                    _unlink_path(self.mmap_path, "mmap file")
                raise
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
        _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                self.mmap_obj.madvise(
                    _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                )
            logger.debug(
                "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            self.mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, self.total_size_bytes)
            logger.debug(
                "MADV_POPULATE_WRITE entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
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

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                base_ptr = self._base.data_ptr()
                result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.is_pinned = False
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
            if self._replicated_layout:
                _cleanup_region_files(self.mmap_path, self.metadata_path)
            else:
                try:
                    os.unlink(self.mmap_path)
                    logger.info("Removed mmap file %s", self.mmap_path)
                except Exception:
                    logger.warning(
                        "Failed to unlink path %s", self.mmap_path, exc_info=True
                    )
            self._creator = False
