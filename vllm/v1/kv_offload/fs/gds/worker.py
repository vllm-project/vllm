# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import contextlib
import os
import uuid
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.gds.cufile_bindings import (
    CUfileBoolConfigParameter,
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileRead,
    cuFileSetParameterBool,
    cuFileWrite,
    open_for_gds,
)
from vllm.v1.kv_offload.fs.worker import FileOp, ThreadPoolFSWorker

logger = init_logger(__name__)


class GDSOffloadingWorker(ThreadPoolFSWorker):
    """GPU<->NVMe transfers via NVIDIA cuFile synchronous API.

    Each file's I/O is submitted to a thread pool so multiple operations
    overlap on the NVMe device (higher queue depth = higher throughput).

    All file syscalls (open, cuFileHandleRegister, I/O,
    cuFileHandleDeregister, close) run inside pool threads to keep the
    scheduler thread free of blocking operations.  Driver open/close and
    the direct-path probe (_validate_gds) run on the caller's
    thread at init/shutdown.

    Compat mode is disabled, so cuFile cannot silently fall back to POSIX
    I/O through bounce buffers.  At init the worker round-trips one KV
    block through root_dir to prove the direct path works on that mount.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        blocks_per_file: int,
        device: torch.device,
        kv_caches: CanonicalKVCaches,
        root_dir: str,
        extra_config: dict[str, Any],
    ):
        if not current_platform.is_cuda():
            raise RuntimeError("GDS offloading requires an NVIDIA CUDA GPU.")
        io_threads = int(extra_config.get("gds_max_io_threads", 32))
        super().__init__(
            file_mapper=file_mapper,
            blocks_per_file=blocks_per_file,
            device=device,
            max_io_threads=io_threads,
        )
        self._driver_open = False
        try:
            self._validate_gds(device, kv_caches, root_dir, io_threads)
        except Exception:
            self.shutdown_dispatch()
            self.shutdown_backend()
            raise

    def _validate_gds(
        self,
        device: torch.device,
        kv_caches: CanonicalKVCaches,
        root_dir: str,
        io_threads: int,
    ) -> None:
        """Open cuFile with compat mode off and prove the direct path works.

        Takes one KV block (block 0 of the first KV tensor), writes it to
        a temp file under root_dir, reads it back, and deletes the file.  With
        compat mode off cuFile has no POSIX fallback, so success means
        GPU<->storage DMA works on this mount.  The data is not checked, only
        that both transfers complete.  This runs before the KV cache holds
        any data, so the read may overwrite the block freely.
        """
        # Staged parameters only apply at driver load, so this must come
        # before cuFileDriverOpen.
        cuFileSetParameterBool(
            CUfileBoolConfigParameter.CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE,
            False,
        )
        cuFileDriverOpen()
        self._driver_open = True

        # A uint8 view of block 0 of the first KV tensor.
        kv = kv_caches.tensors[0]
        size = kv.page_size_bytes
        storage = kv.tensor.untyped_storage()
        block = torch.empty(0, dtype=torch.uint8, device=device)
        block.set_(storage, kv.tensor.data_ptr() - storage.data_ptr(), (size,))
        ptr = block.data_ptr()

        os.makedirs(root_dir, exist_ok=True)
        path = os.path.join(root_dir, f".gds_probe_{uuid.uuid4().hex}.tmp")
        with torch.accelerator.device_index(device.index):
            try:
                fd = open_for_gds(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    handle = cuFileHandleRegister(fd)
                    try:
                        ret = cuFileWrite(handle, ptr, size, 0)
                    finally:
                        cuFileHandleDeregister(handle)
                finally:
                    os.close(fd)
                if ret != size:
                    raise RuntimeError(f"cuFileWrite short write: {ret}/{size}")

                fd = open_for_gds(path, os.O_RDONLY)
                try:
                    handle = cuFileHandleRegister(fd)
                    try:
                        ret = cuFileRead(handle, ptr, size, 0)
                    finally:
                        cuFileHandleDeregister(handle)
                finally:
                    os.close(fd)
                if ret != size:
                    raise RuntimeError(f"cuFileRead short read: {ret}/{size}")
            except Exception as e:
                raise RuntimeError(
                    f"GDS direct-path probe failed under root_dir={root_dir!r} "
                    f"with compat mode disabled: {e}. Run `gdscheck -p` to check "
                    "that GDS is supported for this filesystem and GPU."
                ) from e
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
        logger.info("GDS: direct path verified (compat disabled)")

    def write_block(self, file_path: str, ops: list[FileOp]) -> None:
        """Write to a temp file, then atomically rename it into place, so
        readers never see a partially written file."""
        # File names are content hashes: an existing file holds this block.
        if os.path.exists(file_path):
            return
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = f"{file_path}_{uuid.uuid4().hex}.tmp"
        try:
            fd = open_for_gds(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                handle = cuFileHandleRegister(fd)
                try:
                    for op in ops:
                        ret = cuFileWrite(
                            handle, op.device_ptr, op.size_bytes, op.file_offset
                        )
                        if ret != op.size_bytes:
                            raise RuntimeError(
                                f"cuFileWrite short write: {ret}/{op.size_bytes}"
                            )
                finally:
                    cuFileHandleDeregister(handle)
            finally:
                os.close(fd)
            try:
                os.replace(tmp_path, file_path)
            except FileExistsError:
                # Some shared filesystems fail the rename with EEXIST instead
                # of replacing; another writer already published this block.
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to remove temp file %s: %s", tmp_path, cleanup_exc
                )
            raise

    def read_block(self, file_path: str, ops: list[FileOp]) -> None:
        fd = open_for_gds(file_path, os.O_RDONLY)
        try:
            handle = cuFileHandleRegister(fd)
            try:
                for op in ops:
                    ret = cuFileRead(
                        handle, op.device_ptr, op.size_bytes, op.file_offset
                    )
                    if ret != op.size_bytes:
                        raise RuntimeError(
                            f"cuFileRead short read: {ret}/{op.size_bytes}"
                        )
            finally:
                cuFileHandleDeregister(handle)
        finally:
            os.close(fd)

    def shutdown_backend(self) -> None:
        if self._driver_open:
            self._driver_open = False
            try:
                cuFileDriverClose()
            except RuntimeError as e:
                logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")
