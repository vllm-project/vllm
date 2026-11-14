# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading worker — direct GPU<->NVMe via cuFile sync API."""

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.gds.cufile_bindings import (
    cuFileDriverClose,
    cuFileDriverOpen,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    cuFileRead,
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
    scheduler thread free of blocking operations.  Only cuFileDriverOpen
    and cuFileDriverClose run on the caller's thread at init/shutdown.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        block_size_factor: int,
        device: torch.device,
        extra_config: dict[str, Any],
    ):
        if not current_platform.is_cuda():
            raise RuntimeError("GDS offloading requires an NVIDIA CUDA GPU.")
        super().__init__(
            file_mapper=file_mapper,
            block_size_factor=block_size_factor,
            device=device,
            max_io_threads=int(extra_config.get("gds_max_io_threads", 32)),
        )

        cuFileDriverOpen()
        logger.info("cuFile driver opened")

    def write_block(self, file_path: str, ops: list[FileOp]) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fd = open_for_gds(file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
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
        try:
            cuFileDriverClose()
        except RuntimeError as e:
            logger.warning("cuFileDriverClose failed: %s", e)
        logger.info("GDS worker shut down")
