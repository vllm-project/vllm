# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mapped-host transport for low-latency TP4/TP8 RDNA4 all-reduce."""

import ctypes
import mmap
import os
import tempfile
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64

from .flydsl_kernels.rdna4_all_reduce_mapped import (
    make_mapped_full_launcher,
    make_mapped_rsag_launcher,
)

_CONTROL_BYTES = 4096
_ALIGNMENT = 4096
_PIPELINE_LAUNCH_OFFSET = 128
_PIPELINE_PROGRESS_OFFSET = 256
_PIPELINE_BLOCKS = 16


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


class RDNA4MappedAllReduce:
    """BF16 all-reduce over a shared HIP-mapped host allocation."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        *,
        max_size: int,
        register_host_mapping: Callable[[int, int], int],
        unregister_host_mapping: Callable[[int], None],
        threads: int = 1024,
        pipeline_blocks: int = _PIPELINE_BLOCKS,
    ) -> None:
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_size = max_size
        self.threads = threads
        self.pipeline_blocks = pipeline_blocks
        self.pipeline_min_numel = 131_072 if self.world_size == 4 else 65_536
        self._register_host_mapping = register_host_mapping
        self._unregister_host_mapping = unregister_host_mapping
        self.disabled = True
        self._mapping: mmap.mmap | None = None
        self._host_address = 0
        self._device_address = 0
        self._path: str | None = None

        if self.world_size not in (4, 8) or self.device.type != "cuda":
            return
        if max_size <= 0 or max_size % 16:
            raise ValueError("max_size must be a positive multiple of 16 bytes")
        if threads not in (256, 512, 1024):
            raise ValueError("threads must be 256, 512, or 1024")
        if (
            pipeline_blocks < self.world_size
            or pipeline_blocks > 16
            or pipeline_blocks % self.world_size
        ):
            raise ValueError("pipeline_blocks must be a TP multiple in [TP, 16]")

        self._slot_bytes = _align_up(max_size)
        self._mapping_bytes = _CONTROL_BYTES + self.world_size * 2 * self._slot_bytes
        self._initialize_shared_mapping()
        self._full_launcher = make_mapped_full_launcher(
            world_size=self.world_size,
            threads=threads,
        )
        self.disabled = False

    def _initialize_shared_mapping(self) -> None:
        rank_paths: list[str | None] = [None]
        if self.rank == 0:
            shared_dir = Path("/dev/shm")
            if not shared_dir.is_dir():
                shared_dir = Path(tempfile.gettempdir())
            name = f"vllm_rdna4_mapped_ar_{uuid.uuid4().hex}.shm"
            rank_paths[0] = str(shared_dir / name)
        group_ranks = dist.get_process_group_ranks(self.group)
        dist.broadcast_object_list(rank_paths, src=group_ranks[0], group=self.group)
        path = rank_paths[0]
        assert path is not None
        self._path = path

        if self.rank == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, self._mapping_bytes)
            finally:
                os.close(fd)
        dist.barrier(group=self.group)

        local_error = ""
        try:
            fd = os.open(path, os.O_RDWR)
            try:
                self._mapping = mmap.mmap(
                    fd,
                    self._mapping_bytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
            self._host_address = ctypes.addressof(
                ctypes.c_char.from_buffer(self._mapping)
            )
            self._device_address = self._register_host_mapping(
                self._host_address,
                self._mapping_bytes,
            )
        except Exception as error:  # noqa: BLE001
            local_error = f"rank {self.rank}: {error}"

        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, local_error, group=self.group)
        if self.rank == 0:
            os.unlink(path)
        failures = [error for error in errors if error]
        if failures:
            self.close()
            raise RuntimeError("; ".join(failures))

    def should_use(self, tensor: torch.Tensor) -> bool:
        return (
            not self.disabled
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and 0 < tensor.nbytes <= self.max_size
            and tensor.nbytes % 16 == 0
            and tensor.data_ptr() % 16 == 0
        )

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        stream_ptr: int | None = None,
    ) -> torch.Tensor | None:
        if not self.should_use(tensor):
            return None
        output = torch.empty_like(tensor) if out is None else out
        if (
            output.shape != tensor.shape
            or output.dtype != tensor.dtype
            or output.device != tensor.device
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError(
                "output must match the input and be contiguous and aligned"
            )
        stream = (
            torch.cuda.current_stream(self.device)
            if stream_ptr is None
            else torch.cuda.ExternalStream(stream_ptr)
        )
        numel = tensor.numel()
        common_args = (
            Int64(int(tensor.data_ptr())),
            Int64(int(output.data_ptr())),
            Int64(self._device_address),
            Int64(_CONTROL_BYTES),
            Int64(self._slot_bytes),
        )
        if numel < self.pipeline_min_numel:
            self._full_launcher(
                *common_args,
                Int32(self.rank),
                Int32(numel),
                stream=stream,
            )
        else:
            pack_count = numel // 8
            chunk_packs = _align_up(
                (pack_count + self.pipeline_blocks - 1) // self.pipeline_blocks,
                self.threads,
            )
            launcher = make_mapped_rsag_launcher(
                world_size=self.world_size,
                blocks=self.pipeline_blocks,
                threads=self.threads,
                chunk_packs=chunk_packs,
            )
            launcher(
                *common_args,
                Int64(_PIPELINE_LAUNCH_OFFSET),
                Int64(_PIPELINE_PROGRESS_OFFSET),
                Int32(self.rank),
                Int32(numel),
                stream=stream,
            )
        return output

    def close(self) -> None:
        if self._host_address:
            self._unregister_host_mapping(self._host_address)
            self._host_address = 0
            self._device_address = 0
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self.disabled = True

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = ["RDNA4MappedAllReduce"]
