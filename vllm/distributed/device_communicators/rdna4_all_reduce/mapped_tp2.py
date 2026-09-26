# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP2 runtime for RDNA4 all-reduce over mapped shared host memory."""

from __future__ import annotations

import ctypes
import mmap
import os
import tempfile
import uuid
from collections.abc import Callable, Collection
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64

from .flydsl_kernels.rdna4_all_reduce_tp2 import (
    make_mapped_tp2_full_launcher,
    make_mapped_tp2_pipeline_launcher,
)

_CONTROL_BYTES = 4096
_ALIGNMENT = 4096
_DEFAULT_MAX_NUMEL = 16_777_216
_FULL_1_MAX = 65_536
_FULL_2_MAX = 196_608
_FULL_4_MAX = 786_432
_PIPELINE_8_MAX = 2_097_152
_PIPELINE_10_MAX = 2_490_368
_PIPELINE_11_MAX = 3_801_088
_PIPELINE_12_MAX = 5_242_880

_READY_OFFSET = 0
_LAUNCH_OFFSET = 48
_BLOCK_READY_OFFSET = 96
_PIPELINE_LAUNCH_OFFSET = 288
_PIPELINE_PROGRESS_OFFSET = 464


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


class RDNA4TP2MappedAllReduce:
    """Graph-safe TP2 all-reduce over HIP-mapped shared host memory."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        *,
        max_numel: int = _DEFAULT_MAX_NUMEL,
        preferred_numel: Collection[int] | None = None,
        threads: int = 1024,
        blocks: int | None = None,
        pipeline_packs: int | None = None,
        register_host_mapping: Callable[[int, int], int],
        unregister_host_mapping: Callable[[int], None],
    ) -> None:
        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_numel = max_numel
        self.preferred_numel = (
            None if preferred_numel is None else frozenset(preferred_numel)
        )
        self.threads = threads
        self.blocks = blocks
        self.pipeline_packs = pipeline_packs
        self._register_host_mapping = register_host_mapping
        self._unregister_host_mapping = unregister_host_mapping
        self.disabled = True
        self._mapping: mmap.mmap | None = None
        self._host_address = 0
        self._device_address = 0
        self._path: str | None = None

        if self.world_size != 2 or self.device.type != "cuda":
            return
        if max_numel <= 0:
            raise ValueError("max_numel must be positive")
        if threads != 1024:
            raise ValueError("the validated FlyDSL launch uses 1024 threads")
        if blocks is not None and blocks not in (1, 2, 4, *range(6, 17)):
            raise ValueError("blocks must be 1, 2, 4, or in [6, 16]")
        if pipeline_packs is not None and (
            pipeline_packs < threads or pipeline_packs % threads
        ):
            raise ValueError("pipeline_packs must be a positive multiple of threads")
        if pipeline_packs is not None and blocks not in range(6, 17):
            raise ValueError("a fixed pipeline requires blocks in [6, 16]")
        if pipeline_packs is None and blocks is not None and blocks not in (1, 2, 4):
            raise ValueError("a fixed full-phase launch requires 1, 2, or 4 blocks")
        invalid_sizes = []
        if self.preferred_numel is not None:
            invalid_sizes = [
                size
                for size in self.preferred_numel
                if size <= 0 or size > max_numel or size % 8 != 0
            ]
        if invalid_sizes:
            raise ValueError(
                "preferred_numel entries must be positive multiples of eight "
                f"not exceeding max_numel; invalid={sorted(invalid_sizes)}"
            )

        self._mode_layout: dict[tuple[str, int], tuple[int, int]] = {}
        layout_offset = _CONTROL_BYTES

        def add_mode(kind: str, grid: int, capacity: int) -> None:
            nonlocal layout_offset
            if capacity <= 0:
                return
            slot_bytes = _align_up(capacity * 2)
            self._mode_layout[(kind, grid)] = (layout_offset, slot_bytes)
            layout_offset += 4 * slot_bytes

        if pipeline_packs is not None:
            assert blocks is not None
            add_mode("pipeline", blocks, max_numel)
        elif blocks is not None:
            add_mode("full", blocks, max_numel)
        else:
            add_mode("full", 1, min(max_numel, _FULL_1_MAX))
            add_mode("full", 2, min(max_numel, _FULL_2_MAX))
            add_mode("full", 4, min(max_numel, _FULL_4_MAX))
            if max_numel > _FULL_4_MAX:
                add_mode("pipeline", 8, min(max_numel, _PIPELINE_8_MAX))
            if max_numel > _PIPELINE_8_MAX:
                add_mode("pipeline", 10, min(max_numel, _PIPELINE_10_MAX))
            if max_numel > _PIPELINE_10_MAX:
                add_mode("pipeline", 11, min(max_numel, _PIPELINE_11_MAX))
            if max_numel > _PIPELINE_11_MAX:
                add_mode("pipeline", 12, min(max_numel, _PIPELINE_12_MAX))
            if max_numel > _PIPELINE_12_MAX:
                add_mode("pipeline", 16, max_numel)
        self._mapping_bytes = layout_offset
        self._initialize_shared_mapping()
        self.disabled = False

    def _initialize_shared_mapping(self) -> None:
        rank_paths: list[str | None] = [None]
        if self.rank == 0:
            shared_dir = Path("/dev/shm")
            if not shared_dir.is_dir():
                shared_dir = Path(tempfile.gettempdir())
            rank_paths[0] = str(shared_dir / f"rdna4_tp2_flydsl_{uuid.uuid4().hex}.shm")
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
                self._host_address, self._mapping_bytes
            )
        except Exception as error:  # noqa: BLE001
            local_error = f"rank {self.rank}: {error}"

        errors: list[str | None] = [None, None]
        dist.all_gather_object(errors, local_error, group=self.group)
        if self.rank == 0:
            os.unlink(path)
        failures = [error for error in errors if error]
        if failures:
            self.close()
            raise RuntimeError("; ".join(failures))

    def should_use(self, tensor: torch.Tensor) -> bool:
        numel = tensor.numel()
        return (
            not self.disabled
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and numel > 0
            and numel <= self.max_numel
            and numel % 8 == 0
            and (self.preferred_numel is None or numel in self.preferred_numel)
        )

    @staticmethod
    def _policy(numel: int, threads: int) -> tuple[str, int, int]:
        if numel <= _FULL_1_MAX:
            return "full", 1, 0
        if numel <= _FULL_2_MAX:
            return "full", 2, 0
        if numel <= _FULL_4_MAX:
            return "full", 4, 0
        if numel <= _PIPELINE_8_MAX:
            blocks = 8
            pack_count = numel // 8
            chunk_packs = _align_up(
                (pack_count + blocks - 1) // blocks,
                threads,
            )
            return "pipeline", blocks, chunk_packs
        if numel <= _PIPELINE_10_MAX:
            return "pipeline", 10, 6144
        if numel <= _PIPELINE_11_MAX:
            return "pipeline", 11, 6144
        if numel <= _PIPELINE_12_MAX:
            return "pipeline", 12, 6144
        return "pipeline", 16, 8192

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        stream_ptr: int | None = None,
    ) -> torch.Tensor | None:
        if not self.should_use(tensor):
            return None
        numel = tensor.numel()
        if self.pipeline_packs is not None:
            assert self.blocks is not None
            kind, blocks, chunk_packs = (
                "pipeline",
                self.blocks,
                self.pipeline_packs,
            )
        elif self.blocks is not None:
            kind, blocks, chunk_packs = "full", self.blocks, 0
        else:
            kind, blocks, chunk_packs = self._policy(numel, self.threads)
        data_offset, slot_bytes = self._mode_layout[(kind, blocks)]
        local_slot = self._device_address + data_offset + self.rank * 2 * slot_bytes
        peer_slot = (
            self._device_address + data_offset + (1 - self.rank) * 2 * slot_bytes
        )
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

        if kind == "full":
            mode = {1: 0, 2: 1, 4: 2}[blocks]
            if blocks == 1:
                local_ready = self._control_addr(
                    _READY_OFFSET + (mode * 2 + self.rank) * 8
                )
                peer_ready = self._control_addr(
                    _READY_OFFSET + (mode * 2 + 1 - self.rank) * 8
                )
            else:
                local_ready = self._control_addr(
                    _BLOCK_READY_OFFSET + (mode * 2 + self.rank) * 4 * 8
                )
                peer_ready = self._control_addr(
                    _BLOCK_READY_OFFSET + (mode * 2 + 1 - self.rank) * 4 * 8
                )
            local_launch = self._control_addr(
                _LAUNCH_OFFSET + (mode * 2 + self.rank) * 8
            )
            launcher = make_mapped_tp2_full_launcher(
                blocks=blocks, threads=self.threads
            )
            launcher(
                Int64(int(tensor.data_ptr())),
                Int64(int(output.data_ptr())),
                Int64(local_slot),
                Int64(peer_slot),
                Int64(slot_bytes),
                Int64(local_ready),
                Int64(peer_ready),
                Int64(local_launch),
                Int32(numel),
                stream=stream,
            )
        else:
            mode = blocks - 6
            local_progress = self._control_addr(
                _PIPELINE_PROGRESS_OFFSET + (mode * 2 + self.rank) * 16 * 8
            )
            peer_progress = self._control_addr(
                _PIPELINE_PROGRESS_OFFSET + (mode * 2 + 1 - self.rank) * 16 * 8
            )
            local_launch = self._control_addr(
                _PIPELINE_LAUNCH_OFFSET + (mode * 2 + self.rank) * 8
            )
            launcher = make_mapped_tp2_pipeline_launcher(
                blocks=blocks,
                threads=self.threads,
                chunk_packs=chunk_packs,
            )
            launcher(
                Int64(int(tensor.data_ptr())),
                Int64(int(output.data_ptr())),
                Int64(local_slot),
                Int64(peer_slot),
                Int64(slot_bytes),
                Int64(local_progress),
                Int64(peer_progress),
                Int64(local_launch),
                Int32(numel),
                stream=stream,
            )
        return output

    def _control_addr(self, offset: int) -> int:
        return self._device_address + offset

    def close(self) -> None:
        if self._host_address:
            self._unregister_host_mapping(self._host_address)
            self._host_address = 0
            self._device_address = 0
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = ["RDNA4TP2MappedAllReduce"]
