# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host CUDA VMM peer aliases for an ExtensibleTensor-owned allocation."""

from __future__ import annotations

import array
import os
import socket
import struct
import tempfile
import weakref
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.utils.vmm_driver import CudaVmmDriver

if TYPE_CHECKING:
    from vllm.utils.extensible_tensor import ExtensibleTensor

_BATCH_FDS = 32


class VmmPeerMappings:
    """Reserve peer VAs once and map newly committed owner chunks into them."""

    def __init__(self, owner: ExtensibleTensor, group: dist.ProcessGroup):
        self._owner = weakref.ref(owner)
        self.group = group
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        driver = owner._buffer._driver
        self.device = owner.device
        self.capacity = owner.capacity_bytes
        self.bases = [0] * self.world_size
        self.ranges: list[list[tuple[int, int]]] = [[] for _ in self.bases]
        self._closed = False
        self._failed = False
        self._num_chunks = 0
        self._identical(
            (
                socket.gethostname(),
                owner._max_num_bytes,
                owner.num_segments,
                owner.segment_capacity_bytes,
                owner.granularity,
                owner.bytes_per_segment,
            )
        )
        self._check(
            None
            if isinstance(driver, CudaVmmDriver) and owner._buffer.exportable
            else RuntimeError("Peer storage must be exportable CUDA VMM")
        )
        assert isinstance(driver, CudaVmmDriver)
        self.driver = driver
        error = None
        try:
            for rank in range(self.world_size):
                self.bases[rank] = (
                    owner.base_ptr
                    if rank == self.rank
                    else self.driver.reserve(self.capacity)
                )
            self.pointers = torch.tensor(
                self.bases, dtype=torch.uint64, device=self.device
            )
        except RuntimeError as exc:
            error = exc
        try:
            self._check(error)
            self.refresh()
        except BaseException:
            self._cleanup()
            raise

    def _check(self, error: Exception | None) -> None:
        errors = [None] * self.world_size
        dist.all_gather_object(errors, str(error) if error else None, group=self.group)
        if any(errors):
            self._failed = True
            raise RuntimeError(f"VMM peer operation failed (no fallback): {errors}")

    def _identical(self, value: Any) -> None:
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        if any(v != values[0] for v in values):
            self._failed = True
            raise RuntimeError(
                "PCP requires identical same-host VMM layouts and growth"
            )

    def check_growth(self, size: int) -> None:
        owner = self._owner()
        assert owner is not None
        self._identical((size, owner.bytes_per_segment))
        self._check(
            RuntimeError("VMM peer storage is closed or failed")
            if self._closed or self._failed
            else None
        )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Commit VMM pages outside CUDA graph capture")

    def refresh(self) -> None:
        owner = self._owner()
        assert owner is not None
        chunks = tuple((offset, size) for _, offset, size in owner._buffer._handles)
        self._identical((chunks, owner.bytes_per_segment, self._num_chunks))
        new_chunks = owner._buffer._handles[self._num_chunks :]
        # Owner zeroing must complete before any producer can write through peers.
        torch.accelerator.synchronize(self.device)
        local_fds = []
        error = None
        try:
            for handle, _, _ in new_chunks:
                local_fds.append(self.driver.export_fd(handle))
        except RuntimeError as exc:
            error = exc
        try:
            self._check(error)
            if not new_chunks:
                return
            with ExitStack() as stack:
                error = None
                try:
                    directory = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="vllm_vmm_peer_")
                    )
                    channel = stack.enter_context(
                        socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                    )
                    channel.settimeout(30)
                    path = os.path.join(directory, "fd.sock")
                    channel.bind(path)
                except OSError as exc:
                    error = exc
                self._check(error)
                paths = [None] * self.world_size
                dist.all_gather_object(paths, path, group=self.group)
                for source in range(self.world_size):
                    error = None
                    if self.rank == source:
                        try:
                            for target in range(self.world_size):
                                if target == source:
                                    continue
                                for start in range(0, len(local_fds), _BATCH_FDS):
                                    fds = array.array(
                                        "i", local_fds[start : start + _BATCH_FDS]
                                    )
                                    channel.sendmsg(
                                        [struct.pack("!II", source, start)],
                                        [
                                            (
                                                socket.SOL_SOCKET,
                                                socket.SCM_RIGHTS,
                                                fds,
                                            )
                                        ],
                                        0,
                                        paths[target],
                                    )
                        except OSError as exc:
                            error = exc
                    else:
                        for start in range(0, len(new_chunks), _BATCH_FDS):
                            received = array.array("i")
                            try:
                                payload, ancillary, flags, _ = channel.recvmsg(
                                    8,
                                    socket.CMSG_SPACE(_BATCH_FDS * received.itemsize),
                                    getattr(socket, "MSG_CMSG_CLOEXEC", 0),
                                )
                                for level, kind, data in ancillary:
                                    if (
                                        level == socket.SOL_SOCKET
                                        and kind == socket.SCM_RIGHTS
                                    ):
                                        received.frombytes(
                                            data[
                                                : len(data)
                                                // received.itemsize
                                                * received.itemsize
                                            ]
                                        )
                                expected = min(_BATCH_FDS, len(new_chunks) - start)
                                if (
                                    payload != struct.pack("!II", source, start)
                                    or flags & socket.MSG_CTRUNC
                                    or len(received) != expected
                                ):
                                    raise RuntimeError("Malformed VMM FD exchange")
                                for i, fd in enumerate(received):
                                    if error is not None:
                                        continue
                                    _, offset, size = new_chunks[start + i]
                                    handle = self.driver.import_fd(fd)
                                    try:
                                        self.driver.map(
                                            self.bases[source] + offset,
                                            size,
                                            handle,
                                        )
                                        self.ranges[source].append((offset, size))
                                        self.driver.set_access(
                                            self.bases[source] + offset,
                                            size,
                                            self.device.index,
                                        )
                                    finally:
                                        self.driver.release(handle)
                            except (OSError, RuntimeError) as exc:
                                error = error or exc
                            finally:
                                for fd in received:
                                    os.close(fd)
                    # All receives finish before the next sender starts.
                    self._check(error)
            self._num_chunks = len(chunks)
        finally:
            for fd in local_fds:
                os.close(fd)

    def release_mappings(self) -> None:
        if self._closed:
            return
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group)
        error = None
        for rank, ranges in enumerate(self.ranges):
            remaining = []
            for offset, size in ranges:
                try:
                    self.driver.unmap(self.bases[rank] + offset, size)
                except RuntimeError as exc:
                    error = error or exc
                    remaining.append((offset, size))
            ranges[:] = remaining
        self._check(error)
        self._num_chunks = 0

    def _cleanup(self) -> None:
        for rank, base in enumerate(self.bases):
            if rank == self.rank or not base:
                continue
            for offset, size in self.ranges[rank]:
                with suppress(Exception):
                    self.driver.unmap(base + offset, size)
            self.ranges[rank].clear()
            with suppress(Exception):
                self.driver.free_reserved(base, self.capacity)
            self.bases[rank] = 0
        self._closed = True

    def close(self) -> None:
        if not self._closed:
            self.release_mappings()
            self._cleanup()

    def __del__(self):
        with suppress(Exception):
            self._cleanup()
