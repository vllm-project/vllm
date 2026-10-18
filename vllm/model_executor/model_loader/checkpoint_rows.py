# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table rows served on the host from the checkpoint files (`disk_offload`)."""

import mmap
import os

import torch


def _maps_entry(ptr: int) -> tuple[str, int]:
    """Return the path and file offset of `ptr` from `/proc/self/maps`."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            low, high = (int(bound, 16) for bound in fields[0].split("-"))
            if low <= ptr < high:
                path = fields[5].rstrip("\n") if len(fields) > 5 else ""
                return path, ptr - low + int(fields[2], 16)
    return "", 0


class CheckpointRows:
    """One table plane read on the host from the checkpoint's safetensors files.

    The default loader hands over zero-copy mmap views. `add` keeps each
    checkpoint shard's view until `bind` records its file and offset and
    releases it, so the table is never mapped or allocated after loading.
    Rows stay in the checkpoint format; recently read ones are served from
    the page cache, which the kernel can reclaim.
    """

    def __init__(self, name: str, num_rows: int, dtype: torch.dtype) -> None:
        self.name = name
        self.num_rows = num_rows
        self.dtype = dtype
        self.row_bytes = 0
        self._shards: dict[int, torch.Tensor] = {}
        self._shard_rows = 0
        self._fds: dict[str, int] = {}
        self._sources: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def bound(self) -> bool:
        return self._sources is not None

    def add(self, index: int, view: torch.Tensor, shard_rows: int) -> None:
        """Keep checkpoint shard `index`, which holds `shard_rows` table rows."""
        if self.bound:
            raise RuntimeError(
                f"{self.name} disk offload does not support weight reload"
            )
        if view.dtype != self.dtype:
            raise ValueError(
                f"{self.name} shard dtype {view.dtype} must match the embedding "
                f"dtype {self.dtype}; disk offload serves rows as stored"
            )
        if index in self._shards:
            raise ValueError(f"Duplicate {self.name} shard {index}")
        self._shards[index] = view.view(torch.uint8)
        self._shard_rows = shard_rows
        self.row_bytes = self._shards[index].shape[1]

    def bind(self) -> None:
        """Record each shard's file and offset and release the loader's views."""
        if self.bound:
            return
        rows = sum(shard.shape[0] for shard in self._shards.values())
        if rows != self.num_rows:
            raise ValueError(f"{self.name} shards cover {rows} of {self.num_rows} rows")
        fds, bases = torch.zeros(2, len(self._shards)).long()
        for index, shard in self._shards.items():
            # The loader hands over mmap views, so find their files here.
            path, offset = _maps_entry(shard.data_ptr())
            if not path or not os.path.isfile(path):
                raise ValueError(
                    f"{self.name} shard {index} is not a file-backed view "
                    f"({path or 'anonymous'})"
                )
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            fds[index], bases[index] = self._fds[path], offset
        self._sources = (fds, bases)
        self._shards.clear()

    def read(self, ids: torch.Tensor, rows: torch.Tensor) -> None:
        """Read the rows of sorted distinct `ids` into the uint8 buffer `rows`."""
        assert self._sources is not None, f"{self.name} rows are not bound"
        assert rows.shape == (ids.numel(), self.row_bytes)
        if ids.numel() and not 0 <= ids[0] <= ids[-1] < self.num_rows:
            raise IndexError(f"{self.name} id out of range for {self.num_rows} rows")
        fds, bases = self._sources
        page, row_bytes = mmap.PAGESIZE, self.row_bytes
        shard = ids // self._shard_rows
        start = bases[shard] + (ids - shard * self._shard_rows) * row_bytes
        first, last = start // page, (start + row_bytes - 1) // page
        run = torch.ones_like(first, dtype=torch.bool)
        run[1:] = (first[1:] > last[:-1] + 1) | (shard[1:] != shard[:-1])
        # Queue every page run before the serial reads.
        for fd, lo, hi in zip(
            fds[shard[run]].tolist(),
            first[run].tolist(),
            last[run.roll(-1)].tolist(),
        ):
            os.posix_fadvise(
                fd, lo * page, (hi - lo + 1) * page, os.POSIX_FADV_WILLNEED
            )
        for row, fd, offset in zip(rows.numpy(), fds[shard].tolist(), start.tolist()):
            if os.preadv(fd, [row], offset) != row_bytes:
                raise ValueError(
                    f"{self.name} shard file "
                    f"{os.readlink(f'/proc/self/fd/{fd}')} is truncated at "
                    f"offset {offset}"
                )
