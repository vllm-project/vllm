# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric-memory buffers that share NVLink multicast objects.

Every `symm_mem.rendezvous` of torch's CUDA backend binds its allocation to a
new multicast object. The multicast objects of an NVLink domain (a GB300 NVL72
rack) are one pool shared by every process in it, whatever the buffer sizes, and
a start-up that finds the pool drained fails with `CUDA_ERROR_OUT_OF_MEMORY` from
`cuMulticastBindMem`. The specialized decode builds a mailbox per decoder layer,
so inside `multicast_arena` the buffers of one group are carved from shared
chunks instead: one rendezvous, one multicast object per chunk.
"""

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# The served TP8 decode's 66 mailboxes (62 layer boundaries of 768 KiB, the
# 6 MiB fused all-reduce, the pooler and the TP tail: about 56 MiB) take one
# chunk.
ARENA_CHUNK_BYTES = 128 << 20
# Buffer offsets within a chunk; keeps 16-byte multicast stores aligned and
# the buffers' polled lines apart.
ARENA_ALIGN = 256


class NoMulticastError(RuntimeError):
    """A symmetric buffer that needs a multicast mapping has none (on every
    rank of its group: `_rendezvous` agrees)."""


@dataclass(frozen=True)
class SymmetricBuffer:
    """This rank's `tensor`, the address of every rank's copy in rank order
    (`peers`) and the multicast address of the buffer (`multicast`, 0 without a
    multicast mapping)."""

    tensor: torch.Tensor
    peers: tuple[int, ...]
    multicast: int


def _group_name(group: dist.ProcessGroup | str) -> str:
    return group if isinstance(group, str) else group.group_name


def _rendezvous(
    tensor: torch.Tensor, group: dist.ProcessGroup | str
) -> tuple[tuple[int, ...], int]:
    """The peer addresses and multicast address of `tensor`, the multicast
    address 0 on every rank when the multicast binding failed on any of them
    (torch then skips it on that rank only), so that every rank takes the same
    path. Collective over `group`."""
    handle = symm_mem.rendezvous(tensor, group)
    peers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
    multicast = int(handle.multicast_ptr or 0)
    if isinstance(group, str):
        from torch.distributed.distributed_c10d import _resolve_process_group

        group = _resolve_process_group(group)
    bound = torch.tensor([int(multicast != 0)], dtype=torch.int32, device=tensor.device)
    dist.all_reduce(bound, op=dist.ReduceOp.MIN, group=group)
    if not bound.item():
        logger.warning_once(
            "No NVLink multicast object on some rank of the group (the rack's "
            "multicast pool is exhausted or the platform has none)."
        )
        return peers, 0
    return peers, multicast


class MulticastArena:
    """Carves one group's symmetric buffers out of chunks of `chunk_bytes`,
    each one symmetric allocation with one multicast object. Allocation is
    collective: every rank of the group allocates the same buffers in the same
    order, so a buffer has the same offset in every rank's chunk."""

    def __init__(
        self,
        group: dist.ProcessGroup | str,
        device: torch.device,
        chunk_bytes: int = ARENA_CHUNK_BYTES,
    ) -> None:
        self.group, self.device, self.chunk_bytes = group, device, chunk_bytes
        self.chunks: list[torch.Tensor] = []
        self._peers: tuple[int, ...] = ()
        self._multicast = 0
        self._used = 0

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype) -> SymmetricBuffer:
        nbytes = math.prod(shape) * dtype.itemsize
        if not self.chunks or self._used + nbytes > self.chunks[-1].numel():
            chunk = symm_mem.empty(
                max(nbytes, self.chunk_bytes), dtype=torch.uint8, device=self.device
            )
            self._peers, self._multicast = _rendezvous(chunk, self.group)
            self.chunks.append(chunk)
            self._used = 0
        offset = self._used
        self._used = round_up(offset + nbytes, ARENA_ALIGN)
        tensor = self.chunks[-1][offset : offset + nbytes].view(dtype).view(shape)
        return SymmetricBuffer(
            tensor,
            tuple(pointer + offset for pointer in self._peers),
            self._multicast + offset if self._multicast else 0,
        )


_ARENAS: dict[str, MulticastArena] = {}


@contextlib.contextmanager
def multicast_arena(
    group: dist.ProcessGroup | str,
    device: torch.device,
    chunk_bytes: int = ARENA_CHUNK_BYTES,
) -> Iterator[MulticastArena]:
    """Within the block, `symmetric_buffer` over `group` carves from one arena.
    Raises when an arena over `group` is already active: its chunks would
    ignore `chunk_bytes`."""
    name = _group_name(group)
    if name in _ARENAS:
        raise RuntimeError(f"a multicast arena over group {name} is already active")
    arena = _ARENAS[name] = MulticastArena(group, device, chunk_bytes)
    try:
        yield arena
    finally:
        del _ARENAS[name]


def symmetric_buffer(
    group: dist.ProcessGroup | str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> SymmetricBuffer:
    """A `shape` `dtype` symmetric buffer over `group`, from the group's
    `multicast_arena` when one is active, else its own allocation (and its own
    multicast object). Collective over `group`."""
    arena = _ARENAS.get(_group_name(group))
    if arena is not None:
        return arena.allocate(shape, dtype)
    tensor = symm_mem.empty(shape, dtype=dtype, device=device)
    peers, multicast = _rendezvous(tensor, group)
    return SymmetricBuffer(tensor, peers, multicast)
