# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric peer memory for the mono kernels' in-kernel all-reduces."""

import torch
import torch.distributed as dist
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address.

    K4 pushes a layer's attention partials into every peer's ``attn`` region and
    its FFN partials into the ``ffn`` region, then polls its own. Regions are
    reused by every layer and step (the mailbox epoch tells a stale pair from a
    fresh one) and no rank can overwrite a peer's unread pairs: to push layer L's
    FFN partials a rank must have passed layer L's attention reduce, which needs
    every peer's layer-L attention push, and a peer only pushes that after it
    finished reading layer L-1's FFN region. So ranks are at most half a layer
    apart and the two regions double-buffer each other.
    """

    def __init__(self, nbytes: int, group, rank: int, npes: int, device: torch.device):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses = []
            for peer, handle in enumerate(handles):
                if peer == rank:
                    addresses.append(self.local)
                else:
                    base = UncachedIpcHeap.open_mem_handle(handle)
                    self._opened.append(base)
                    addresses.append(base)
            dist.barrier(group=group)
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent."""
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0
