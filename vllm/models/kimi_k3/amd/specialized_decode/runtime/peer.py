# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/peer_memory.py
# atom/mono/runtime/consensus.py
"""Peer memory for in-kernel TP collectives: one uncached buffer a rank, every
rank holding every peer's address, and the TP agreements that must precede
anything collective."""

import torch
import torch.distributed as dist


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address, for a
    non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address
    (``addresses``, int64, indexed by TP rank). What lives where, and why no rank
    overwrites a peer's unread data, is the model's layout's to say.

    ``bytes`` is the whole buffer: the kernels tag every pair with the step's
    epoch, so it is never zeroed between steps.

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so a
    runner reaches it only after ``tp_agree``. A failed mapping on any rank raises
    ``RuntimeError`` on all of them.
    """

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

        self._heap = UncachedIpcHeap
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise RuntimeError(
                    "peer memory handshake failed on "
                    f"{'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent. Only once no
        kernel of any rank can still touch it (the runner is being dropped)."""
        opened, self._opened = self._opened, []
        for base in opened:
            self._heap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            self._heap.free_device_mem(self.local)
            self.local = 0


# Decisions every TP rank must take alike.
#
# A mono kernel waits in-kernel on every peer rank, so a rank that turned mono off
# alone -- or failed to build what the others launch -- leaves the others spinning
# on the GPU for ever. Each such decision is a local verdict agreed on over the TP
# group's CPU group before anything collective (peer handshake, launch) follows.
#


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group, or None for a
    single rank."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())
