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

from vllm.models.common.mono.spec import MonoUnsupported


def tp_group() -> tuple[int, int, object | None]:
    """This rank's (rank, world size, CPU group) in the TP group, and
    (0, 1, None) before distributed is initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1, None
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    return tp.rank_in_group, tp.world_size, tp.cpu_group


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address,
    for a non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer per rank, every rank holding every peer's address
    (``addresses``, int64, indexed by TP rank). What lives where, and why no
    rank overwrites a peer's unread data, is the model's layout's to say.

    ``bytes`` is the whole buffer: the kernels tag every pair with the step's
    epoch, so it is never zeroed between steps.

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so
    a runner reaches it only after ``tp_agree``. A failed mapping on any rank
    raises ``MonoUnsupported`` on all of them.
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
            except RuntimeError as err:
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported(
                    "peer memory handshake failed on "
                    f"{'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Closes the peer mappings and frees the local buffer; idempotent.
        Only once no kernel of any rank can still touch it (the runner is being
        dropped)."""
        opened, self._opened = self._opened, []
        for base in opened:
            self._heap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            self._heap.free_device_mem(self.local)
            self.local = 0


def tp_agree(ok: bool, group) -> bool:
    """Agrees a verdict every TP rank must take alike.

    A MonoKernel waits in-kernel on every peer rank, so a rank that turned mono
    off alone, or failed to build what the others launch, leaves the others
    spinning on the GPU for ever. Every rank must call this at the same point.

    Args:
        ok: This rank's local verdict.
        group: The TP group's CPU (gloo) process group, or None for one rank.

    Returns:
        True when every rank of ``group`` passed ``ok`` true.

    """
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def bind_agreed(bind, group) -> None:
    """``bind()`` on this rank (raising ``MonoUnsupported`` to refuse), then
    every rank's verdict agreed: a rank that raised re-raises, the others raise
    too."""
    try:
        bind()
    except Exception:
        tp_agree(False, group)
        raise
    if not tp_agree(True, group):
        raise MonoUnsupported("another TP rank refused mono")
