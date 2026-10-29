# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric peer memory for in-kernel TP reductions."""

from __future__ import annotations

import contextlib
from typing import Self

import torch

from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (
    MonoUnsupported,
    tp_agree,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.hip_ipc import (
    close_ipc_handle,
    get_allocation_base,
    get_ipc_handle,
    open_ipc_handle,
)


def _ipc_heap():
    """Aiter's uncached IPC heap, imported on use.

    `PeerBuffer` needs it; `SymmetricPeerBuffer` below drives HIP directly and
    needs nothing from aiter. Importing it at module scope would make the whole
    module -- and so every test of either class -- require an AITER build,
    which the CPU test gate does not have.
    """
    from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap

    return UncachedIpcHeap


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

    ``bytes`` is the data region, the part zeroed between steps; the step
    fence's slots sit past it (``step_begin``).

    The constructor is collective over ``group`` (the handle exchange, then an
    agreement that every rank mapped every peer): every rank must reach it, so a
    runner reaches it only after ``tp_agree``. A failed mapping on any rank raises
    ``MonoUnsupported`` on all of them.
    """

    def __init__(
        self,
        nbytes: int,
        group,
        rank: int,
        npes: int,
        device: torch.device,
    ):
        # imported here, not at module scope: `step_begin` compiles FlyDSL on
        # the way in, and `SymmetricPeerBuffer` below needs neither.
        from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.step_begin import (
            FENCE_BYTES,
        )

        self.local = _ipc_heap().alloc_uncached(nbytes + FENCE_BYTES)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles = _ipc_heap().gather_object_list_via_broadcast(
                group, _ipc_heap().get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = _ipc_heap().open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise MonoUnsupported(
                    f"peer memory handshake failed on {'this' if failure else 'another'}"
                    " TP rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def kernel_args(self) -> dict:
        """A kernel's view of the group, by ABI name: this rank's buffer
        (``sym``), the table of every rank's (``peers``) and the TP ``rank``."""
        return {
            "sym": self.local,
            "peers": self.addresses.data_ptr(),
            "rank": self.rank,
        }

    def close(self) -> None:
        """Close the peer mappings and free the local buffer; idempotent. Only once no
        kernel of any rank can still touch it (the runner is being dropped)."""
        opened, self._opened = self._opened, []
        for base in opened:
            _ipc_heap().close_mem_handle(base)
        if self.local:
            self.bytes = None
            _ipc_heap().free_device_mem(self.local)
            self.local = 0


# ---------------------------------------------------------------------------
# A SECOND implementation of the same idea, used by GLM-5.2's kernel.
#
# `PeerBuffer` above gets its memory from aiter (`UncachedIpcHeap`);
# `SymmetricPeerBuffer` below drives HIP directly through `hip_ipc`. They are
# not two copies of one routine -- they differ in how they allocate, exchange
# and free IPC handles -- so merging them is a behaviour change on a GPU, not a
# move, and a wrong merge is silent wrong numerics rather than a failing
# import. They sit together so the duplication is visible and the next person
# to touch peer memory has to decide, instead of finding the second one later.
# ---------------------------------------------------------------------------


class SymmetricPeerBuffer:
    """Allocate one symmetric buffer and exchange its address with every rank.

    Remote HIP IPC mappings are closed by :meth:`close`. The local allocation is
    owned by ``storage`` and stays alive for the wrapper's lifetime.
    """

    def __init__(self, nbytes: int, rank: int = 0, npes: int = 1, group=None):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be positive, got {nbytes}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        device = None
        self.rank = rank
        self.npes = npes
        self.group = group
        self._remote_bases: list[int] = []
        self._safety_barrier_complete = False

        if npes == 1:
            device = torch.device("cuda", torch.cuda.current_device())
            self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            self.local_address = self.storage.data_ptr()
            addresses = [self.local_address]
        else:
            import torch.distributed as dist

            local_error = None
            mine = None
            try:
                device = torch.device("cuda", torch.cuda.current_device())
                self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
                self.local_address = self.storage.data_ptr()
                base = get_allocation_base(self.local_address)
                mine = (get_ipc_handle(base), self.local_address - base)
            except Exception as error:  # noqa: BLE001 - reported to every rank below
                local_error = f"{type(error).__name__}: {error}"

            readiness = [None] * npes
            dist.all_gather_object(readiness, (local_error, mine), group=group)
            failed = [
                (peer, status[0])
                for peer, status in enumerate(readiness)
                if status[0] is not None
            ]
            if failed:
                detail = "; ".join(f"rank {peer}: {error}" for peer, error in failed)
                # MonoUnsupported, not RuntimeError: the GLM runner's _build
                # and LazyRunner both route on MonoUnsupported, so an OOM or a
                # failed IPC export here must disable mono, not take the server
                # down with it. The old PeerBuffer raised MonoUnsupported for
                # the same reason.
                raise MonoUnsupported(
                    f"symmetric peer allocation/export failed: {detail}"
                )

            addresses = []
            open_error = None
            for peer_rank, (_, peer) in enumerate(readiness):
                handle, offset = peer
                if peer_rank == rank:
                    addresses.append(self.local_address)
                    continue
                try:
                    remote_base = open_ipc_handle(handle)
                    self._remote_bases.append(remote_base)
                    addresses.append(remote_base + offset)
                # reported to every rank below
                except Exception as error:  # noqa: BLE001
                    open_error = f"{type(error).__name__}: {error}"
                    break

            open_status = [None] * npes
            dist.all_gather_object(open_status, open_error, group=group)
            failed = [
                (peer, error)
                for peer, error in enumerate(open_status)
                if error is not None
            ]
            if failed:
                for remote_base in self._remote_bases:
                    with contextlib.suppress(Exception):  # best-effort unwind
                        close_ipc_handle(remote_base)
                self._remote_bases.clear()
                detail = "; ".join(f"rank {peer}: {error}" for peer, error in failed)
                raise MonoUnsupported(f"symmetric peer open failed: {detail}")

            dist.barrier(group=group)
        assert device is not None
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Synchronize every rank, then close remote mappings.

        Every rank in the peer group must call this method in the same order.
        Repeated calls are safe. Failed closes remain tracked for a retry.
        """
        if not self._safety_barrier_complete:
            torch.cuda.synchronize(self.storage.device)
            if self.npes > 1:
                import torch.distributed as dist

                if not dist.is_initialized():
                    raise RuntimeError(
                        "the distributed process group must remain initialized until peer buffers close"
                    )
                dist.barrier(group=self.group)
            self._safety_barrier_complete = True
        failed = []
        first_error = None
        for base in self._remote_bases:
            try:
                close_ipc_handle(base)
            # every handle is freed, then the first error raised
            except Exception as exc:  # noqa: BLE001
                failed.append(base)
                if first_error is None:
                    first_error = exc
        self._remote_bases = failed
        if first_error is not None:
            raise first_error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
