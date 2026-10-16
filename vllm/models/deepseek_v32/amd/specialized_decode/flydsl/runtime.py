# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as
# vendored by ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/runtime.py).
# The original source code was licensed under the Apache License 2.0 and included the
# following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths
#   rewritten to this package; torch.accelerator in place of torch.cuda device /
#   synchronize calls; close() and the context manager (unused: the buffers live for the
#   process) removed.

"""Owned HIP IPC peer memory used by fused multi-GPU kernels."""

from __future__ import annotations

import contextlib

import torch

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.ipc import (
    close_ipc_handle,
    get_allocation_base,
    get_ipc_handle,
    open_ipc_handle,
)


class SymmetricPeerBuffer:
    """Allocate one symmetric buffer and exchange its address with every rank.

    Remote HIP IPC mappings stay open for the life of the process. The local
    allocation is owned by ``storage`` and stays alive for the wrapper's lifetime.
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

        if npes == 1:
            device = torch.device("cuda", torch.accelerator.current_device_index())
            self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            self.local_address = self.storage.data_ptr()
            addresses = [self.local_address]
        else:
            import torch.distributed as dist

            local_error = None
            mine = None
            try:
                device = torch.device("cuda", torch.accelerator.current_device_index())
                self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
                self.local_address = self.storage.data_ptr()
                base = get_allocation_base(self.local_address)
                mine = (get_ipc_handle(base), self.local_address - base)
            except Exception as error:
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
                raise RuntimeError(f"symmetric peer allocation/export failed: {detail}")

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
                except Exception as error:
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
                    with contextlib.suppress(Exception):
                        close_ipc_handle(remote_base)
                self._remote_bases.clear()
                detail = "; ".join(f"rank {peer}: {error}" for peer, error in failed)
                raise RuntimeError(f"symmetric peer open failed: {detail}")

            dist.barrier(group=group)
        assert device is not None
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)
