# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device buffers that every rank of a group maps through IPC handles."""

import ctypes

import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)


class PeerBuffer:
    """One zeroed ``cudaMalloc`` buffer a rank, mapped by every rank of a group.

    The monokernel polls these mailboxes with system-scope loads, so they come
    from a plain device allocation: not torch's caching allocator (pool
    offsets) and not the uncached custom all-reduce buffers. Freed by
    ``close()`` or when the object is collected (like ``CustomAllreduce``).
    """

    def __init__(self, nbytes: int, group: ProcessGroup) -> None:
        self.nbytes = nbytes
        self.pointers: list[int] = []
        self._opened: list[ctypes.c_void_p] = []
        self.local: ctypes.c_void_p | None = None
        self.lib = CudaRTLibrary()
        self.rank = dist.get_rank(group=group)
        self.local = self.lib.cudaMalloc(nbytes)
        self.lib.cudaMemset(self.local, 0, nbytes)
        self.lib.cudaDeviceSynchronize()
        handle = self.lib.cudaIpcGetMemHandle(self.local)

        handles: list[bytes | None] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(handles, bytes(handle), group=group)

        for rank, raw in enumerate(handles):
            if rank == self.rank:
                pointer = self.local
            else:
                assert raw is not None
                pointer = self.lib.cudaIpcOpenMemHandle(
                    cudaIpcMemHandle_t.from_buffer_copy(raw)
                )
                self._opened.append(pointer)
            assert pointer.value is not None
            self.pointers.append(pointer.value)

    def close(self) -> None:
        """Unmap the peers' buffers and free this rank's; idempotent."""
        while self._opened:
            self.lib.cudaIpcCloseMemHandle(self._opened.pop())
        if self.local is not None:
            self.lib.cudaFree(self.local)
            self.local = None
        self.pointers = []

    def __del__(self) -> None:
        self.close()
