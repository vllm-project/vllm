# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device buffers that every rank of a group maps through IPC handles."""

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
    offsets) and not the uncached custom all-reduce buffers.
    """

    def __init__(self, nbytes: int, group: ProcessGroup) -> None:
        self.nbytes = nbytes
        self.lib = CudaRTLibrary()
        self.local = self.lib.cudaMalloc(nbytes)
        self.lib.cudaMemset(self.local, 0, nbytes)
        self.lib.cudaDeviceSynchronize()
        handle = self.lib.cudaIpcGetMemHandle(self.local)

        world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        handles: list[bytes | None] = [None] * world_size
        dist.all_gather_object(handles, bytes(handle), group=group)

        self.pointers: list[int] = []
        for rank, raw in enumerate(handles):
            if rank == self.rank:
                pointer = self.local
            else:
                assert raw is not None
                pointer = self.lib.cudaIpcOpenMemHandle(
                    cudaIpcMemHandle_t.from_buffer_copy(raw)
                )
            assert pointer.value is not None
            self.pointers.append(pointer.value)

    def close(self) -> None:
        """Free this rank's allocation; call on every rank after the last use."""
        if self.pointers:
            self.lib.cudaFree(self.local)
            self.pointers = []
