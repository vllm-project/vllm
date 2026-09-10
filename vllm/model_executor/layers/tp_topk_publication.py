# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish disjoint TP query rows into the existing final TopK layout."""

from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


@triton.jit
def _publish_rows(
    src,
    peers,
    start,
    rows,
    width: tl.constexpr,
    stride: tl.constexpr,
    rank: tl.constexpr,
    BLOCK: tl.constexpr,
):
    peer = tl.program_id(1)
    if peer != rank:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = offsets // width
        col = offsets % width
        address = (start + row) * stride + col
        values = tl.load(src + address, row < rows, other=-1)
        dst = tl.load(peers + peer).to(tl.pointer_type(tl.int32))
        tl.store(dst + address, values, row < rows)


class TPTopKPublication:
    def __init__(self, buffer: torch.Tensor, group: "GroupCoordinator") -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.handle = symm_mem.rendezvous(buffer, group.device_group.group_name)
        self.rank = group.rank_in_group
        self.world_size = group.world_size
        # Resolve view addresses once, including any allocator storage offset.
        self.views = [
            self.handle.get_buffer(r, tuple(buffer.shape), buffer.dtype)
            for r in range(self.world_size)
        ]
        self.peers = torch.tensor(
            [view.data_ptr() for view in self.views],
            dtype=torch.uint64,
            device=buffer.device,
        )

    def begin(self) -> None:
        # Every lane has finished its preceding consumer and local buffer clear.
        self.handle.barrier(channel=0)

    def finish(self) -> None:
        # Publish all producer stores before any sparse MLA consumer starts.
        self.handle.barrier(channel=0)

    def publish(self, buffer: torch.Tensor, start: int, rows: int, width: int) -> None:
        _publish_rows[(triton.cdiv(rows * width, 1024), self.world_size)](
            buffer, self.peers, start, rows, width, buffer.stride(0), self.rank, 1024
        )
        self.finish()


_publications: WeakValueDictionary[int, TPTopKPublication] = WeakValueDictionary()


def allocate_topk_buffer(
    rows: int, width: int, *, dtype: torch.dtype, device: str | torch.device
) -> torch.Tensor:
    if not envs.VLLM_TP_TOPK_DIRECT or get_tp_group().world_size == 1:
        return torch.empty((rows, width), dtype=dtype, device=device)
    import torch.distributed._symmetric_memory as symm_mem

    buffer = symm_mem.empty((rows, width), dtype=dtype, device=device)
    publication = TPTopKPublication(buffer, get_tp_group())
    # The model owns the original tensor for the lifetime of its consumers.
    # Lookup by address also works when custom-op dispatch wraps the Tensor.
    buffer._tp_topk_publication = publication
    _publications[buffer.data_ptr()] = publication
    return buffer


def get_topk_publication(buffer: torch.Tensor) -> TPTopKPublication | None:
    return _publications.get(buffer.data_ptr())
