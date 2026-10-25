# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffers of the context-parallel decode indexer (TP4, gfx950).

Rank ``me`` holds index head ``me``; ``kernel_cp_push`` scores a quarter of every
request's blocks for all four heads and stores head h's scores into rank h's
receive buffer. Buffers and flags are uncached memory mapped by the peers; the
call counts stay on the device, so the handshake replays in CUDA graphs.
"""

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (
    kernel_cp_push as _kernel,
)

WORLD = _kernel.WORLD


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class IndexCP:
    def __init__(
        self, rank: int, group, device: torch.device, max_rows: int, max_width: int
    ):
        self.rank, self.device = rank, device
        self.max_rows, self.max_width = max_rows, max_width
        score_bytes = max_rows * max_width * 4
        iq_bytes = WORLD * max_rows * _kernel.D
        # Each rank's buffer, mapped by every peer (uncached on ROCm).
        self.score_ptrs, self.iq_ptrs, self.flag_ptrs = (
            CustomAllreduce.create_shared_buffer(n, group=group)
            for n in (score_bytes, iq_bytes, _kernel.FLAG_BYTES)
        )
        self.score_rcv = torch.as_tensor(
            _DevicePointer(self.score_ptrs[rank], score_bytes), device=device
        ).view(torch.float32)
        flags = torch.as_tensor(
            _DevicePointer(self.flag_ptrs[rank], _kernel.FLAG_BYTES), device=device
        )
        flags.zero_()
        # Call counts (the flag values), equal on every rank.
        self.counts = torch.zeros(_kernel.CNT_WORDS, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)

    def score_push(
        self,
        iq: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        width: int,
        grid: int = 0,
        query_len: int = 4,
    ) -> torch.Tensor:
        """Score this rank's blocks of every head from its index-q rows ``iq``.

        Returns this rank's head over every block, [1, rows, width]: a view of
        the receive buffer, valid until the next call.
        """
        from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode.host import (
            minimax_m3_nvfp4_index_score_push_cp,
        )

        rows = iq.shape[0]
        assert rows <= self.max_rows and width <= self.max_width
        minimax_m3_nvfp4_index_score_push_cp(
            iq,
            index_cache,
            self.score_ptrs,
            width,
            rows * width * 4,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            self.rank,
            self.counts,
            self.flag_ptrs,
            self.iq_ptrs,
            self.max_rows,
            grid=grid,
            query_len=query_len,
        )
        return self.score_rcv[: rows * width].view(1, rows, width)


# None means the single-head decode path.
_CP: IndexCP | None = None


def init(device: torch.device, max_width: int) -> None:
    """Create this rank's ``IndexCP``; a collective, called once outside capture."""
    global _CP
    if _CP is not None:
        return
    from vllm.distributed import get_tp_group
    from vllm.platforms.rocm import on_gfx950

    tp = get_tp_group()
    if tp.world_size != WORLD or not on_gfx950():
        return
    _CP = IndexCP(tp.rank_in_group, tp.cpu_group, device, _kernel.MAX_ROWS, max_width)


def get() -> IndexCP | None:
    return _CP
