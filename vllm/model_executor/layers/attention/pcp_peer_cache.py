# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-major cache views and visibility fencing for direct PCP stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_vmm import RankMajorPeerView

_MAX_FENCE_SPINS = 100_000_000


@triton.jit
def _trap_if_nonzero(value):
    """Unconditionally compile a device trap when a bounded wait times out."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def make_rank_major_tensor_view(
    allocation: RankMajorPeerView,
    local_tensor: torch.Tensor,
) -> torch.Tensor:
    """Mirror a tensor view across every rank segment of a VMM allocation.

    ``local_tensor`` may be a packed or strided view into ``local_view``. The
    returned tensor adds a leading PCP-rank dimension without copying bytes.
    """
    element_size = local_tensor.element_size()
    if allocation.bytes_per_rank % element_size != 0:
        raise ValueError(
            "Peer allocation stride is not divisible by the tensor element size: "
            f"{allocation.bytes_per_rank} bytes vs {element_size}."
        )

    local_offset_bytes = local_tensor.data_ptr() - allocation.local_view.data_ptr()
    view_span = 0 if local_tensor.numel() == 0 else 1
    for size, stride in zip(local_tensor.shape, local_tensor.stride()):
        if size > 0:
            view_span += (size - 1) * stride
    view_span_bytes = view_span * element_size
    if (
        local_offset_bytes < 0
        or local_offset_bytes + view_span_bytes > allocation.bytes_per_rank
    ):
        raise ValueError("Tensor view lies outside its local peer allocation.")
    if local_offset_bytes % element_size != 0:
        raise ValueError("Tensor view is not aligned to its element size.")

    global_typed = allocation.global_view.view(local_tensor.dtype)
    rank_stride = allocation.bytes_per_rank // element_size
    return torch.as_strided(
        global_typed,
        size=(allocation.world_size, *local_tensor.shape),
        stride=(rank_stride, *local_tensor.stride()),
        storage_offset=local_offset_bytes // element_size,
    )


@triton.jit
def _publish_fence_kernel(
    peer_signal_ptr,
    peer_rank_stride,
    epoch,
    parity,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    destination_rank = tl.program_id(0)
    if destination_rank < world_size:
        signal_ptr = (
            peer_signal_ptr
            + destination_rank * peer_rank_stride
            + parity * world_size
            + source_rank
        )
        # The release publishes every cache store from the preceding kernel to
        # all GPUs in the peer-memory domain.
        tl.atomic_xchg(signal_ptr, epoch, sem="release", scope="sys")


@triton.jit
def _wait_fence_kernel(
    local_signal_ptr,
    epoch,
    parity,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    source_rank = tl.arange(0, BLOCK_SIZE)
    mask = source_rank < world_size
    signal_ptr = local_signal_ptr + parity * world_size + source_rank
    observed = tl.atomic_add(
        signal_ptr,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < MAX_SPINS):
        observed = tl.atomic_add(
            signal_ptr,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


class PCPPeerCacheFence:
    """Reusable release/acquire fence for one direct-store PCP group.

    Publication and waiting are separate kernels. Stream ordering therefore
    completes all producer cache stores before any CTA publishes its signal,
    avoiding an intra-grid producer/consumer deadlock.
    """

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        from vllm.distributed.device_communicators.cuda_vmm import (
            create_rank_major_peer_view,
        )

        self._group = group
        self._world_size = group.size()
        self._rank = group.rank()
        self._epoch = 0
        self._allocation = create_rank_major_peer_view(
            (2, self._world_size),
            dtype=torch.int32,
            group=group,
            require_native_atomics=True,
            device=device,
        )
        self._allocation.local_view.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=group)
        self._peer_signals = make_rank_major_tensor_view(
            self._allocation, self._allocation.local_view
        )

    def __call__(self) -> None:
        self._epoch = self._epoch % 0x7FFFFFFE + 1
        parity = self._epoch & 1
        _publish_fence_kernel[(self._world_size,)](
            self._peer_signals,
            self._peer_signals.stride(0),
            self._epoch,
            parity,
            source_rank=self._rank,
            world_size=self._world_size,
        )
        _wait_fence_kernel[(1,)](
            self._allocation.local_view,
            self._epoch,
            parity,
            world_size=self._world_size,
            BLOCK_SIZE=triton.next_power_of_2(self._world_size),
            MAX_SPINS=_MAX_FENCE_SPINS,
        )

    def close(self) -> None:
        # Quiesce every producer/consumer before any rank unmaps its signal
        # segment. A local synchronize alone cannot prove that a peer GPU has
        # finished a remote store or acquire wait against this allocation.
        torch.cuda.synchronize(self._allocation.device)
        dist.barrier(group=self._group)
        self._allocation.close()
