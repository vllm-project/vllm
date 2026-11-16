# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct DCP top-k candidate exchange over PyTorch symmetric memory.

Replaces the NCCL all-gather of the DCP indexer merge for small batches. Each
rank packs its local (score, global token id) candidates, stores them into
every peer's inbox slice, then publishes an epoch and waits for all peers. The
stable top-k selector reads the completed local inbox directly. Two alternating
inboxes let the next exchange start without a read acknowledgement: a rank can
only overwrite a slot after every peer has published the exchange in between,
i.e. after their selectors consumed it.

On SM90+, programmatic dependent launches overlap the dispatch of the pack,
push, publish and select kernels; device-side waits keep the data dependencies.
"""

import functools

import torch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.cp_common import DirectCPWorkspace, direct_cp_enabled

logger = init_logger(__name__)


@triton.jit
def _push_candidates_kernel(
    peer_inbox_ptrs,
    write_seq,
    packed,
    num_elems,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    local_elems: tl.constexpr,
    slot_elems: tl.constexpr,
    block_size: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offs < num_elems
    row = offs // local_elems
    rem = offs % local_elems
    slot = tl.load(write_seq) & 1
    dst_offs = slot * slot_elems + (row * world_size + my_rank) * local_elems + rem
    # The push may be dispatched before the preceding pack kernel retires.
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    values = tl.load(packed + offs, mask=mask)

    ptrs = peer_inbox_ptrs.to(tl.pointer_type(tl.uint64))
    for peer in tl.static_range(world_size):
        peer_inbox = tl.load(ptrs + peer).to(tl.pointer_type(tl.float32))
        tl.store(peer_inbox + dst_offs, values, mask=mask)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _publish_and_wait_kernel(
    peer_signal_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    # Do not publish this rank's epoch until every push CTA completed its stores.
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    ptrs = peer_signal_ptrs.to(tl.pointer_type(tl.uint64))
    my_signal = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    epoch = tl.atomic_add(my_signal, 1, sem="release", scope="sys") + 1

    peer = tl.arange(0, block_size)
    mask = peer < world_size
    peer_signal = tl.load(ptrs + peer, mask=mask, other=0).to(tl.pointer_type(tl.int64))
    observed = tl.atomic_add(peer_signal, 0, mask=mask, sem="acquire", scope="sys")
    pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(peer_signal, 0, mask=mask, sem="acquire", scope="sys")
        pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
        spins += 1
    # Trap instead of silently reading a stale inbox if a peer never arrives.
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[pending],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    # Release the selector only after every peer epoch is visible.
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()


class DirectDCPTopKWorkspace(DirectCPWorkspace):
    """Symmetric ping-pong inboxes for the DCP top-k candidate exchange."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        device: torch.device,
        max_rows: int,
        candidates_per_rank: int,
    ) -> None:
        super().__init__(group, device, num_ubatches=1)
        self.max_rows = max_rows
        self.candidates_per_rank = candidates_per_rank
        self.use_pdl = current_platform.is_arch_support_pdl()
        # [ubatch, ping-pong slot, row, source rank, candidate, (score, token id)]
        inbox_shape = (1, 2, max_rows, self.world_size, candidates_per_rank, 2)
        inboxes, peer_inbox_ptrs = self._allocate(inbox_shape, torch.float32)
        signals, peer_signal_ptrs = self._allocate((1, 1), torch.int64)
        self.inboxes = inboxes[0].reshape(
            2, max_rows, self.world_size * candidates_per_rank, 2
        )
        self.peer_inbox_ptrs = peer_inbox_ptrs[0]
        # This rank's published epoch; its parity selects the inbox to write.
        self.write_seq = signals[0]
        self.peer_signal_ptrs = peer_signal_ptrs[0]
        self.packed = torch.empty(
            (max_rows, candidates_per_rank, 2), dtype=torch.float32, device=device
        )

    def merge(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        cp_interleave: int,
        row_starts: torch.Tensor | None,
        score_bits: int = 32,
        id_bits: int = 32,
    ) -> None:
        from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
            pack_dcp_topk_candidates_cutedsl,
        )
        from vllm.model_executor.kernels.attention.dsa.dcp_topk_pingpong_cutedsl import (  # noqa: E501
            stable_topk_from_pingpong_candidates_cutedsl,
        )

        rows = topk_indices.shape[0]
        assert 0 < rows <= self.max_rows
        pack_dcp_topk_candidates_cutedsl(
            logits,
            topk_indices,
            self.packed[:rows],
            self.rank,
            self.world_size,
            cp_interleave,
            row_starts,
        )
        num_elems = rows * self.candidates_per_rank * 2
        block_size = 1024
        _push_candidates_kernel[(triton.cdiv(num_elems, block_size),)](
            self.peer_inbox_ptrs,
            self.write_seq,
            self.packed,
            num_elems,
            my_rank=self.rank,
            world_size=self.world_size,
            local_elems=self.candidates_per_rank * 2,
            slot_elems=self.max_rows * self.world_size * self.candidates_per_rank * 2,
            block_size=block_size,
            launch_pdl=self.use_pdl,
        )
        _publish_and_wait_kernel[(1,)](
            self.peer_signal_ptrs,
            my_rank=self.rank,
            world_size=self.world_size,
            block_size=triton.next_power_of_2(self.world_size),
            max_spins=100_000_000,
            launch_pdl=self.use_pdl,
        )
        stable_topk_from_pingpong_candidates_cutedsl(
            self.inboxes,
            self.write_seq,
            self.candidates_per_rank,
            out=topk_indices,
            use_pdl=self.use_pdl,
            score_bits=score_bits,
            id_bits=id_bits,
        )


@functools.cache
def get_direct_dcp_topk_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_rows: int,
    candidates_per_rank: int,
    use_direct: bool | None,
) -> DirectDCPTopKWorkspace | None:
    """Collective on first call (symmetric-memory rendezvous): every DCP rank
    must call it with the same arguments, outside CUDA graph capture."""
    if not direct_cp_enabled(group, torch.float32, use_direct):
        return None
    assert group.device_group is not None
    workspace = DirectDCPTopKWorkspace(
        group.device_group, device, max_rows, candidates_per_rank
    )
    logger.info_once(
        "Using symmetric-memory direct DCP top-k candidate exchange "
        "(max_rows=%d, candidates_per_rank=%d).",
        max_rows,
        candidates_per_rank,
    )
    return workspace
