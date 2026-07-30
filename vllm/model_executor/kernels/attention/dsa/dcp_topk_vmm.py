# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side DCP top-k merge over owner-local CUDA VMM storage.

Each rank owns only its ``(max_rows, local_candidates, 2)`` candidate tensor.
CUDA VMM maps those owner allocations into every peer process. After each
producer packs its local candidates, the stable-top-k consumer loads candidates
directly from the rank-major peer mapping; no rank materializes a full gathered
candidate inbox.

The two int64 sequence counters in each owner allocation make reuse and
publication CUDA-graph safe:

1. wait until every consumer has acknowledged the previous owner epoch;
2. pack candidates into the owner-local allocation;
3. release-increment ``write_seq`` and acquire-wait for all producers;
4. consume all owner mappings directly in stable top-k;
5. release-increment ``read_seq``.

This experimental path is fail-closed. It never falls back to an all-gather or
to the symmetric-memory full-inbox implementation.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)
from vllm.distributed.device_communicators.peer_memory import (
    make_rank_major_tensor_view,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_HEADER_BYTES = 256
_WRITE_SEQ = tl.constexpr(0)
_READ_SEQ = tl.constexpr(1)
_MAX_FENCE_SPINS = 100_000_000


@triton.jit
def _trap_if_nonzero(value):
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


@triton.jit
def _wait_writable_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    my_write_seq = tl.atomic_add(
        peer_flags + my_rank * peer_stride + _WRITE_SEQ,
        0,
        sem="acquire",
        scope="sys",
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride + _READ_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < my_write_seq), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride + _READ_SEQ,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < my_write_seq), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _publish_and_wait_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    epoch = (
        tl.atomic_add(
            peer_flags + my_rank * peer_stride + _WRITE_SEQ,
            1,
            sem="release",
            scope="sys",
        )
        + 1
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride + _WRITE_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride + _WRITE_SEQ,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _ack_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
):
    tl.atomic_add(
        peer_flags + my_rank * peer_stride + _READ_SEQ,
        1,
        sem="release",
        scope="sys",
    )


@dataclass
class DcpTopkVmmWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    local_candidates_count: int
    allocation: RankMajorPeerView
    local_candidates: torch.Tensor
    peer_candidates: torch.Tensor
    peer_flags: torch.Tensor

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    @property
    def candidate_payload_bytes_per_rank(self) -> int:
        return self.max_rows * self.local_candidates_count * 2 * 4

    def merge(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_tokens: int,
        dcp_rank: int,
        dcp_world_size: int,
        cp_interleave: int,
        row_starts: torch.Tensor | None,
    ) -> None:
        from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
            pack_dcp_topk_candidates_cutedsl,
            stable_topk_from_rank_major_candidates_cutedsl,
        )

        if dcp_rank != self.my_rank or dcp_world_size != self.world_size:
            raise RuntimeError(
                "DCP VMM workspace geometry changed after initialization: "
                f"workspace=({self.my_rank}, {self.world_size}), "
                f"request=({dcp_rank}, {dcp_world_size})."
            )
        rows = topk_indices.shape[0]
        if rows > self.max_rows:
            raise RuntimeError(
                f"DCP VMM workspace has {self.max_rows} rows, requested {rows}."
            )

        _wait_writable_kernel[(1,)](
            self.peer_flags,
            self.peer_flags.stride(0),
            my_rank=self.my_rank,
            world_size=self.world_size,
            block_size=triton.next_power_of_2(self.world_size),
            max_spins=_MAX_FENCE_SPINS,
        )
        pack_dcp_topk_candidates_cutedsl(
            logits,
            topk_indices[:, : self.local_candidates_count],
            self.local_candidates[:rows],
            dcp_rank,
            dcp_world_size,
            cp_interleave,
            row_starts,
        )
        _publish_and_wait_kernel[(1,)](
            self.peer_flags,
            self.peer_flags.stride(0),
            my_rank=self.my_rank,
            world_size=self.world_size,
            block_size=triton.next_power_of_2(self.world_size),
            max_spins=_MAX_FENCE_SPINS,
        )
        stable_topk_from_rank_major_candidates_cutedsl(
            self.peer_candidates[:, :rows],
            topk_tokens,
            out=topk_indices,
        )
        _ack_kernel[(1,)](
            self.peer_flags,
            self.peer_flags.stride(0),
            my_rank=self.my_rank,
        )

    def close(self) -> None:
        self.peer_flags = None
        self.peer_candidates = None
        self.local_candidates = None
        self.allocation.close()


def create_dcp_topk_vmm_workspace_for_group(
    max_rows: int,
    local_candidates: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpTopkVmmWorkspace:
    """Collectively create an owner-local workspace for a CPU-capable group."""
    world_size = group.size()
    rank = group.rank()
    if world_size <= 1:
        raise RuntimeError("DCP VMM workspace requires dcp_world_size > 1.")
    if (world_size * local_candidates) % 512 != 0:
        raise RuntimeError(
            "DCP VMM stable-topK requires total candidates to be a multiple "
            f"of 512; got {world_size} * {local_candidates}."
        )

    payload_bytes = max_rows * local_candidates * 2 * 4
    requested_bytes = _HEADER_BYTES + payload_bytes
    allocation = create_rank_major_peer_view(
        (requested_bytes,),
        dtype=torch.uint8,
        group=group,
        require_native_atomics=True,
        device=device,
    )
    assert allocation.local_view is not None

    allocation.local_view[:requested_bytes].zero_()
    torch.accelerator.synchronize()
    dist.barrier(group=group)

    local_flags = allocation.local_view[: 2 * 8].view(torch.int64)
    peer_flags = make_rank_major_tensor_view(allocation, local_flags)
    local_payload = allocation.local_view[_HEADER_BYTES : _HEADER_BYTES + payload_bytes]
    local_candidate_tensor = local_payload.view(torch.float32).view(
        max_rows,
        local_candidates,
        2,
    )
    peer_candidates = make_rank_major_tensor_view(
        allocation,
        local_candidate_tensor,
    )
    return DcpTopkVmmWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        local_candidates_count=local_candidates,
        allocation=allocation,
        local_candidates=local_candidate_tensor,
        peer_candidates=peer_candidates,
        peer_flags=peer_flags,
    )


def create_dcp_topk_vmm_workspace(
    max_rows: int,
    local_candidates: int,
) -> DcpTopkVmmWorkspace:
    """Collectively create an owner-local workspace for the DCP group."""
    from vllm.distributed import get_dcp_group

    dcp_group = get_dcp_group()
    return create_dcp_topk_vmm_workspace_for_group(
        max_rows,
        local_candidates,
        dcp_group.cpu_group,
        dcp_group.device,
    )


_workspace: DcpTopkVmmWorkspace | None = None
_workspace_failed = False


def get_dcp_topk_vmm_workspace(
    max_rows: int,
    local_candidates: int,
    dcp_world_size: int,
) -> DcpTopkVmmWorkspace | None:
    """Create or fetch the singleton VMM workspace, with no fallback."""
    global _workspace, _workspace_failed
    if dcp_world_size <= 1:
        return None
    if _workspace_failed:
        raise RuntimeError("DCP VMM top-k workspace is unavailable.")
    if _workspace is not None:
        if (
            _workspace.max_rows < max_rows
            or _workspace.local_candidates_count != local_candidates
            or _workspace.world_size != dcp_world_size
        ):
            raise RuntimeError(
                "DCP VMM workspace shape mismatch: "
                f"workspace=({_workspace.max_rows}, "
                f"{_workspace.local_candidates_count}, {_workspace.world_size}), "
                f"request=({max_rows}, {local_candidates}, {dcp_world_size})."
            )
        return _workspace

    try:
        _workspace = create_dcp_topk_vmm_workspace(max_rows, local_candidates)
    except Exception as exc:
        _workspace_failed = True
        raise RuntimeError(
            "DCP VMM top-k workspace initialization failed; refusing to "
            "fall back to another exchange path."
        ) from exc
    logger.info_once(
        "Using owner-local CUDA VMM DCP top-k workspace "
        "(max_rows=%d, candidates_per_rank=%d, physical_bytes_per_rank=%d).",
        _workspace.max_rows,
        _workspace.local_candidates_count,
        _workspace.physical_bytes_per_rank,
    )
    return _workspace
