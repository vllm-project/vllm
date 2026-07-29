# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side DCP top-k merge over owner-local symmetric memory.

Each rank owns only its ``(max_rows, local_candidates, 2)`` candidate tensor.
PyTorch symmetric-memory rendezvous exposes locally valid pointers to every
owner allocation. After each producer packs its local candidates, the
stable-top-k consumer follows the peer-pointer table directly; no rank
materializes a full gathered candidate inbox.

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
from typing import Any

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

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
    peer_flag_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    ptrs = peer_flag_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    my_write_seq = tl.atomic_add(
        my_flags + _WRITE_SEQ,
        0,
        sem="acquire",
        scope="sys",
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    peer_flags = tl.load(ptrs + peer, mask=mask, other=0).to(tl.pointer_type(tl.int64))
    observed = tl.atomic_add(
        peer_flags + _READ_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < my_write_seq), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + _READ_SEQ,
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
    peer_flag_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    ptrs = peer_flag_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    epoch = (
        tl.atomic_add(
            my_flags + _WRITE_SEQ,
            1,
            sem="release",
            scope="sys",
        )
        + 1
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    peer_flags = tl.load(ptrs + peer, mask=mask, other=0).to(tl.pointer_type(tl.int64))
    observed = tl.atomic_add(
        peer_flags + _WRITE_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + _WRITE_SEQ,
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
    peer_flag_ptrs,
    my_rank: tl.constexpr,
):
    ptrs = peer_flag_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    tl.atomic_add(
        my_flags + _READ_SEQ,
        1,
        sem="release",
        scope="sys",
    )


@dataclass
class DcpTopkSymmWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    local_candidates_count: int
    candidate_handle: Any
    flag_handle: Any
    allocation_bytes_per_rank: int
    local_candidates: torch.Tensor | None
    local_flags: torch.Tensor | None
    peer_candidates: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    )
    flag_ptrs: torch.Tensor | None

    @property
    def logical_bytes_per_rank(self) -> int:
        return self.candidate_payload_bytes_per_rank + 2 * 8 + self.world_size * 8

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
            stable_topk_from_peer_candidates_cutedsl,
        )

        if dcp_rank != self.my_rank or dcp_world_size != self.world_size:
            raise RuntimeError(
                "DCP symmetric-memory workspace geometry changed after "
                "initialization: "
                f"workspace=({self.my_rank}, {self.world_size}), "
                f"request=({dcp_rank}, {dcp_world_size})."
            )
        rows = topk_indices.shape[0]
        if rows > self.max_rows:
            raise RuntimeError(
                "DCP symmetric-memory workspace has "
                f"{self.max_rows} rows, requested {rows}."
            )
        if (
            self.local_candidates is None
            or self.peer_candidates is None
            or self.flag_ptrs is None
        ):
            raise RuntimeError("DCP symmetric-memory workspace is closed.")

        _wait_writable_kernel[(1,)](
            self.flag_ptrs,
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
            self.flag_ptrs,
            my_rank=self.my_rank,
            world_size=self.world_size,
            block_size=triton.next_power_of_2(self.world_size),
            max_spins=_MAX_FENCE_SPINS,
        )
        stable_topk_from_peer_candidates_cutedsl(
            self.peer_candidates,
            self.world_size,
            rows,
            self.local_candidates_count,
            topk_tokens,
            out=topk_indices,
        )
        _ack_kernel[(1,)](
            self.flag_ptrs,
            my_rank=self.my_rank,
        )

    def close(self) -> None:
        if self.local_candidates is None:
            return
        torch.accelerator.synchronize(self.local_candidates.device)
        self.flag_ptrs = None
        self.peer_candidates = None
        self.local_flags = None
        self.local_candidates = None
        self.flag_handle = None
        self.candidate_handle = None


def create_dcp_topk_symm_workspace_for_group(
    max_rows: int,
    local_candidates: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpTopkSymmWorkspace:
    """Collectively create one owner-local candidate shard per DCP rank."""
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except ImportError as exc:
        raise RuntimeError("PyTorch symmetric memory is unavailable.") from exc

    world_size = group.size()
    rank = group.rank()
    if world_size <= 1:
        raise RuntimeError(
            "DCP symmetric-memory workspace requires dcp_world_size > 1."
        )
    if world_size != 4:
        raise RuntimeError("DCP symmetric-memory peer-shard top-k requires DCP4.")
    if (world_size * local_candidates) % 512 != 0:
        raise RuntimeError(
            "DCP symmetric-memory stable-topK requires total candidates to be "
            f"a multiple of 512; got {world_size} * {local_candidates}."
        )

    local_candidates_tensor = symm_mem.empty(
        (max_rows, local_candidates, 2),
        dtype=torch.float32,
        device=device,
    )
    local_flags = symm_mem.empty(
        (2,),
        dtype=torch.int64,
        device=device,
    )
    local_candidates_tensor.zero_()
    local_flags.zero_()
    torch.accelerator.synchronize(device)
    candidate_handle = symm_mem.rendezvous(
        local_candidates_tensor,
        group.group_name,
    )
    flag_handle = symm_mem.rendezvous(
        local_flags,
        group.group_name,
    )
    if candidate_handle is None or flag_handle is None:
        raise RuntimeError("DCP symmetric-memory rendezvous returned no handle.")
    candidate_handle.barrier()
    flag_handle.barrier()
    peer_candidates = (
        candidate_handle.get_buffer(
            0,
            (max_rows, local_candidates, 2),
            torch.float32,
        ),
        candidate_handle.get_buffer(
            1,
            (max_rows, local_candidates, 2),
            torch.float32,
        ),
        candidate_handle.get_buffer(
            2,
            (max_rows, local_candidates, 2),
            torch.float32,
        ),
        candidate_handle.get_buffer(
            3,
            (max_rows, local_candidates, 2),
            torch.float32,
        ),
    )
    flag_ptrs = torch.tensor(
        flag_handle.buffer_ptrs,
        dtype=torch.int64,
        device=device,
    )
    if bool(flag_ptrs.eq(0).any()):
        raise RuntimeError("DCP symmetric-memory peer mapping contains a null pointer.")

    return DcpTopkSymmWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        local_candidates_count=local_candidates,
        candidate_handle=candidate_handle,
        flag_handle=flag_handle,
        allocation_bytes_per_rank=(
            int(candidate_handle.buffer_size)
            + int(flag_handle.buffer_size)
            + flag_ptrs.numel() * flag_ptrs.element_size()
        ),
        local_candidates=local_candidates_tensor,
        local_flags=local_flags,
        peer_candidates=peer_candidates,
        flag_ptrs=flag_ptrs,
    )


def create_dcp_topk_symm_workspace(
    max_rows: int,
    local_candidates: int,
) -> DcpTopkSymmWorkspace:
    """Collectively create an owner-local workspace for the DCP group."""
    from vllm.distributed import get_dcp_group

    dcp_group = get_dcp_group()
    return create_dcp_topk_symm_workspace_for_group(
        max_rows,
        local_candidates,
        dcp_group.device_group,
        dcp_group.device,
    )


_workspace: DcpTopkSymmWorkspace | None = None
_workspace_failed = False


def get_dcp_topk_symm_workspace(
    max_rows: int,
    local_candidates: int,
    dcp_world_size: int,
) -> DcpTopkSymmWorkspace | None:
    """Create or fetch the singleton symmetric workspace, with no fallback."""
    global _workspace, _workspace_failed
    if dcp_world_size <= 1:
        return None
    if _workspace_failed:
        raise RuntimeError("DCP symmetric-memory top-k workspace is unavailable.")
    if _workspace is not None:
        if (
            _workspace.max_rows < max_rows
            or _workspace.local_candidates_count != local_candidates
            or _workspace.world_size != dcp_world_size
        ):
            raise RuntimeError(
                "DCP symmetric-memory workspace shape mismatch: "
                f"workspace=({_workspace.max_rows}, "
                f"{_workspace.local_candidates_count}, {_workspace.world_size}), "
                f"request=({max_rows}, {local_candidates}, {dcp_world_size})."
            )
        return _workspace

    try:
        _workspace = create_dcp_topk_symm_workspace(max_rows, local_candidates)
    except Exception as exc:
        _workspace_failed = True
        raise RuntimeError(
            "DCP symmetric-memory top-k workspace initialization failed; "
            "refusing to fall back to another exchange path."
        ) from exc
    logger.info_once(
        "Using owner-local symmetric-memory DCP top-k workspace "
        "(max_rows=%d, candidates_per_rank=%d, logical_bytes_per_rank=%d, "
        "allocation_bytes_per_rank=%d).",
        _workspace.max_rows,
        _workspace.local_candidates_count,
        _workspace.logical_bytes_per_rank,
        _workspace.allocation_bytes_per_rank,
    )
    return _workspace
