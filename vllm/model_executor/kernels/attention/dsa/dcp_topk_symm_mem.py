# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DCP top-k candidate exchange over PyTorch symmetric memory.

Replaces the NCCL all-gather in the DCP sparse-indexer top-k merge with a
one-shot all-gather implemented as device-side remote writes: each rank packs
its (score, global_id) candidates into its own slice of every rank's
symmetric-memory "inbox", so after a flag handshake each rank's inbox holds
the full gathered layout in local memory and the existing CuTeDSL selector
runs on it unchanged. All synchronization is device-side (sequence flags with
sys-scope acquire/release), so the whole chain is CUDA-graph capturable: no
host branches, no NCCL launch.

Per-rank symmetric buffer layout (offsets in bytes):
  [0]    write_seq   int64 - exchanges this rank has published
  [8]    read_seq    int64 - exchanges this rank has finished consuming
  [256]  inbox       (max_rows, world, L, 2) fp32 gathered candidates

Per merge invocation (exchange n, 1-based), every rank runs:
  1. wait_writable: spin until every peer's read_seq >= my write_seq (== n-1),
     i.e. nobody is still reading the inbox slices we are about to overwrite.
  2. pack candidates into rank slice `my_rank` of the local inbox.
  3. push: copy that slice into rank slice `my_rank` of every peer's inbox.
  4. publish_and_wait: release-increment my write_seq to n, then spin until
     every peer's write_seq >= n.
  5. run the stable top-k selector over the (now complete, local) inbox.
  6. ack: release-increment my read_seq to n.
All ranks run the same per-layer sequence, so the local write_seq doubles as
the expected exchange number on both the producer and consumer side.
"""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_HEADER_BYTES = 256
_WRITE_SEQ_OFF = tl.constexpr(0)  # int64 index into the flag header
_READ_SEQ_OFF = tl.constexpr(1)
_PUSH_BLOCK = 1024


@triton.jit
def _wait_writable_kernel(
    buffer_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    my_write_seq = tl.load(my_flags + _WRITE_SEQ_OFF)
    for peer in tl.static_range(world_size):
        if peer != my_rank:
            peer_flags = tl.load(ptrs + peer).to(tl.pointer_type(tl.int64))
            read_seq = tl.atomic_add(
                peer_flags + _READ_SEQ_OFF, 0, sem="acquire", scope="sys"
            )
            while read_seq < my_write_seq:
                read_seq = tl.atomic_add(
                    peer_flags + _READ_SEQ_OFF, 0, sem="acquire", scope="sys"
                )


@triton.jit
def _push_candidates_kernel(
    buffer_ptrs,
    num_elems,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    local_elems: tl.constexpr,  # L * 2, elements per rank slice per row
    header_elems: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Broadcast this rank's packed slice from the local inbox into the same
    # slice of every peer's inbox. Element i of the slice lives at flat inbox
    # offset (row * world_size + my_rank) * local_elems + rem.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_elems
    row = offs // local_elems
    rem = offs % local_elems
    flat = (row * world_size + my_rank) * local_elems + rem

    ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    my_base = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.float32))
    vals = tl.load(my_base + header_elems + flat, mask=mask)
    for peer in tl.static_range(world_size):
        if peer != my_rank:
            peer_base = tl.load(ptrs + peer).to(tl.pointer_type(tl.float32))
            tl.store(peer_base + header_elems + flat, vals, mask=mask)


@triton.jit
def _publish_and_wait_kernel(
    buffer_ptrs,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    epoch = tl.atomic_add(my_flags + _WRITE_SEQ_OFF, 1, sem="release", scope="sys") + 1
    for peer in tl.static_range(world_size):
        if peer != my_rank:
            peer_flags = tl.load(ptrs + peer).to(tl.pointer_type(tl.int64))
            write_seq = tl.atomic_add(
                peer_flags + _WRITE_SEQ_OFF, 0, sem="acquire", scope="sys"
            )
            while write_seq < epoch:
                write_seq = tl.atomic_add(
                    peer_flags + _WRITE_SEQ_OFF, 0, sem="acquire", scope="sys"
                )


@triton.jit
def _ack_kernel(
    buffer_ptrs,
    my_rank: tl.constexpr,
):
    ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    my_flags = tl.load(ptrs + my_rank).to(tl.pointer_type(tl.int64))
    tl.atomic_add(my_flags + _READ_SEQ_OFF, 1, sem="release", scope="sys")


@dataclass
class DcpTopkSymmMemWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    local_candidates: int
    buffer_ptrs_dev: int
    inbox: torch.Tensor  # local (max_rows, world, L, 2) fp32 view
    # get_buffer views do not own the allocation; keep it (and the rendezvous
    # handle) alive for the workspace lifetime.
    local_buffer: torch.Tensor
    handle: object

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
            stable_topk_from_gathered_candidates_cutedsl,
        )

        rows = topk_indices.shape[0]
        _wait_writable_kernel[(1,)](
            self.buffer_ptrs_dev,
            my_rank=self.my_rank,
            world_size=self.world_size,
        )
        pack_dcp_topk_candidates_cutedsl(
            logits,
            topk_indices[:, : self.local_candidates],
            self.inbox[:rows, self.my_rank],
            dcp_rank,
            dcp_world_size,
            cp_interleave,
            row_starts,
        )
        num_elems = rows * self.local_candidates * 2
        _push_candidates_kernel[(triton.cdiv(num_elems, _PUSH_BLOCK),)](
            self.buffer_ptrs_dev,
            num_elems,
            my_rank=self.my_rank,
            world_size=self.world_size,
            local_elems=self.local_candidates * 2,
            header_elems=_HEADER_BYTES // 4,
            BLOCK=_PUSH_BLOCK,
        )
        _publish_and_wait_kernel[(1,)](
            self.buffer_ptrs_dev,
            my_rank=self.my_rank,
            world_size=self.world_size,
        )
        gathered = self.inbox[:rows].reshape(
            rows, self.world_size * self.local_candidates, 2
        )
        stable_topk_from_gathered_candidates_cutedsl(
            gathered, topk_tokens, out=topk_indices
        )
        _ack_kernel[(1,)](
            self.buffer_ptrs_dev,
            my_rank=self.my_rank,
        )


_workspace: DcpTopkSymmMemWorkspace | None = None
_workspace_failed = False


def _create_workspace(
    max_rows: int, local_candidates: int
) -> DcpTopkSymmMemWorkspace | None:
    import torch.distributed._symmetric_memory as symm_mem

    from vllm.distributed import get_dcp_group

    group = get_dcp_group()
    device_group = group.device_group
    if device_group is None:
        return None

    world = group.world_size
    inbox_bytes = max_rows * world * local_candidates * 2 * 4
    local = symm_mem.empty(
        (_HEADER_BYTES + inbox_bytes,),
        dtype=torch.uint8,
        device=group.device,
    )
    local.zero_()
    handle = symm_mem.rendezvous(local, device_group.group_name)
    # Publish the zeroed flags before any rank starts polling them.
    handle.barrier(channel=0)

    inbox = handle.get_buffer(
        group.rank_in_group,
        (max_rows, world, local_candidates, 2),
        torch.float32,
        storage_offset=_HEADER_BYTES // 4,
    )
    return DcpTopkSymmMemWorkspace(
        my_rank=group.rank_in_group,
        world_size=world,
        max_rows=max_rows,
        local_candidates=local_candidates,
        buffer_ptrs_dev=handle.buffer_ptrs_dev,
        inbox=inbox,
        local_buffer=local,
        handle=handle,
    )


def get_dcp_topk_symm_mem_workspace(
    max_rows: int,
    local_candidates: int,
    dcp_world_size: int,
) -> DcpTopkSymmMemWorkspace | None:
    """Create (once) or fetch the symm-mem candidate-exchange workspace.

    Returns None when unsupported; callers fall back to the NCCL all-gather.
    The first call must be made collectively by all DCP ranks (it performs a
    rendezvous) and before CUDA graph capture.
    """
    global _workspace, _workspace_failed
    if _workspace_failed:
        return None
    if _workspace is not None:
        if (
            _workspace.local_candidates != local_candidates
            or _workspace.max_rows < max_rows
        ):
            return None
        return _workspace

    if dcp_world_size <= 1 or (dcp_world_size * local_candidates) % 512 != 0:
        _workspace_failed = True
        return None
    try:
        _workspace = _create_workspace(max_rows, local_candidates)
    except Exception:
        logger.warning(
            "DCP top-k symmetric-memory workspace unavailable; "
            "falling back to the all-gather merge path.",
            exc_info=True,
        )
        _workspace = None
    if _workspace is None:
        _workspace_failed = True
    else:
        logger.info_once(
            "Using symmetric-memory fused DCP top-k candidate exchange "
            "(max_rows=%d, candidates_per_rank=%d).",
            _workspace.max_rows,
            _workspace.local_candidates,
        )
    return _workspace
