# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM producer fanout for DCP MLA query-head shards.

Each rank produces only its local FP8 query heads, but writes that small shard
directly into every consumer's owner-local dense query inbox. The unchanged
sparse-attention backend then reads a complete local query without Query
AllGather or remote query loads.

The workspace is reusable and CUDA-graph safe. Device sequence counters prevent
producers from overwriting an inbox before its consumer has finished. Setup and
publication failures are fail-closed; intentional large-row or non-decode
collective routing must be selected before invoking this workspace.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.profiler import record_function

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

_SIGNAL_RESERVE_BYTES = 256
_SIGNAL_BYTES = 2 * 8
_WRITE_SEQ = tl.constexpr(0)
_READ_SEQ = tl.constexpr(1)
_MAX_FENCE_SPINS = 100_000_000
DEFAULT_MAX_ROWS = 128
_logged_consume_rows: set[int] = set()


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
def _publish_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
):
    tl.atomic_add(
        peer_flags + my_rank * peer_stride + _WRITE_SEQ,
        1,
        sem="release",
        scope="sys",
    )


@triton.jit
def _wait_published_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    epoch = tl.atomic_add(
        peer_flags + my_rank * peer_stride + _WRITE_SEQ,
        0,
        sem="acquire",
        scope="sys",
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
class DcpQueryVmmWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    local_heads: int
    query_dim: int
    group: ProcessGroup
    device: torch.device
    allocation: RankMajorPeerView
    local_consumer_query: torch.Tensor
    peer_consumer_queries: torch.Tensor
    peer_flags: torch.Tensor
    publishing_rows: int = 0
    published_rows: int = 0
    consumer_rows: int = 0

    @property
    def total_heads(self) -> int:
        return self.local_heads * self.world_size

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    @property
    def payload_bytes_per_rank(self) -> int:
        return self.max_rows * self.total_heads * self.query_dim

    def _validate_live(self) -> None:
        if self.allocation.closed:
            raise RuntimeError("DCP query VMM workspace is closed.")
        current_device = torch.accelerator.current_device_index()
        if current_device != self.device.index:
            raise RuntimeError(
                "DCP query VMM current device changed after initialization: "
                f"workspace={self.device}, current=cuda:{current_device}."
            )

    def begin_publish(self, rows: int) -> torch.Tensor:
        """Wait for reuse safety and return this producer's fanout targets."""
        self._validate_live()
        if self.publishing_rows or self.published_rows:
            raise RuntimeError("DCP query VMM publication is already in progress.")
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                "DCP query VMM producer row bound violated: "
                f"max_rows={self.max_rows}, requested={rows}."
            )
        with record_function("dcp.query_vmm.wait_reuse"):
            _wait_writable_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )
        self.publishing_rows = rows
        head_start = self.my_rank * self.local_heads
        head_end = head_start + self.local_heads
        return self.peer_consumer_queries[:, :rows, head_start:head_end]

    def finish_publish(self) -> None:
        """Release-publish this rank's completed shard fanout."""
        self._validate_live()
        if not self.publishing_rows:
            raise RuntimeError("DCP query VMM finish_publish requires begin_publish.")
        self.published_rows = self.publishing_rows
        self.publishing_rows = 0
        with record_function("dcp.query_vmm.publish_fanout"):
            _publish_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
            )

    def acquire_local_query(self, rows: int) -> torch.Tensor:
        """Wait for every producer and expose the complete local query."""
        self._validate_live()
        if not self.published_rows:
            raise RuntimeError(
                "DCP query VMM acquire requires finish_publish on this call."
            )
        if self.consumer_rows:
            raise RuntimeError("DCP query VMM consumer read is already in progress.")
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                "DCP query VMM consumer row bound violated: "
                f"max_rows={self.max_rows}, requested={rows}."
            )
        if rows > self.published_rows:
            raise RuntimeError(
                "DCP query VMM consumer rows exceed the producer rows: "
                f"producer={self.published_rows}, consumer={rows}."
            )
        self.published_rows = 0
        self.consumer_rows = rows
        if rows not in _logged_consume_rows:
            _logged_consume_rows.add(rows)
            logger.info(
                "Executing CUDA VMM DCP query producer-fanout for decode rows=%d.",
                rows,
            )

        with record_function("dcp.query_vmm.acquire_local_fanout"):
            _wait_published_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )

        return self.local_consumer_query[:rows]

    def acknowledge(self) -> None:
        """Release the owner shards after every direct consumer has finished."""
        self._validate_live()
        if not self.consumer_rows:
            raise RuntimeError("DCP query VMM acknowledge has no active consumer read.")
        self.consumer_rows = 0
        with record_function("dcp.query_vmm.ack"):
            _ack_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
            )

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        self.peer_flags = None
        self.peer_consumer_queries = None
        self.local_consumer_query = None
        self.allocation.close()


def create_dcp_query_vmm_workspace_for_group(
    max_rows: int,
    local_heads: int,
    query_dim: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpQueryVmmWorkspace:
    """Collectively create one owner-local FP8 query workspace."""
    world_size = group.size()
    rank = group.rank()
    if world_size <= 1:
        raise RuntimeError("DCP query VMM requires dcp_world_size > 1.")
    if max_rows <= 0 or local_heads <= 0 or query_dim <= 0:
        raise ValueError(
            "DCP query VMM dimensions must be positive; "
            f"got max_rows={max_rows}, local_heads={local_heads}, "
            f"query_dim={query_dim}."
        )

    total_heads = world_size * local_heads
    consumer_query_bytes = max_rows * total_heads * query_dim
    allocation = create_rank_major_peer_view(
        (consumer_query_bytes + _SIGNAL_RESERVE_BYTES,),
        dtype=torch.uint8,
        group=group,
        require_native_atomics=True,
        device=device,
    )
    assert allocation.local_view is not None
    assert allocation.global_view is not None
    allocation.local_view.zero_()
    torch.accelerator.synchronize()
    dist.barrier(group=group)

    signal_offset = allocation.bytes_per_rank - _SIGNAL_BYTES
    if consumer_query_bytes > signal_offset:
        raise RuntimeError(
            "DCP query VMM allocation has no safe signal tail after the "
            f"consumer query: query_bytes={consumer_query_bytes}, "
            f"signal_offset={signal_offset}."
        )
    local_flags = allocation.local_view[
        signal_offset : signal_offset + _SIGNAL_BYTES
    ].view(torch.int64)
    peer_flags = make_rank_major_tensor_view(allocation, local_flags)
    local_consumer_query = (
        allocation.local_view[:consumer_query_bytes]
        .view(torch.float8_e4m3fn)
        .view(max_rows, total_heads, query_dim)
    )
    peer_consumer_queries = make_rank_major_tensor_view(
        allocation,
        local_consumer_query,
    )
    return DcpQueryVmmWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        local_heads=local_heads,
        query_dim=query_dim,
        group=group,
        device=device,
        allocation=allocation,
        local_consumer_query=local_consumer_query,
        peer_consumer_queries=peer_consumer_queries,
        peer_flags=peer_flags,
    )


_workspace: DcpQueryVmmWorkspace | None = None
_workspace_failed = False


def get_dcp_query_vmm_workspace(
    max_rows: int,
    local_heads: int,
    query_dim: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpQueryVmmWorkspace:
    """Create or fetch the singleton query workspace, refusing fallback."""
    global _workspace, _workspace_failed
    world_size = group.size()
    if _workspace_failed:
        raise RuntimeError("DCP query VMM workspace is unavailable.")
    if _workspace is not None:
        actual = (
            _workspace.max_rows,
            _workspace.local_heads,
            _workspace.query_dim,
            _workspace.world_size,
        )
        requested = (max_rows, local_heads, query_dim, world_size)
        requested_device = torch.device(device)
        if requested_device.index is None:
            requested_device = torch.device(
                f"cuda:{torch.accelerator.current_device_index()}"
            )
        if (
            actual != requested
            or _workspace.group is not group
            or _workspace.my_rank != group.rank()
            or _workspace.device != requested_device
        ):
            raise RuntimeError(
                "DCP query VMM workspace identity changed after initialization: "
                f"workspace_geometry={actual}, request_geometry={requested}, "
                f"workspace_rank={_workspace.my_rank}, request_rank={group.rank()}, "
                f"workspace_device={_workspace.device}, "
                f"request_device={requested_device}, "
                f"same_group={_workspace.group is group}."
            )
        return _workspace

    try:
        _workspace = create_dcp_query_vmm_workspace_for_group(
            max_rows,
            local_heads,
            query_dim,
            group,
            device,
        )
    except Exception as exc:
        _workspace_failed = True
        raise RuntimeError(
            "DCP query VMM workspace initialization failed; refusing to "
            "fall back to an explicit collective path."
        ) from exc
    logger.info_once(
        "Using CUDA VMM DCP query producer-fanout "
        "(max_rows=%d, local_heads=%d, total_heads=%d, query_dim=%d, "
        "physical_bytes_per_rank=%d).",
        _workspace.max_rows,
        _workspace.local_heads,
        _workspace.total_heads,
        _workspace.query_dim,
        _workspace.physical_bytes_per_rank,
    )
    return _workspace


def close_dcp_query_vmm_workspace() -> None:
    """Collectively close and reset the singleton workspace."""
    global _workspace, _workspace_failed
    if _workspace is not None:
        _workspace.close()
    _workspace = None
    _workspace_failed = False
