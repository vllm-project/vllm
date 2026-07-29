# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side DCP output/LSE merge over owner-local CUDA VMM storage.

Every rank publishes its local-KV attention output and LSE into its own physical
allocation. The consumer on rank ``r`` directly reads only query heads owned by
``r`` from every peer mapping, combines the shard LSEs, scales each partial
output, and writes the final local-head result. This replaces the decode-time
LSE AllGather and output ReduceScatter without materializing a gathered tensor.

The workspace is reusable and CUDA-graph safe. Device-side sequence counters
prevent an owner from overwriting a generation before every consumer has read
it. Initialization is fail-closed; callers must choose any intentional
large-row explicit path before invoking :meth:`merge`.
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

_HEADER_BYTES = 256
_WRITE_SEQ = tl.constexpr(0)
_READ_SEQ = tl.constexpr(1)
_MAX_FENCE_SPINS = 100_000_000
DEFAULT_MAX_ROWS = 128
_logged_merge_rows: set[int] = set()


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
def _consumer_merge_kernel(
    peer_outputs,
    peer_lses,
    merged_output,
    merged_lse,
    peer_output_rank_stride,
    peer_output_batch_stride,
    peer_output_head_stride,
    peer_output_dim_stride,
    peer_lse_rank_stride,
    peer_lse_batch_stride,
    peer_lse_head_stride,
    merged_output_batch_stride,
    merged_output_head_stride,
    merged_output_dim_stride,
    merged_lse_batch_stride,
    merged_lse_head_stride,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    is_base_e: tl.constexpr,
    block_dim: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    dim_block_idx = tl.program_id(2).to(tl.int64)
    global_head_idx = my_rank * local_heads + local_head_idx

    peer = tl.arange(0, world_size)
    lse_offsets = (
        peer * peer_lse_rank_stride
        + batch_idx * peer_lse_batch_stride
        + global_head_idx * peer_lse_head_stride
    )
    lse = tl.load(peer_lses + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    if is_base_e:
        weights = tl.exp(lse - lse_max)
        weight_sum = tl.sum(weights, axis=0)
        final_lse = tl.log(weight_sum) + lse_max
    else:
        weights = tl.exp2(lse - lse_max)
        weight_sum = tl.sum(weights, axis=0)
        final_lse = tl.log2(weight_sum) + lse_max
    weights = tl.where(weight_sum == 0.0, 0.0, weights / weight_sum)

    dim = dim_block_idx * block_dim + tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    output_offsets = (
        peer[:, None] * peer_output_rank_stride
        + batch_idx * peer_output_batch_stride
        + global_head_idx * peer_output_head_stride
        + dim[None, :] * peer_output_dim_stride
    )
    partial_output = tl.load(peer_outputs + output_offsets, mask=dim_mask[None, :])
    output = tl.sum(partial_output.to(tl.float32) * weights[:, None], axis=0)
    merged_output_offsets = (
        batch_idx * merged_output_batch_stride
        + local_head_idx * merged_output_head_stride
        + dim * merged_output_dim_stride
    )
    tl.store(merged_output + merged_output_offsets, output, mask=dim_mask)

    if dim_block_idx == 0:
        merged_lse_offset = (
            batch_idx * merged_lse_batch_stride
            + local_head_idx * merged_lse_head_stride
        )
        tl.store(merged_lse + merged_lse_offset, final_lse)


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
class DcpOutputVmmWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    total_heads: int
    head_dim: int
    group: ProcessGroup
    device: torch.device
    allocation: RankMajorPeerView
    local_partial_output: torch.Tensor
    local_partial_lse: torch.Tensor
    local_merged_output: torch.Tensor
    local_merged_lse: torch.Tensor
    peer_partial_outputs: torch.Tensor
    peer_partial_lses: torch.Tensor
    peer_flags: torch.Tensor

    @property
    def local_heads(self) -> int:
        return self.total_heads // self.world_size

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    @property
    def payload_bytes_per_rank(self) -> int:
        partial_output = (
            self.max_rows * self.total_heads * self.head_dim * torch.bfloat16.itemsize
        )
        partial_lse = self.max_rows * self.total_heads * torch.float32.itemsize
        merged_output = (
            self.max_rows * self.local_heads * self.head_dim * torch.bfloat16.itemsize
        )
        merged_lse = self.max_rows * self.local_heads * torch.float32.itemsize
        return partial_output + partial_lse + merged_output + merged_lse

    def merge(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        *,
        is_lse_base_on_e: bool,
        return_lse: bool = False,
        barrier_protected_reuse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.allocation.closed:
            raise RuntimeError("DCP output VMM workspace is closed.")
        current_device = torch.accelerator.current_device_index()
        if current_device != self.device.index:
            raise RuntimeError(
                "DCP output VMM current device changed after initialization: "
                f"workspace={self.device}, current=cuda:{current_device}."
            )
        rows = partial_output.shape[0]
        expected_output_shape = (rows, self.total_heads, self.head_dim)
        expected_lse_shape = (rows, self.total_heads)
        if tuple(partial_output.shape) != expected_output_shape:
            raise RuntimeError(
                "DCP output VMM partial-output shape mismatch: "
                f"expected {expected_output_shape}, got {tuple(partial_output.shape)}."
            )
        if tuple(partial_lse.shape) != expected_lse_shape:
            raise RuntimeError(
                "DCP output VMM LSE shape mismatch: "
                f"expected {expected_lse_shape}, got {tuple(partial_lse.shape)}."
            )
        if rows > self.max_rows:
            raise RuntimeError(
                f"DCP output VMM workspace has {self.max_rows} rows, requested {rows}."
            )
        if partial_output.dtype != torch.bfloat16:
            raise RuntimeError(
                "DCP output VMM currently requires BF16 attention output; "
                f"got {partial_output.dtype}."
            )
        if partial_lse.dtype != torch.float32:
            raise RuntimeError(
                f"DCP output VMM currently requires FP32 LSE; got {partial_lse.dtype}."
            )
        if partial_output.device != self.device or partial_lse.device != self.device:
            raise RuntimeError(
                "DCP output VMM inputs must be on the workspace device "
                f"{self.device}; got {partial_output.device} and {partial_lse.device}."
            )
        if not partial_output.is_contiguous() or not partial_lse.is_contiguous():
            raise RuntimeError("DCP output VMM inputs must be contiguous.")
        if rows not in _logged_merge_rows:
            _logged_merge_rows.add(rows)
            logger.info(
                "Executing owner-local CUDA VMM DCP output/LSE compute-gather "
                "for decode rows=%d.",
                rows,
            )

        if not barrier_protected_reuse:
            with record_function("dcp.output_lse.vmm.wait_reuse"):
                _wait_writable_kernel[(1,)](
                    self.peer_flags,
                    self.peer_flags.stride(0),
                    my_rank=self.my_rank,
                    world_size=self.world_size,
                    block_size=triton.next_power_of_2(self.world_size),
                    max_spins=_MAX_FENCE_SPINS,
                )
        with record_function("dcp.output_lse.vmm.publish_owner_and_acquire_peers"):
            self.local_partial_output[:rows].copy_(partial_output)
            self.local_partial_lse[:rows].copy_(partial_lse)
            _publish_and_wait_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )

        output = self.local_merged_output[:rows]
        lse = self.local_merged_lse[:rows]
        # GLM-5.2 uses D=512. Keep one consumer program per (row, local
        # head), avoiding duplicate peer-LSE work across two D tiles.
        block_dim = min(512, triton.next_power_of_2(self.head_dim))
        with record_function("dcp.output_lse.vmm.consumer_compute_gather"):
            _consumer_merge_kernel[
                (rows, self.local_heads, triton.cdiv(self.head_dim, block_dim))
            ](
                self.peer_partial_outputs,
                self.peer_partial_lses,
                output,
                lse,
                self.peer_partial_outputs.stride(0),
                self.peer_partial_outputs.stride(1),
                self.peer_partial_outputs.stride(2),
                self.peer_partial_outputs.stride(3),
                self.peer_partial_lses.stride(0),
                self.peer_partial_lses.stride(1),
                self.peer_partial_lses.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                local_heads=self.local_heads,
                head_dim=self.head_dim,
                my_rank=self.my_rank,
                world_size=self.world_size,
                is_base_e=is_lse_base_on_e,
                block_dim=block_dim,
            )
        if not barrier_protected_reuse:
            with record_function("dcp.output_lse.vmm.ack"):
                _ack_kernel[(1,)](
                    self.peer_flags,
                    self.peer_flags.stride(0),
                    my_rank=self.my_rank,
                )
        if return_lse:
            return output, lse
        return output

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        self.peer_flags = None
        self.peer_partial_lses = None
        self.peer_partial_outputs = None
        self.local_merged_lse = None
        self.local_merged_output = None
        self.local_partial_lse = None
        self.local_partial_output = None
        self.allocation.close()


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def create_dcp_output_vmm_workspace_for_group(
    max_rows: int,
    total_heads: int,
    head_dim: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpOutputVmmWorkspace:
    """Collectively create one owner-local output/LSE workspace."""
    world_size = group.size()
    rank = group.rank()
    if world_size <= 1:
        raise RuntimeError("DCP output VMM requires dcp_world_size > 1.")
    if total_heads % world_size:
        raise RuntimeError(
            f"total_heads={total_heads} is not divisible by world_size={world_size}."
        )
    if max_rows <= 0 or total_heads <= 0 or head_dim <= 0:
        raise ValueError(
            "DCP output VMM dimensions must be positive; "
            f"got max_rows={max_rows}, total_heads={total_heads}, head_dim={head_dim}."
        )

    local_heads = total_heads // world_size
    partial_output_bytes = max_rows * total_heads * head_dim * torch.bfloat16.itemsize
    partial_lse_bytes = max_rows * total_heads * torch.float32.itemsize
    merged_output_bytes = max_rows * local_heads * head_dim * torch.bfloat16.itemsize
    merged_lse_bytes = max_rows * local_heads * torch.float32.itemsize

    partial_output_offset = _HEADER_BYTES
    partial_lse_offset = _align_up(
        partial_output_offset + partial_output_bytes, torch.float32.itemsize
    )
    merged_output_offset = _align_up(
        partial_lse_offset + partial_lse_bytes, torch.bfloat16.itemsize
    )
    merged_lse_offset = _align_up(
        merged_output_offset + merged_output_bytes, torch.float32.itemsize
    )
    requested_bytes = merged_lse_offset + merged_lse_bytes

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
    local_partial_output = (
        allocation.local_view[
            partial_output_offset : partial_output_offset + partial_output_bytes
        ]
        .view(torch.bfloat16)
        .view(max_rows, total_heads, head_dim)
    )
    local_partial_lse = (
        allocation.local_view[
            partial_lse_offset : partial_lse_offset + partial_lse_bytes
        ]
        .view(torch.float32)
        .view(max_rows, total_heads)
    )
    local_merged_output = (
        allocation.local_view[
            merged_output_offset : merged_output_offset + merged_output_bytes
        ]
        .view(torch.bfloat16)
        .view(max_rows, local_heads, head_dim)
    )
    local_merged_lse = (
        allocation.local_view[merged_lse_offset : merged_lse_offset + merged_lse_bytes]
        .view(torch.float32)
        .view(max_rows, local_heads)
    )

    return DcpOutputVmmWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        total_heads=total_heads,
        head_dim=head_dim,
        group=group,
        device=device,
        allocation=allocation,
        local_partial_output=local_partial_output,
        local_partial_lse=local_partial_lse,
        local_merged_output=local_merged_output,
        local_merged_lse=local_merged_lse,
        peer_partial_outputs=make_rank_major_tensor_view(
            allocation, local_partial_output
        ),
        peer_partial_lses=make_rank_major_tensor_view(allocation, local_partial_lse),
        peer_flags=peer_flags,
    )


_workspaces: dict[int, DcpOutputVmmWorkspace] = {}
_workspace_failed = False


def get_dcp_output_vmm_workspace(
    max_rows: int,
    total_heads: int,
    head_dim: int,
    group: ProcessGroup,
    device: torch.device,
    workspace_slot: int = 0,
) -> DcpOutputVmmWorkspace:
    """Create or fetch one workspace slot, refusing all fallback."""
    global _workspace_failed
    world_size = group.size()
    if workspace_slot < 0:
        raise ValueError("DCP output VMM workspace slot must be nonnegative.")
    if _workspace_failed:
        raise RuntimeError("DCP output VMM workspace is unavailable.")
    workspace = _workspaces.get(workspace_slot)
    if workspace is not None:
        actual = (
            workspace.max_rows,
            workspace.total_heads,
            workspace.head_dim,
            workspace.world_size,
        )
        requested = (max_rows, total_heads, head_dim, world_size)
        requested_device = torch.device(device)
        if requested_device.index is None:
            requested_device = torch.device(
                f"cuda:{torch.accelerator.current_device_index()}"
            )
        if (
            actual != requested
            or workspace.group is not group
            or workspace.my_rank != group.rank()
            or workspace.device != requested_device
        ):
            raise RuntimeError(
                "DCP output VMM workspace identity changed after initialization: "
                f"workspace_slot={workspace_slot}, "
                f"workspace_geometry={actual}, request_geometry={requested}, "
                f"workspace_rank={workspace.my_rank}, request_rank={group.rank()}, "
                f"workspace_device={workspace.device}, "
                f"request_device={requested_device}, "
                f"same_group={workspace.group is group}."
            )
        return workspace

    try:
        workspace = create_dcp_output_vmm_workspace_for_group(
            max_rows,
            total_heads,
            head_dim,
            group,
            device,
        )
        _workspaces[workspace_slot] = workspace
    except Exception as exc:
        _workspace_failed = True
        raise RuntimeError(
            "DCP output VMM workspace initialization failed; refusing to "
            "fall back to an explicit collective path."
        ) from exc
    logger.info_once(
        "Using owner-local CUDA VMM DCP output/LSE compute-gather "
        "(workspace_slot=%d, max_rows=%d, total_heads=%d, head_dim=%d, "
        "physical_bytes_per_rank=%d).",
        workspace_slot,
        workspace.max_rows,
        workspace.total_heads,
        workspace.head_dim,
        workspace.physical_bytes_per_rank,
    )
    return workspace


def close_dcp_output_vmm_workspace() -> None:
    """Collectively close and reset all workspace slots."""
    global _workspace_failed
    for workspace in _workspaces.values():
        workspace.close()
    _workspaces.clear()
    _workspace_failed = False
