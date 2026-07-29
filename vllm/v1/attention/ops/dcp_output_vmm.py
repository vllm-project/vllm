# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-direct DCP output/LSE merge over owner-local CUDA VMM storage.

Every rank publishes each destination's head slice directly into that owner's
physical allocation. The consumer combines only local receive storage after
acquiring one generation signal from every producer. This replaces the
decode-time LSE AllGather and output ReduceScatter without materializing a
gathered tensor.

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
_MAX_FENCE_SPINS = 100_000_000
DEFAULT_MAX_ROWS = 512
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
def _direct_publish_kernel(
    partial_output,
    partial_lse,
    peer_outputs,
    peer_lses,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_token_stride,
    lse_head_stride,
    peer_output_dest_stride,
    peer_output_source_stride,
    peer_output_token_stride,
    peer_output_head_stride,
    peer_output_dim_stride,
    peer_lse_dest_stride,
    peer_lse_source_stride,
    peer_lse_token_stride,
    peer_lse_head_stride,
    my_rank: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_items: tl.constexpr,
    head_block_size: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    destination_rank = tl.program_id(1).to(tl.int64)
    item = tl.arange(0, block_items)
    item_mask = item < local_heads * head_dim
    local_head_idx = item // head_dim
    dim = item % head_dim
    source_head_idx = destination_rank * local_heads + local_head_idx

    source_output_offset = (
        token_idx * output_token_stride
        + source_head_idx * output_head_stride
        + dim * output_dim_stride
    )
    destination_output_offset = (
        destination_rank * peer_output_dest_stride
        + my_rank * peer_output_source_stride
        + token_idx * peer_output_token_stride
        + local_head_idx * peer_output_head_stride
        + dim * peer_output_dim_stride
    )
    value = tl.load(partial_output + source_output_offset, mask=item_mask)
    tl.store(peer_outputs + destination_output_offset, value, mask=item_mask)

    lse_local_head_idx = tl.arange(0, head_block_size)
    lse_mask = lse_local_head_idx < local_heads
    lse_source_head_idx = destination_rank * local_heads + lse_local_head_idx
    source_lse_offset = (
        token_idx * lse_token_stride + lse_source_head_idx * lse_head_stride
    )
    destination_lse_offset = (
        destination_rank * peer_lse_dest_stride
        + my_rank * peer_lse_source_stride
        + token_idx * peer_lse_token_stride
        + lse_local_head_idx * peer_lse_head_stride
    )
    tl.store(
        peer_lses + destination_lse_offset,
        tl.load(partial_lse + source_lse_offset, mask=lse_mask),
        mask=lse_mask,
    )


@triton.jit
def _direct_signal_kernel(
    local_epoch,
    peer_signals,
    peer_signal_dest_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
):
    epoch = tl.atomic_add(local_epoch, 1, sem="acq_rel", scope="gpu") + 1
    destination_rank = tl.arange(0, block_size)
    mask = destination_rank < world_size
    tl.atomic_xchg(
        peer_signals + destination_rank * peer_signal_dest_stride + my_rank,
        epoch,
        mask=mask,
        sem="release",
        scope="sys",
    )


@triton.jit
def _direct_consumer_merge_kernel(
    local_outputs,
    local_lses,
    local_signals,
    local_epoch,
    merged_output,
    merged_lse,
    local_output_source_stride,
    local_output_token_stride,
    local_output_head_stride,
    local_output_dim_stride,
    local_lse_source_stride,
    local_lse_token_stride,
    local_lse_head_stride,
    merged_output_token_stride,
    merged_output_head_stride,
    merged_output_dim_stride,
    merged_lse_token_stride,
    merged_lse_head_stride,
    world_size: tl.constexpr,
    is_base_e: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    signal_block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    expected_epoch = tl.atomic_add(
        local_epoch,
        0,
        sem="acquire",
        scope="gpu",
    )
    signal_source = tl.arange(0, signal_block_size)
    signal_mask = signal_source < world_size
    observed = tl.atomic_add(
        local_signals + signal_source,
        0,
        mask=signal_mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(signal_mask & (observed < expected_epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            local_signals + signal_source,
            0,
            mask=signal_mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(signal_mask & (observed < expected_epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)

    source_rank = tl.arange(0, world_size)
    lse_offset = (
        source_rank * local_lse_source_stride
        + token_idx * local_lse_token_stride
        + local_head_idx * local_lse_head_stride
    )
    lse = tl.load(local_lses + lse_offset)
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

    dim = tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    output_offset = (
        source_rank[:, None] * local_output_source_stride
        + token_idx * local_output_token_stride
        + local_head_idx * local_output_head_stride
        + dim[None, :] * local_output_dim_stride
    )
    partial_output = tl.load(local_outputs + output_offset, mask=dim_mask[None, :])
    output = tl.sum(partial_output.to(tl.float32) * weights[:, None], axis=0)
    merged_output_offset = (
        token_idx * merged_output_token_stride
        + local_head_idx * merged_output_head_stride
        + dim * merged_output_dim_stride
    )
    tl.store(merged_output + merged_output_offset, output, mask=dim_mask)
    merged_lse_offset = (
        token_idx * merged_lse_token_stride + local_head_idx * merged_lse_head_stride
    )
    tl.store(merged_lse + merged_lse_offset, final_lse)


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
    local_epoch: torch.Tensor

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

        block_dim = min(512, triton.next_power_of_2(self.head_dim))
        with record_function("dcp.output_lse.vmm.producer_direct_publish"):
            _direct_publish_kernel[(rows, self.world_size)](
                partial_output,
                partial_lse,
                self.peer_partial_outputs,
                self.peer_partial_lses,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_lse.stride(0),
                partial_lse.stride(1),
                self.peer_partial_outputs.stride(0),
                self.peer_partial_outputs.stride(1),
                self.peer_partial_outputs.stride(2),
                self.peer_partial_outputs.stride(3),
                self.peer_partial_outputs.stride(4),
                self.peer_partial_lses.stride(0),
                self.peer_partial_lses.stride(1),
                self.peer_partial_lses.stride(2),
                self.peer_partial_lses.stride(3),
                my_rank=self.my_rank,
                local_heads=self.local_heads,
                head_dim=self.head_dim,
                block_items=triton.next_power_of_2(self.local_heads * self.head_dim),
                head_block_size=triton.next_power_of_2(self.local_heads),
                num_warps=8,
            )
            _direct_signal_kernel[(1,)](
                self.local_epoch,
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
            )

        output = self.local_merged_output[:rows]
        lse = self.local_merged_lse[:rows]
        with record_function("dcp.output_lse.vmm.consumer_compute_gather"):
            _direct_consumer_merge_kernel[(rows, self.local_heads)](
                self.local_partial_output,
                self.local_partial_lse,
                self.peer_flags[self.my_rank],
                self.local_epoch,
                output,
                lse,
                self.local_partial_output.stride(0),
                self.local_partial_output.stride(1),
                self.local_partial_output.stride(2),
                self.local_partial_output.stride(3),
                self.local_partial_lse.stride(0),
                self.local_partial_lse.stride(1),
                self.local_partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                world_size=self.world_size,
                is_base_e=is_lse_base_on_e,
                head_dim=self.head_dim,
                block_dim=block_dim,
                signal_block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
                num_warps=4,
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
        self.local_epoch = None
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

    local_flags = allocation.local_view[: world_size * 4].view(torch.int32)
    peer_flags = make_rank_major_tensor_view(allocation, local_flags)
    local_partial_output = (
        allocation.local_view[
            partial_output_offset : partial_output_offset + partial_output_bytes
        ]
        .view(torch.bfloat16)
        .view(world_size, max_rows, local_heads, head_dim)
    )
    local_partial_lse = (
        allocation.local_view[
            partial_lse_offset : partial_lse_offset + partial_lse_bytes
        ]
        .view(torch.float32)
        .view(world_size, max_rows, local_heads)
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
        local_epoch=torch.zeros(1, dtype=torch.int32, device=device),
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
