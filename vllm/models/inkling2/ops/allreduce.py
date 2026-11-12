# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MNNVL all-reduce fused with Inkling2's hidden-stream short conv and the
AttnRes mix + RMSNorm of the next sublayer.

Each rank pushes its bf16 partial rows (the attention output, or the MoE
expert rows combined in fp32 over each token's top-k slots and rounded once)
into every rank's Lamport buffer (one shot), or each row's channel shards to
their owner ranks, which reduce and broadcast them (two shots). Every row is
summed over the ranks in rank order in fp32 (identical bits on all ranks),
rounded once to bf16 and run through the short conv (conv-state cache write,
optional residual update) and the AttnRes mix + RMSNorm in the same kernel.
"""

import torch
import torch.distributed as dist

from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import round_up

_NUM_LAMPORT_BUFFERS = 3
_HIDDEN_SIZE = 4608


def _multicast_memory(size: int, group: dist.ProcessGroup):
    """FlashInfer's symmetric multicast memory at the minimum allocation and
    multicast granularities (its recommended multicast granularity rounds
    every allocation up to 512 MiB)."""
    from cuda.bindings import driver as cuda
    from flashinfer.comm.mnnvl import (
        SymmDeviceMemory,
        TorchDistBackend,
        checkCudaErrors,
    )

    class MinimumGranularityMemory(SymmDeviceMemory):
        def _get_allocation_prop(self, buf_size: int):
            allocation_prop, mc_prop = super()._get_allocation_prop(buf_size)
            granularity = checkCudaErrors(
                cuda.cuMemGetAllocationGranularity(
                    allocation_prop,
                    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            )
            mc_prop.size = round_up(buf_size + self.SIGNAL_PAD_SIZE, granularity)
            self._mc_granularity = checkCudaErrors(
                cuda.cuMulticastGetGranularity(
                    mc_prop,
                    cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM,
                )
            )
            granularity = max(granularity, self._mc_granularity)
            self.allocation_size = mc_prop.size = round_up(mc_prop.size, granularity)
            return allocation_prop, mc_prop

    return MinimumGranularityMemory(
        size,
        group.size(),
        dist.get_rank(group),
        torch.accelerator.current_device_index(),
        TorchDistBackend(group=group),
    )


class Inkling2AllReduceWorkspace:
    """Multicast Lamport buffers and generation flags of the fused kernel."""

    @classmethod
    def create(
        cls, group: dist.ProcessGroup, hidden_size: int
    ) -> "Inkling2AllReduceWorkspace | None":
        if (
            group.size() not in (2, 4, 8)
            or hidden_size != _HIDDEN_SIZE
            or not hasattr(torch.ops._C, "inkling2_allreduce_conv_mix")
            or not has_flashinfer()
        ):
            return None
        from flashinfer.comm.mnnvl import (
            TorchDistBackend,
            all_ranks_agree,
            is_multicast_supported,
        )

        device = torch.accelerator.current_device_index()
        supported = torch.cuda.get_device_capability(device)[0] == 10
        supported = supported and is_multicast_supported(device)
        if not all_ranks_agree(TorchDistBackend(group=group), supported):
            return None
        return cls(group, hidden_size)

    def __init__(self, group: dist.ProcessGroup, hidden_size: int) -> None:
        self.world_size = group.size()
        self.rank = dist.get_rank(group)
        # The (attention, MoE) batches whose all-reduce runs fused: in one
        # shot up to the first bound, in two shots in the token range.
        if self.world_size <= 4:
            self.one_shot_max_tokens = (64, 96)
            self.two_shot_tokens = ((65, 128), (97, 128))
        else:
            self.one_shot_max_tokens = (32, 32)
            self.two_shot_tokens = ((33, 128), (33, 128))
        # One shot stages every rank's bf16 rows; each of the two two-shot
        # stages holds every row once.
        token_bytes = hidden_size * 2
        buffer_bytes = max(
            max(self.one_shot_max_tokens) * self.world_size * token_bytes,
            2 * max(hi for _, hi in self.two_shot_tokens) * token_bytes,
        )
        self.buffer_bytes = buffer_bytes = round_up(buffer_bytes, 32)
        self.token_bytes = token_bytes
        # FlashInfer only maps the multicast memory; the Lamport protocol and
        # its flags are ours.
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.buffer = _multicast_memory(_NUM_LAMPORT_BUFFERS * buffer_bytes, group)
        self.multicast_ptr = self.buffer.get_multicast_ptr()
        if not self.multicast_ptr:
            raise RuntimeError("Inkling2 all-reduce requires NVLink multicast")
        self.local_ptr = self.buffer.get_unicast_ptr(self.rank)
        self.buffer_ptrs = self.buffer.get_buffer_ptrs_dev()
        self.buffer.lamport_initialize(self.rank, torch.float32)
        # [current, dirty, bytes per buffer, dirty stages, bytes to clear x4,
        # arrival count]
        self.flags = torch.tensor(
            [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=device,
        )
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def max_tokens(self, two_shot: bool) -> int:
        rows = 2 if two_shot else self.world_size
        return self.buffer_bytes // (rows * self.token_bytes)

    def plan(self, num_tokens: int) -> list[tuple[bool, bool]]:
        """Whether each sublayer's (attention, MoE) all-reduce of
        `num_tokens` runs fused, and whether in two shots."""
        return [
            (num_tokens <= one or lo <= num_tokens <= hi, num_tokens > one)
            for one, (lo, hi) in zip(self.one_shot_max_tokens, self.two_shot_tokens)
        ]


def all_reduce_conv_mix(
    workspace: Inkling2AllReduceWorkspace,
    x: torch.Tensor | None,
    combine: tuple[torch.Tensor, torch.Tensor] | None,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    eps: float = 1e-6,
    two_shot: bool = False,
) -> torch.Tensor:
    """All-reduce this rank's partial sum over TP, then `short_conv` it into
    `conv_out` (with the optional `(residual + addend) + y` update), then
    `attn_res` over `[*stack, (last +) conv_out]` with each query.

    The partial sum is the bf16 `x`, or `combine = (down, src2dst)`: each
    token's top-k expert rows of `down` summed in order in fp32 and rounded
    once to bf16. Returns the `(Q, T, D)` bf16 mix outputs.
    """
    num_tokens, d = conv_out.shape
    assert num_tokens > 0 and conv_out.is_contiguous()
    assert num_tokens <= workspace.max_tokens(two_shot)
    if x is not None:
        assert x.is_contiguous() and x.shape == (num_tokens, d)
        assert x.dtype == torch.bfloat16
    else:
        assert combine is not None
        down, src2dst = combine
        assert down.dtype == torch.bfloat16 and down.is_contiguous()
        assert src2dst.dtype == torch.int32 and src2dst.numel() % num_tokens == 0
    for t in (residual, addend, last):
        assert t is None or (t.is_contiguous() and t.shape == conv_out.shape)
    assert residual is None or addend is not None
    assert stack.stride(1) == d and stack.stride(2) == 1
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    assert conv_weight.shape == (d, 4) and conv_weight.is_contiguous()
    assert d == cache.shape[1] * stream_width and cache.stride(3) == 1
    assert positions.dtype == slot_mapping.dtype == torch.int64
    out = conv_out.new_empty((queries.shape[0], num_tokens, d))
    torch.ops._C.inkling2_allreduce_conv_mix(
        x,
        None if combine is None else combine[0],
        None if combine is None else combine[1],
        two_shot,
        workspace.multicast_ptr,
        workspace.local_ptr,
        workspace.buffer_ptrs,
        workspace.flags,
        workspace.rank,
        workspace.world_size,
        cache,
        conv_weight,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        stream_offset,
        stream_width,
        5**-0.5,
        residual,
        addend,
        conv_out,
        stack,
        last,
        queries,
        norm_weight,
        eps,
        norm_eps,
        out,
    )
    return out
