# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deadlock-free fused RS + short-conv + AG + residual + RMSNorm.

The public integration surface is ``LamportRSConv.rs_sconv_ag_add_norm``.

Liveness
--------
Large grids are deliberately split at the two communication dependencies:

  1. ``_publish_input_kernel`` only publishes rank partials.
  2. ``_reduce_insert_kernel`` waits, reduces, and inserts the local shard.
  3. ``_sconv_publish_kernel`` only computes and publishes the local result.
  4. ``_gather_norm_kernel`` waits, gathers, and normalizes.

Without PDL, CUDA stream order completes a producer before its consumer.  With
PDL, every producer CTA posts all peer stores before triggering its dependent,
and the consumer executes ``gdc_wait`` before polling.  A consumer therefore
waits only for stores from a producer that is already running (or complete) on
another GPU.  It never waits for another block in its own grid.  Consequently
spinning consumers cannot occupy resources needed by any producer, and the
wait-for graph has no cycle.  The proof is independent of grid size, block
dispatch order, and occupancy.

For one token, the first three phases use eight independent channel slices to
expose enough CTA parallelism for decode latency.  Each slice has exclusive
ownership of its cache and Lamport columns; the gather/RMSNorm phase retains
one CTA per token so no cross-CTA reduction or completion counter is needed.

Immediate buffer reuse (including replay of a captured CUDA graph) is also
safe.  Rank R cannot republish input for call n+1 until its gather for n has
finished; that gather waited for owner O's output, which O publishes only
after consuming R's input for n.  Likewise, R cannot republish output for n+1
until its reduction for n+1 has observed destination D's input; D publishes
that input only after its gather consumed R's output for n.  Thus every prior
read happens-before a same-slot rewrite.  Three generations reduce incidental
coupling for ordinary launches, but correctness does not depend on rotation.

The payload itself is the Lamport flag.  A 32-bit store publishes two bf16s
atomically; 0x80008000 (two negative zeroes) denotes an empty pair.  Real
negative zeroes are changed to positive zero before publication.  Consumers
use volatile 32-bit loads and restore the sentinel after consuming a slot.
"""

from __future__ import annotations

import os

import torch

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

_MAX_TOKENS = 16384
_EMPTY_PAIR = tl.constexpr(0x80008000) if HAS_TRITON else 0x80008000


@triton.jit
def _pack_bf16_pairs(values):
    """Pack bf16 values into atomic u32 pairs and reserve negative zero."""
    lo, hi = tl.split(values.reshape([values.shape[0] // 2, 2]))
    lo = lo.to(tl.uint16, bitcast=True)
    hi = hi.to(tl.uint16, bitcast=True)
    lo = tl.where(lo == 0x8000, 0, lo).to(tl.uint32)
    hi = tl.where(hi == 0x8000, 0, hi).to(tl.uint32)
    return lo | (hi << 16)


@triton.jit
def _unpack_bf16_pairs(values):
    lo = (values & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (values >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.interleave(lo, hi)


@triton.jit
def _wait_pairs(ptr, offsets, mask):
    values = tl.load(ptr + offsets, mask=mask, other=0, volatile=True)
    while tl.max(tl.where(mask & (values == _EMPTY_PAIR), 1, 0)) != 0:
        values = tl.load(ptr + offsets, mask=mask, other=0, volatile=True)
    return values


@triton.jit
def _publish_input_kernel(
    stage_ptr,
    peer_ptrs,
    peer_offset_u32,
    stride_stage_t,
    C: tl.constexpr,
    CS: tl.constexpr,
    CS_P2: tl.constexpr,
    SPLITS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Publish this rank's full partial row into every shard owner's slots."""
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = CS // SPLITS
    pair = tl.arange(0, CS_P2 // 2)
    pair_mask = pair < CSS // 2
    elem = tl.arange(0, CS_P2)
    elem_mask = elem < CSS
    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    # Address generation is independent of the preceding stream kernel.  The
    # acquire remains before stage/peer loads, which may consume its writes.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    for owner in tl.static_range(WORLD):
        values = tl.load(
            stage_ptr + token * stride_stage_t + owner * CS + split * CSS + elem,
            mask=elem_mask,
            other=0.0,
        )
        packed = _pack_bf16_pairs(values)
        base = tl.load(ptrs + owner).to(tl.pointer_type(tl.uint32))
        dst = (token * WORLD + RANK) * (CS // 2) + split * (CSS // 2) + pair
        tl.store(base + peer_offset_u32 + dst, packed, mask=pair_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _reduce_insert_kernel(
    input_peer_ptrs,
    input_peer_offset_u32,
    cache_ptr,
    slot_ptr,
    stride_cache_block,
    stride_cache_head,
    stride_cache_token,
    stride_cache_dim,
    block_size,
    C: tl.constexpr,
    CS: tl.constexpr,
    CS_P2: tl.constexpr,
    SPLITS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    CACHE_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Consume all partials for this rank and insert the reduced cache row."""
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = CS // SPLITS
    source = tl.arange(0, WORLD)
    pair = tl.arange(0, CS_P2 // 2)
    pair_mask = pair < CSS // 2
    offsets = (
        (token * WORLD + source)[:, None] * (CS // 2)
        + split * (CSS // 2)
        + pair[None, :]
    )
    mask = tl.full([WORLD], True, tl.int1)[:, None] & pair_mask[None, :]
    input_ptrs = input_peer_ptrs.to(tl.pointer_type(tl.uint64))
    input_u32 = tl.load(input_ptrs + RANK).to(tl.pointer_type(tl.uint32))
    input_u32 += input_peer_offset_u32
    # Slot metadata and the cache destination do not depend on input publish.
    slot = tl.load(slot_ptr + token)
    valid = slot >= 0
    channel = tl.arange(0, CS_P2)
    channel_mask = channel < CSS
    global_channel = split * CSS + channel
    head = tl.minimum(global_channel // HEAD_SIZE, CS // HEAD_SIZE - 1)
    dim = CACHE_OFFSET + global_channel % HEAD_SIZE
    safe_slot = tl.maximum(slot, 0).to(tl.int64)
    dst = (
        cache_ptr
        + (safe_slot // block_size) * stride_cache_block
        + head * stride_cache_head
        + (safe_slot % block_size) * stride_cache_token
        + dim * stride_cache_dim
    )
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    packed = _wait_pairs(input_u32, offsets, mask)

    lo = (packed & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    reduced = tl.interleave(
        tl.sum(lo.to(tl.float32), axis=0).to(tl.bfloat16),
        tl.sum(hi.to(tl.float32), axis=0).to(tl.bfloat16),
    )
    tl.store(
        input_u32 + offsets,
        tl.full([WORLD, CS_P2 // 2], _EMPTY_PAIR, tl.uint32),
        mask=mask,
    )

    tl.store(dst, reduced, mask=valid & channel_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _sconv_publish_kernel(
    peer_ptrs,
    peer_offset_u32,
    residual_ptr,
    weight_ptr,
    cache_ptr,
    position_ptr,
    sequence_ptr,
    slot_ptr,
    block_table_ptr,
    stride_residual_t,
    stride_cache_block,
    stride_cache_head,
    stride_cache_token,
    stride_cache_dim,
    stride_block_table_r,
    max_blocks,
    block_size,
    C: tl.constexpr,
    CS: tl.constexpr,
    CS_P2: tl.constexpr,
    SPLITS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    CACHE_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    WINDOW: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Compute this rank's shard, then publish it to every rank."""
    tl.static_assert(WINDOW == 4)
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = CS // SPLITS
    channel = tl.arange(0, CS_P2)
    channel_mask = channel < CSS
    global_channel = split * CSS + channel
    head = tl.minimum(global_channel // HEAD_SIZE, CS // HEAD_SIZE - 1)
    dim = CACHE_OFFSET + global_channel % HEAD_SIZE
    slot = tl.load(slot_ptr + token)
    valid = slot >= 0
    position = tl.load(position_ptr + token)
    sequence = tl.load(sequence_ptr + token)
    safe_slot = tl.maximum(slot, 0).to(tl.int64)
    own_ptr = (
        cache_ptr
        + (safe_slot // block_size) * stride_cache_block
        + head * stride_cache_head
        + (safe_slot % block_size) * stride_cache_token
        + dim * stride_cache_dim
    )
    # These loads were made visible before reduce/insert could signal us and
    # are independent of its cache store.  Hoisting them gives PDL useful work
    # to overlap while retaining the acquire before every cache read.
    residual = tl.load(
        residual_ptr + token * stride_residual_t + RANK * CS + global_channel,
        mask=channel_mask,
        other=0.0,
    )
    weight0 = tl.load(
        weight_ptr + global_channel * WINDOW,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight1 = tl.load(
        weight_ptr + global_channel * WINDOW + 1,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight2 = tl.load(
        weight_ptr + global_channel * WINDOW + 2,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight3 = tl.load(
        weight_ptr + global_channel * WINDOW + 3,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    current = tl.load(own_ptr, mask=valid & channel_mask, other=0.0)

    conv = tl.zeros([CS_P2], tl.float32)
    for tap_idx in tl.static_range(WINDOW):
        source_position = position - (WINDOW - 1) + tap_idx
        take = valid & (source_position >= 0)
        if tap_idx == WINDOW - 1:
            value = tl.where(take, current.to(tl.float32), 0.0)
        else:
            safe_position = tl.maximum(source_position, 0)
            logical_block = tl.minimum(safe_position // block_size, max_blocks - 1)
            physical_block = tl.load(
                block_table_ptr + sequence * stride_block_table_r + logical_block,
                mask=take,
                other=0,
            ).to(tl.int64)
            source_ptr = (
                cache_ptr
                + physical_block * stride_cache_block
                + head * stride_cache_head
                + (safe_position % block_size) * stride_cache_token
                + dim * stride_cache_dim
            )
            cached = tl.load(source_ptr, mask=take & channel_mask, other=0.0)
            value = tl.where(take, cached.to(tl.float32), 0.0)
        if tap_idx == 0:
            weight = weight0
        elif tap_idx == 1:
            weight = weight1
        elif tap_idx == 2:
            weight = weight2
        else:
            weight = weight3
        conv += value * weight

    # Preserve both bf16 rounding points of the original sublayer.
    short_conv_with_skip = (conv + current.to(tl.float32)).to(tl.bfloat16)
    output = (residual.to(tl.float32) + short_conv_with_skip.to(tl.float32)).to(
        tl.bfloat16
    )

    pair = tl.arange(0, CS_P2 // 2)
    pair_mask = pair < CSS // 2
    packed = _pack_bf16_pairs(output)
    row_offset = token * (C // 2) + RANK * (CS // 2) + split * (CSS // 2) + pair

    for destination in tl.static_range(WORLD):
        base = tl.load(ptrs + destination).to(tl.pointer_type(tl.uint32))
        tl.store(base + peer_offset_u32 + row_offset, packed, mask=pair_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _gather_norm_kernel(
    output_peer_ptrs,
    output_peer_offset_u32,
    norm_weight_ptr,
    normed_ptr,
    residual_out_ptr,
    eps,
    stride_output_t,
    C: tl.constexpr,
    C_P2: tl.constexpr,
    RANK: tl.constexpr,
    HAS_NORM: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Consume a complete row; one CTA owns all outputs for one token."""
    token = tl.program_id(0).to(tl.int64)
    pair = tl.arange(0, C_P2 // 2)
    pair_mask = pair < C // 2
    output_ptrs = output_peer_ptrs.to(tl.pointer_type(tl.uint64))
    output_u32 = tl.load(output_ptrs + RANK).to(tl.pointer_type(tl.uint32))
    output_u32 += output_peer_offset_u32
    offsets = token * (C // 2) + pair
    channel = tl.arange(0, C_P2)
    channel_mask = channel < C
    # Gamma is independent of the preceding sconv publication.  Keep the
    # acquire immediately before polling the Lamport output slots.
    if HAS_NORM:
        weight = tl.load(norm_weight_ptr + channel, mask=channel_mask, other=0.0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    packed = _wait_pairs(output_u32, offsets, pair_mask)
    row = _unpack_bf16_pairs(packed)
    tl.store(
        residual_out_ptr + token * stride_output_t + channel, row, mask=channel_mask
    )
    if HAS_NORM:
        row_f32 = tl.where(channel_mask, row.to(tl.float32), 0.0)
        inv_rms = tl.rsqrt(tl.sum(row_f32 * row_f32, axis=0) / C + eps)
        tl.store(
            normed_ptr + token * stride_output_t + channel,
            (row_f32 * inv_rms * weight.to(tl.float32)).to(tl.bfloat16),
            mask=channel_mask,
        )
    tl.store(
        output_u32 + offsets,
        tl.full([C_P2 // 2], _EMPTY_PAIR, tl.uint32),
        mask=pair_mask,
    )
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _validate_lamport_init_kernel(
    input_peer_ptrs,
    output_peer_ptrs,
    bad_ptr,
    num_pairs,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Validate both complete local allocations through their fabric pointers."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_pairs
    ptrs_in = input_peer_ptrs.to(tl.pointer_type(tl.uint64))
    ptrs_out = output_peer_ptrs.to(tl.pointer_type(tl.uint64))
    local_in = tl.load(ptrs_in + RANK).to(tl.pointer_type(tl.uint32))
    local_out = tl.load(ptrs_out + RANK).to(tl.pointer_type(tl.uint32))
    value_in = tl.load(local_in + offsets, mask=mask, other=_EMPTY_PAIR)
    value_out = tl.load(local_out + offsets, mask=mask, other=_EMPTY_PAIR)
    bad = tl.max(
        tl.where(mask & ((value_in != _EMPTY_PAIR) | (value_out != _EMPTY_PAIR)), 1, 0)
    )
    if bad != 0:
        tl.atomic_max(bad_ptr, 1)


class LamportRSConv:
    """Persistent symmetric buffers for one TP group."""

    def _initialize_lamport_buffers(self) -> None:
        """Arm every slot and collectively verify the exact sentinel bits.

        Do not replace this with a zero-fill: Lamport readiness distinguishes
        the bf16 bit pattern for -0.0 (0x8000) from every published payload.
        The validation is deliberately collective so one rank cannot enter the
        first polling kernel while another rank still has an unarmed buffer.
        """
        if not self.buf_in.is_contiguous() or not self.buf_out.is_contiguous():
            raise RuntimeError("Lamport symmetric buffers must be contiguous")
        if self.buf_in.numel() % 2 or self.buf_out.numel() % 2:
            raise RuntimeError("Lamport buffers must contain whole uint32 pairs")

        # Fill through int16 so each bf16 lane receives the exact -0.0 bits.
        # This covers all three generations and all max-token slots, including
        # slots that a smaller first invocation does not touch.
        self.buf_in.view(torch.int16).fill_(-0x8000)
        self.buf_out.view(torch.int16).fill_(-0x8000)
        torch.accelerator.synchronize(self.device)
        self.tp.barrier()

        # Scan the complete local allocations.  Since every rank performs the
        # scan and participates in MAX, success proves every symmetric backing
        # allocation was armed before any rank is allowed to use generation 0.
        bad = torch.logical_or(
            self.buf_in.view(torch.int16).ne(-0x8000).any(),
            self.buf_out.view(torch.int16).ne(-0x8000).any(),
        ).to(dtype=torch.float32)
        bad = self.tp.all_reduce(bad)
        if int(bad.item()) != 0:
            raise RuntimeError("Lamport sentinel initialization failed on a TP rank")
        self.tp.barrier()

    def _initialize_mnnvl_buffers(self) -> None:
        """Initialize and validate FlashInfer fabric-mapped Lamport storage."""
        self._mnnvl_input_handle.lamport_initialize(self.rank, torch.bfloat16)
        self._mnnvl_output_handle.lamport_initialize(self.rank, torch.bfloat16)
        torch.accelerator.synchronize(self.device)
        self.tp.barrier()

        bad = torch.zeros((), dtype=torch.int32, device=self.device)
        num_pairs = self.num_buffers * self.max_tokens * self.hidden_size // 2
        _validate_lamport_init_kernel[(triton.cdiv(num_pairs, 256),)](
            self.input_peer_ptrs,
            self.output_peer_ptrs,
            bad,
            num_pairs,
            RANK=self.rank,
            BLOCK=256,
            num_warps=4,
        )
        bad = self.tp.all_reduce(bad.to(torch.float32))
        if int(bad.item()) != 0:
            raise RuntimeError("MNNVL Lamport sentinel initialization failed")
        self.tp.barrier()

    def __init__(
        self, hidden_size: int, window_size: int, max_tokens: int = _MAX_TOKENS
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        tp = get_tp_group()
        self.tp = tp
        self.group = tp.device_group
        self.world_size = tp.world_size
        self.rank = tp.rank_in_group
        self.device = torch.device(tp.device)
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"TP world size must be 2, 4, or 8, got {self.world_size}")
        if hidden_size % (2 * self.world_size) != 0:
            raise ValueError("hidden size must produce an even shard on every rank")
        if window_size != 4:
            raise ValueError(f"short-conv window size must be 4, got {window_size}")
        if max_tokens < 1 or max_tokens > _MAX_TOKENS:
            raise ValueError(f"max_tokens must be in [1, {_MAX_TOKENS}]")

        is_cross_node = not all(in_the_same_node_as(tp.cpu_group))
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.shard_size = hidden_size // self.world_size
        # Three generations follow FlashInfer's Lamport layout.  A generation
        # is reused only after two intervening collective calls.
        self.num_buffers = 3
        self.input_generation_bytes = max_tokens * hidden_size * 2
        self.output_generation_bytes = max_tokens * hidden_size * 2
        self.use_pdl = torch.cuda.get_device_capability(self.device)[0] >= 9
        if is_cross_node:
            try:
                from flashinfer.comm.mnnvl import (
                    McastGPUBuffer,
                    TorchDistBackend,
                    is_mnnvl_fabric_supported,
                )
            except ImportError as error:
                raise RuntimeError(
                    "cross-node TP requires FlashInfer MNNVL support"
                ) from error

            local_supported = int(
                is_mnnvl_fabric_supported(torch.accelerator.current_device_index())
            )
            unsupported = torch.tensor(
                1 - local_supported, dtype=torch.float32, device=self.device
            )
            unsupported = tp.all_reduce(unsupported)
            if int(unsupported.item()) != 0:
                raise RuntimeError("cross-node TP is supported only on MNNVL fabric")

            comm_backend = TorchDistBackend(self.group)
            allocation_bytes = self.num_buffers * max_tokens * hidden_size * 2
            self._mnnvl_input_handle = McastGPUBuffer(
                allocation_bytes,
                self.world_size,
                self.rank,
                self.device,
                comm_backend,
            )
            self._mnnvl_output_handle = McastGPUBuffer(
                allocation_bytes,
                self.world_size,
                self.rank,
                self.device,
                comm_backend,
            )
            self.input_peer_ptrs = self._mnnvl_input_handle.get_buffer_ptrs_dev()
            self.output_peer_ptrs = self._mnnvl_output_handle.get_buffer_ptrs_dev()
            self._initialize_mnnvl_buffers()
            logger.info("using FlashInfer fabric-mapped MNNVL Lamport buffers")
        else:
            self.buf_in = symm_mem.empty(
                self.num_buffers,
                max_tokens,
                self.world_size,
                self.shard_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.buf_out = symm_mem.empty(
                self.num_buffers,
                max_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            group_name = self.group.group_name
            input_handle = symm_mem.rendezvous(self.buf_in, group_name)
            output_handle = symm_mem.rendezvous(self.buf_out, group_name)
            self.input_peer_ptrs = input_handle.buffer_ptrs_dev
            self.output_peer_ptrs = output_handle.buffer_ptrs_dev
            self._input_handle = input_handle
            self._output_handle = output_handle
            self._initialize_lamport_buffers()
        self.generation = 0

    def usable(self, num_tokens: int) -> bool:
        return 0 < num_tokens <= self.max_tokens

    def rs_sconv_ag_add_norm(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        conv_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float,
        cache: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        seq_idx: torch.Tensor,
        slot_mapping: torch.Tensor,
        off_s: int,
        ws: int,
        block_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ``(normed | None, new_residual)``, both shaped ``[T, 6144]``."""
        tokens, hidden_size = residual.shape
        if not self.usable(tokens):
            raise ValueError(f"num_tokens must be in [1, {self.max_tokens}]")
        if hidden_size != self.hidden_size or residual.dtype != torch.bfloat16:
            raise ValueError("residual must be bf16 [T, 6144]")
        if (
            input_tensor.shape != residual.shape
            or input_tensor.dtype != torch.bfloat16
            or input_tensor.stride(1) != 1
        ):
            raise ValueError("input_tensor must be channel-contiguous bf16 [T, 6144]")
        shard_size = hidden_size // self.world_size
        if conv_weight.shape != (shard_size, self.window_size):
            raise ValueError(
                f"conv_weight must have shape [{shard_size}, {self.window_size}]"
            )
        if (
            conv_weight.dtype != torch.bfloat16
            or conv_weight.stride(0) != self.window_size
        ):
            raise ValueError("conv_weight must be contiguous bf16")
        if norm_weight is not None and (
            norm_weight.shape != (hidden_size,) or norm_weight.dtype != torch.bfloat16
        ):
            raise ValueError("norm_weight must be bf16 [6144] or None")
        if cache.dtype != torch.bfloat16 or cache.ndim != 4:
            raise ValueError("cache must be a 4-D bf16 tensor")
        if shard_size % ws != 0 or cache.shape[1] != shard_size // ws:
            raise ValueError("cache head layout is inconsistent with ws")
        if off_s < 0 or off_s + ws > cache.shape[3]:
            raise ValueError("cache channel offset is out of bounds")

        index = self.generation
        input_offset = index * self.input_generation_bytes // 4
        output_offset = index * self.output_generation_bytes // 4
        normed = torch.empty_like(residual) if norm_weight is not None else None
        residual_out = torch.empty_like(residual)
        phase_splits = 8 if tokens == 1 and shard_size % 8 == 0 else 1
        phase_tile_p2 = triton.next_power_of_2(shard_size // phase_splits)
        phase_grid = (tokens, phase_splits)
        # Wide CTAs win before the grid saturates; smaller CTAs reduce register
        # pressure once high-throughput batches provide enough parallelism.
        if 128 <= tokens <= 2048:
            phase_warps = 16
        elif tokens > 2048:
            phase_warps = 8
        else:
            phase_warps = 4
        gather_warps = 4 if tokens >= 256 else 8
        _publish_input_kernel[phase_grid](
            input_tensor,
            self.input_peer_ptrs,
            input_offset,
            input_tensor.stride(0),
            C=hidden_size,
            CS=shard_size,
            CS_P2=phase_tile_p2,
            SPLITS=phase_splits,
            RANK=self.rank,
            WORLD=self.world_size,
            USE_PDL=self.use_pdl,
            launch_pdl=self.use_pdl,
            num_warps=phase_warps,
        )
        _reduce_insert_kernel[phase_grid](
            self.input_peer_ptrs,
            input_offset,
            cache,
            slot_mapping,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            block_size,
            C=hidden_size,
            CS=shard_size,
            CS_P2=phase_tile_p2,
            SPLITS=phase_splits,
            HEAD_SIZE=ws,
            CACHE_OFFSET=off_s,
            RANK=self.rank,
            WORLD=self.world_size,
            USE_PDL=self.use_pdl,
            launch_pdl=self.use_pdl,
            num_warps=phase_warps,
        )
        _sconv_publish_kernel[phase_grid](
            self.output_peer_ptrs,
            output_offset,
            residual,
            conv_weight,
            cache,
            positions,
            seq_idx,
            slot_mapping,
            block_table,
            residual.stride(0),
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            block_table.stride(0),
            block_table.shape[1],
            block_size,
            C=hidden_size,
            CS=shard_size,
            CS_P2=phase_tile_p2,
            SPLITS=phase_splits,
            HEAD_SIZE=ws,
            CACHE_OFFSET=off_s,
            RANK=self.rank,
            WORLD=self.world_size,
            WINDOW=self.window_size,
            USE_PDL=self.use_pdl,
            launch_pdl=self.use_pdl,
            num_warps=phase_warps,
        )
        _gather_norm_kernel[(tokens,)](
            self.output_peer_ptrs,
            output_offset,
            norm_weight if norm_weight is not None else residual,
            normed if normed is not None else residual_out,
            residual_out,
            eps,
            residual.stride(0),
            C=hidden_size,
            C_P2=triton.next_power_of_2(hidden_size),
            RANK=self.rank,
            HAS_NORM=norm_weight is not None,
            USE_PDL=self.use_pdl,
            launch_pdl=self.use_pdl,
            num_warps=gather_warps,
        )
        self.generation = (index + 1) % self.num_buffers
        return normed, residual_out


_STATE: LamportRSConv | None = None
_STATE_FAILED = False


def initialize_lamport_rs_conv(
    hidden_size: int, window_size: int, max_num_batched_tokens: int
) -> None:
    """Collectively initialize the TP-group state during model construction."""
    global _STATE, _STATE_FAILED
    if _STATE is not None:
        if _STATE.hidden_size != hidden_size or _STATE.window_size != window_size:
            raise RuntimeError("all Lamport users must share hidden and window sizes")
        return
    if _STATE_FAILED or os.environ.get("LAMPORT_RS_SCONV", "1") == "0":
        return
    try:
        max_tokens = min(_MAX_TOKENS, max_num_batched_tokens)
        _STATE = LamportRSConv(hidden_size, window_size, max_tokens=max_tokens)
    except Exception:
        _STATE_FAILED = True
        logger.exception("fused collective unavailable; use the NCCL fallback")


def get_lamport_rs_conv(hidden_size: int, window_size: int) -> LamportRSConv | None:
    """Return the state initialized with the model, or ``None`` for fallback."""
    if _STATE is not None and (
        _STATE.hidden_size != hidden_size or _STATE.window_size != window_size
    ):
        raise RuntimeError("all Lamport users must share hidden and window sizes")
    return _STATE
