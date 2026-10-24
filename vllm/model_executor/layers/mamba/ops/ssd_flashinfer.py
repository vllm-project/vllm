# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 prefill SSD scan through FlashInfer's fused SSD (Cake backend).

Runs the chunked scan of ``mamba_chunk_scan_combined_varlen`` in two kernel
launches instead of five, with the packed-varlen ``cu_seqlens`` form (the
kernels derive their chunk metadata on the device).
"""

from functools import cache, lru_cache

import torch

from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.triton_utils import tl, triton

# Shapes compiled into FlashInfer's Cake SSD programs.
FLASHINFER_SSD_HEAD_DIM = 64
FLASHINFER_SSD_STATE_SIZE = 128


@lru_cache(maxsize=16)
def get_ssd_runner(
    device_index: int,
    stream_id: int,
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
    has_initial_states: bool,
):
    # Runner workspaces are mutable, so like flashinfer.mamba.ssd_combined_fwd
    # keep one runner per stream.
    from flashinfer.mamba import SSDCombined

    with torch.accelerator.device_index(device_index):
        return SSDCombined(
            chunk_size=128,
            nheads=nheads,
            headdim=FLASHINFER_SSD_HEAD_DIM,
            dstate=FLASHINFER_SSD_STATE_SIZE,
            ngroups=ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=has_initial_states,
            has_varlen=True,
            has_z=False,
            backend="cake",
        )


def mamba_chunk_scan_flashinfer(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor | None,
) -> None:
    """Scan the prefill tokens and write each sequence's final state.

    x: (seqlen, nheads, headdim) bf16, may be a strided view
    dt: (seqlen, nheads)
    A, dt_bias: (nheads,) fp32
    B, C: (seqlen, ngroups, dstate) bf16, may be strided views
    D: (nheads,) bf16
    cu_seqlens: (batch + 1,) int32, starting at 0
    out: (seqlen, nheads, headdim) bf16, contiguous; written in place
    ssm_state: (num_slots, nheads, headdim, dstate) state cache
    state_indices: (batch,) cache slot of each sequence
    has_initial_states: (batch,) bool, or None when no sequence has a state
    """
    initial_states = (
        gather_initial_states(ssm_state, state_indices, has_initial_states)
        if has_initial_states is not None
        else None
    )
    runner = get_ssd_runner(
        x.device.index,
        torch.accelerator.current_stream(x.device).stream_id,
        x.size(1),
        B.size(1),
        ssm_state.dtype,
        initial_states is not None,
    )
    _, final_states = runner.run(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        cu_seqlens=cu_seqlens,
        out=out.unsqueeze(0),
        return_final_states=True,
    )
    scatter_states(ssm_state, final_states, state_indices)


# Static-shape launch of a piecewise-captured mixer (splitting_ops leave out
# vllm::mamba_mixer2). Programs along the sequence axis of the state gather /
# scatter: a launch width only, since each program strides over the real
# prefill sequences, so any value is correct.
_STATIC_STATE_COPY_SEQS = 16
# The rows of the static launch outside the prefill tokens (decode tokens and
# padding) get this dt. FlashInfer's SSD preprocess
# (flashinfer-ai/flashinfer#4576) and vLLM's Triton softplus (mamba_ssm.py) both
# compute softplus(x) as log(1 + exp(x)) for x <= 20, and exp underflows to 0
# in float32 below about -88, so softplus(dt + dt_bias) is exactly 0: with x, B
# and C zeroed these rows leave their sequence's state unchanged and output 0.
# This needs dt_limit[0] == 0; the mixer always passes (0, inf).
_IDENTITY_DT = tl.constexpr(-1.0e4)
# Programs of the static scan's input preparation: a launch width only, the
# programs stride over the rows.
_STATIC_PREPARE_PROGRAMS = 1024

# Per device: the initial-state rows and the dt copy of the static launch,
# allocated by allocate_static_ssd_scratch() for the largest capture size and
# shared by every size and layer.
_static_ssd_scratch: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def allocate_static_ssd_scratch(
    device: torch.device,
    max_seqs: int,
    max_tokens: int,
    nheads: int,
    state_dtype: torch.dtype,
) -> None:
    """Allocate, outside CUDA graph capture, the scratch shared by the static
    scans on `device` (the mixer's warmup does, for its largest launch)."""
    if device.index is None:
        device = torch.device(device.type, torch.accelerator.current_device_index())
    scratch = _static_ssd_scratch.get(device.index)
    if (
        scratch is not None
        and scratch[0].size(0) >= max_seqs
        and scratch[1].size(0) >= max_tokens
        and scratch[0].size(1) == nheads
        and scratch[0].dtype == state_dtype
    ):
        return
    _static_ssd_scratch[device.index] = (
        torch.empty(
            max_seqs,
            nheads,
            FLASHINFER_SSD_HEAD_DIM,
            FLASHINFER_SSD_STATE_SIZE,
            dtype=state_dtype,
            device=device,
        ),
        torch.empty(max_tokens, nheads, dtype=torch.bfloat16, device=device),
    )


@cache
def get_static_ssd_runner(
    device_index: int,
    num_tokens: int,
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
):
    # One runner per padded size. A runner keeps the workspace of its call
    # shape (index tables, the final states) and rebuilds it only for a new
    # shape, so the warmup builds each size's workspace outside any capture and
    # the graphs captured at that size keep using it, with no rebuild recorded
    # in them. The graphs replay on one stream.
    from flashinfer.mamba import SSDCombined

    with torch.accelerator.device_index(device_index):
        return SSDCombined(
            chunk_size=128,
            nheads=nheads,
            headdim=FLASHINFER_SSD_HEAD_DIM,
            dstate=FLASHINFER_SSD_STATE_SIZE,
            ngroups=ngroups,
            io_dtype=torch.bfloat16,
            state_dtype=state_dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=True,
            has_varlen=True,
            has_z=False,
            backend="cake",
        )


@triton.jit
def _prepare_static_scan_kernel(
    x_ptr,
    b_ptr,
    c_ptr,
    dt_ptr,
    dt_out_ptr,
    prefill_bounds_ptr,
    stride_x,
    stride_b,
    stride_c,
    stride_dt,
    num_tokens,
    x_size: tl.constexpr,
    bc_size: tl.constexpr,
    nheads: tl.constexpr,
    X_BLOCK: tl.constexpr,
    BC_BLOCK: tl.constexpr,
    H_BLOCK: tl.constexpr,
):
    """Copy dt into the scan's dt, and make every row outside the prefill
    tokens [prefill_bounds[0], prefill_bounds[1]) an identity row: x, B and C
    zeroed in place and dt = _IDENTITY_DT."""
    start = tl.load(prefill_bounds_ptr)
    end = tl.load(prefill_bounds_ptr + 1)
    heads = tl.arange(0, H_BLOCK)
    x_offsets = tl.arange(0, X_BLOCK)
    bc_offsets = tl.arange(0, BC_BLOCK)
    x_zeros = tl.zeros([X_BLOCK], dtype=x_ptr.dtype.element_ty)
    bc_zeros = tl.zeros([BC_BLOCK], dtype=b_ptr.dtype.element_ty)
    for token in tl.range(
        tl.program_id(0), num_tokens, tl.num_programs(0), num_stages=1
    ):
        row = token.to(tl.int64)
        is_prefill = (token >= start) & (token < end)
        dt = tl.load(dt_ptr + row * stride_dt + heads, mask=heads < nheads)
        tl.store(
            dt_out_ptr + row * nheads + heads,
            tl.where(is_prefill, dt, _IDENTITY_DT),
            mask=heads < nheads,
        )
        if not is_prefill:
            tl.store(
                x_ptr + row * stride_x + x_offsets, x_zeros, mask=x_offsets < x_size
            )
            tl.store(
                b_ptr + row * stride_b + bc_offsets,
                bc_zeros,
                mask=bc_offsets < bc_size,
            )
            tl.store(
                c_ptr + row * stride_c + bc_offsets,
                bc_zeros,
                mask=bc_offsets < bc_size,
            )


def _prepare_static_scan(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    dt_out: torch.Tensor,
    prefill_bounds: torch.Tensor,
) -> None:
    num_tokens, nheads = dt.shape
    for t in (x, B, C):
        assert t[0].is_contiguous()
    assert dt.stride(1) == 1 and dt_out.is_contiguous()
    assert B[0].numel() == C[0].numel()
    _prepare_static_scan_kernel[(min(num_tokens, _STATIC_PREPARE_PROGRAMS),)](
        x,
        B,
        C,
        dt,
        dt_out,
        prefill_bounds,
        x.stride(0),
        B.stride(0),
        C.stride(0),
        dt.stride(0),
        num_tokens,
        x_size=x[0].numel(),
        bc_size=B[0].numel(),
        nheads=nheads,
        X_BLOCK=triton.next_power_of_2(x[0].numel()),
        BC_BLOCK=triton.next_power_of_2(B[0].numel()),
        H_BLOCK=triton.next_power_of_2(nheads),
        num_warps=4,
    )


@triton.jit
def _copy_static_states_kernel(
    src_ptr,
    dst_ptr,
    indices_ptr,
    has_initial_states_ptr,
    num_seqs_ptr,
    stride_cache_slot,
    row_size: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sequence b of the static batch is row b of the launch's state rows.
    GATHER: cache -> rows (zero without an initial state); else rows -> cache
    (skipping negative slots)."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size
    num_seqs = tl.load(num_seqs_ptr)
    for b in tl.range(tl.program_id(1), num_seqs, tl.num_programs(1), num_stages=1):
        slot = tl.load(indices_ptr + b).to(tl.int64)
        row = b.to(tl.int64) * row_size + offsets
        if GATHER:
            has_initial_state = (tl.load(has_initial_states_ptr + b) != 0) & (slot >= 0)
            values = tl.load(
                src_ptr
                + tl.where(has_initial_state, slot, 0) * stride_cache_slot
                + offsets,
                mask=mask & has_initial_state,
                other=0.0,
            )
            tl.store(dst_ptr + row, values, mask=mask)
        else:
            values = tl.load(src_ptr + row, mask=mask & (slot >= 0))
            tl.store(
                dst_ptr + slot * stride_cache_slot + offsets,
                values,
                mask=mask & (slot >= 0),
            )


def _copy_static_states(
    cache: torch.Tensor,
    rows: torch.Tensor,
    indices: torch.Tensor,
    has_initial_states: torch.Tensor | None,
    num_seqs: torch.Tensor,
) -> None:
    row_size = cache[0].numel()
    assert cache[0].is_contiguous() and rows.is_contiguous()
    assert rows[0].numel() == row_size and rows.dtype == cache.dtype
    gather = has_initial_states is not None
    block_size = 1024
    grid = (
        triton.cdiv(row_size, block_size),
        min(indices.numel(), _STATIC_STATE_COPY_SEQS),
    )
    _copy_static_states_kernel[grid](
        cache if gather else rows,
        rows if gather else cache,
        indices,
        has_initial_states if gather else indices,
        num_seqs,
        cache.stride(0),
        row_size=row_size,
        GATHER=gather,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


def mamba_chunk_scan_flashinfer_static(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefill_bounds: torch.Tensor,
    out: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    num_seqs: torch.Tensor,
) -> None:
    """FlashInfer's scan over the whole static batch of a piecewise-captured
    mixer, [decode tokens | prefill tokens | padding] (T tokens), with every
    shape fixed by T and no host synchronization (the kernels derive their
    segments from cu_seqlens on the device).

    cu_seqlens: (num_slots + 1,) int32, [0, starts of prefills 2..n, E, ...,
        E]: num_slots sequence slots (mamba2_static_flashinfer_seqs), the n
        real prefills then empty ones, where E <= T is the first 128-token
        boundary after the prefill tokens (or T). The
        decode rows before the first prefill and the padding rows before E are
        identity rows of the first and the last sequence (x, B, C zeroed in
        place and dt = _IDENTITY_DT in the scan's copy of dt): they leave its
        state unchanged and output 0; the rows after E belong to no sequence
        and are not scanned (their out rows are left unwritten). The decode
        sub-path overwrites the decode rows of out afterwards; the padding
        rows are unused. Zeroing the rows outside the prefill tokens also
        keeps whatever the conv1d left in them, which it does not write, out
        of the real sequences: FlashInfer's 128-token tiles span sequence
        boundaries (0 * NaN). Empty slots must start on a 128-token boundary
        (an empty sequence starting inside a chunk breaks the final state of
        the sequence before it).
    prefill_bounds: (2,) int32, the prefill tokens [start, end).
    state_indices, has_initial_states: (num_slots,) int32 of the prefill slots.
    num_seqs: (1,) int32, the real prefill sequences.
    Other arguments as mamba_chunk_scan_flashinfer; the scratch must have been
    allocated (allocate_static_ssd_scratch) and each size's runner warmed up
    outside capture.
    """
    num_tokens, nheads, _ = x.shape
    num_slots = state_indices.numel()
    assert cu_seqlens.numel() == num_slots + 1
    device_index = x.device.index
    initial_rows, dt_rows = _static_ssd_scratch[device_index]
    initial_states = initial_rows[:num_slots]
    dt_scan = dt_rows[:num_tokens]
    assert initial_states.size(0) == num_slots and dt_scan.size(0) == num_tokens
    _prepare_static_scan(x, B, C, dt, dt_scan, prefill_bounds)
    _copy_static_states(
        ssm_state, initial_states, state_indices, has_initial_states, num_seqs
    )
    runner = get_static_ssd_runner(
        device_index, num_tokens, nheads, B.size(1), ssm_state.dtype
    )
    _, final_states = runner.run(
        x.unsqueeze(0),
        dt_scan.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        cu_seqlens=cu_seqlens,
        out=out.unsqueeze(0),
        return_final_states=True,
    )
    _copy_static_states(ssm_state, final_states, state_indices, None, num_seqs)


# (device index, padded size, slots, heads, groups, state dtype) of the
# static scans prepared by prepare_static_ssd.
_prepared_static_scans: set[tuple] = set()


def prepare_static_ssd(
    device: torch.device,
    launches: dict[int, int],
    nheads: int,
    ngroups: int,
    state_dtype: torch.dtype,
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> None:
    """Outside CUDA graph capture, prepare the static scans of `launches`
    (padded size -> sequence slots): allocate their shared scratch, and run
    one scan per size not prepared yet, which compiles FlashInfer's programs
    and builds the size's runner workspace. Inputs are laid out like the
    mixer's (x, B and C strided views of one conv1d output row)."""
    if not launches:
        return
    max_size = max(launches)
    allocate_static_ssd_scratch(
        device, max(launches.values()), max_size, nheads, state_dtype
    )
    pending = [
        size
        for size, slots in sorted(launches.items())
        if (device.index, size, slots, nheads, ngroups, state_dtype)
        not in _prepared_static_scans
    ]
    if not pending:
        return
    x_size = nheads * FLASHINFER_SSD_HEAD_DIM
    bc_size = ngroups * FLASHINFER_SSD_STATE_SIZE
    xBC = torch.randn(
        max_size, x_size + 2 * bc_size, device=device, dtype=torch.bfloat16
    )
    x = xBC[:, :x_size].view(max_size, nheads, FLASHINFER_SSD_HEAD_DIM)
    B = xBC[:, x_size : x_size + bc_size].view(
        max_size, ngroups, FLASHINFER_SSD_STATE_SIZE
    )
    C = xBC[:, x_size + bc_size :].view(max_size, ngroups, FLASHINFER_SSD_STATE_SIZE)
    dt = torch.randn(max_size, 2 * nheads, device=device, dtype=torch.bfloat16)[
        :, nheads:
    ]
    out = torch.empty(
        max_size, nheads, FLASHINFER_SSD_HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros(
        2,
        nheads,
        FLASHINFER_SSD_HEAD_DIM,
        FLASHINFER_SSD_STATE_SIZE,
        device=device,
        dtype=state_dtype,
    )
    num_seqs = torch.ones(1, device=device, dtype=torch.int32)
    for size in pending:
        slots = launches[size]
        # One prefill sequence in the middle of the batch.
        state_indices = torch.full((slots,), -1, device=device, dtype=torch.int32)
        state_indices[0] = 1
        cu_seqlens = torch.full((slots + 1,), size, device=device, dtype=torch.int32)
        cu_seqlens[0] = 0
        prefill_bounds = torch.tensor(
            [size // 4, size // 2], device=device, dtype=torch.int32
        )
        mamba_chunk_scan_flashinfer_static(
            x[:size],
            dt[:size],
            A,
            B[:size],
            C[:size],
            D,
            dt_bias,
            cu_seqlens,
            prefill_bounds,
            out[:size],
            cache,
            state_indices,
            torch.zeros_like(state_indices),
            num_seqs,
        )
        _prepared_static_scans.add(
            (device.index, size, slots, nheads, ngroups, state_dtype)
        )
