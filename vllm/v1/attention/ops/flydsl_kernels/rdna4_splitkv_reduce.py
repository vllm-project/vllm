# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compact wave32 reducer for split-KV decode partials."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath

from .runtime import run_compiled as _run_compiled

WAVE_SIZE = 32
LOG2E = 1.4426950408889634
SUPPORTED_SPLITS = (2, 4, 8, 16)
SUPPORTED_HEAD_DIMS = (128, 256)


def _flat_view(tensor: fx.Tensor, elem_type=None) -> fx.Tensor:
    iterator = fx.get_iter(tensor)
    if elem_type is not None:
        iterator = fx.recast_iter(elem_type, iterator)
    return fx.make_view(iterator, fx.make_layout(1 << 30, 1))


@functools.lru_cache(maxsize=128)
def compile_local_reduce(
    *,
    head_dim: int,
    splits: int,
    num_query_heads: int,
    output_dtype: str,
    split_block_size: int,
    waves_per_block: int,
):
    """Compile a one-wave-per-row, split-specialized LSE reducer."""

    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"unsupported head dimension {head_dim}")
    if splits not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split count {splits}")
    if output_dtype not in ("bf16", "f16"):
        raise ValueError(f"unsupported output dtype {output_dtype!r}")
    if waves_per_block not in (1, 2, 4):
        raise ValueError(f"unsupported waves per block {waves_per_block}")
    out_type = fx.BFloat16 if output_dtype == "bf16" else fx.Float16
    values_per_lane = head_dim // WAVE_SIZE
    block_threads = WAVE_SIZE * waves_per_block

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def reduce_kernel(
        mid_out_ptr: fx.Tensor,
        mid_lse_ptr: fx.Tensor,
        seq_lens_ptr: fx.Tensor,
        query_start_loc_ptr: fx.Tensor,
        output_ptr: fx.Tensor,
        batch: fx.Int32,
        out_stride0: fx.Int32,
        out_stride1: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        logical_row = fx.Int32(gpu.block_idx.x) * waves_per_block + wave
        total_rows = batch * num_query_heads
        row_valid = logical_row < total_rows
        safe_row = row_valid.select(logical_row, fx.Int32(0))
        seq = safe_row // num_query_heads
        qh = safe_row % num_query_heads

        mid_out = _flat_view(mid_out_ptr, fx.Float32)
        mid_lse = _flat_view(mid_lse_ptr, fx.Float32)
        seq_lens = _flat_view(seq_lens_ptr)
        query_start_loc = _flat_view(query_start_loc_ptr)
        output = _flat_view(output_ptr, out_type)

        seq_len = fx.Int32(seq_lens[seq])
        split_len = (
            ((seq_len + splits - 1) // splits + split_block_size - 1)
            // split_block_size
            * split_block_size
        )
        query_row = fx.Int32(query_start_loc[seq])
        is_decode = fx.Int32(query_start_loc[seq + 1]) - query_row == 1

        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        one = fx.Float32(1.0)
        lse_base = seq * ml_stride0 + qh * ml_stride1
        mo_base = seq * mo_stride0 + qh * mo_stride1
        out_base = query_row * out_stride0 + qh * out_stride1
        init_state = [neg_inf, zero] + [zero for _ in range_constexpr(values_per_lane)]

        # This is deliberately the oracle's left-to-right FP32 combine,
        # not a global-max weighted sum. Empty tail splits are selected
        # away without reading their uninitialized scratch rows. A runtime
        # loop reuses the small live accumulator set; the fixed upper bound
        # remains baked into each split-count specialization.
        for split, state in range(  # type: ignore[call-overload]
            fx.Int32(0), fx.Int32(splits), fx.Int32(1), init=init_state
        ):
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            accum = [
                fx.Float32(state[2 + element])
                for element in range_constexpr(values_per_lane)
            ]
            split_i32 = fx.Int32(split)
            split_active = split_i32 * split_len < seq_len
            safe_split = split_active.select(split_i32, fx.Int32(0))
            partial_lse = fx.Float32(mid_lse[lse_base + safe_split * ml_stride2])
            next_max = split_active.select(
                fx.max(running_max, partial_lse), running_max
            )
            partial_is_new_max = partial_lse > running_max
            rescale = fmath.exp2(
                partial_is_new_max.select(
                    running_max - partial_lse,
                    partial_lse - running_max,
                )
                * LOG2E
            )
            alpha = split_active.select(partial_is_new_max.select(rescale, one), one)
            beta = split_active.select(partial_is_new_max.select(one, rescale), zero)

            for element in range_constexpr(values_per_lane):
                d = lane + element * WAVE_SIZE
                partial = fx.Float32(mid_out[mo_base + safe_split * mo_stride2 + d])
                accum[element] = accum[element] * alpha + partial * beta
            results = yield [next_max, running_sum * alpha + beta] + accum

        running_sum = fx.Float32(results[1])
        accum = [
            fx.Float32(results[2 + element])
            for element in range_constexpr(values_per_lane)
        ]
        inv_sum = one / (running_sum + 1.0e-10)
        for element in range_constexpr(values_per_lane):
            d = lane + element * WAVE_SIZE
            if row_valid & is_decode:
                output[out_base + d] = (accum[element] * inv_sum).to(out_type)

    @flyc.jit
    def launch_reduce(
        mid_out: fx.Tensor,
        mid_lse: fx.Tensor,
        seq_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        output: fx.Tensor,
        batch: fx.Int32,
        out_stride0: fx.Int32,
        out_stride1: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduce_kernel(
            mid_out,
            mid_lse,
            seq_lens,
            query_start_loc,
            output,
            batch,
            out_stride0,
            out_stride1,
            mo_stride0,
            mo_stride1,
            mo_stride2,
            ml_stride0,
            ml_stride1,
            ml_stride2,
        ).launch(
            grid=(
                (batch * num_query_heads + waves_per_block - 1) // waves_per_block,
                1,
                1,
            ),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_reduce


def _launch_reduce(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    splits: int,
    split_block_size: int,
    stream: torch.cuda.Stream,
) -> None:
    output_dtype = "bf16" if output.dtype == torch.bfloat16 else "f16"
    total_rows = int(seq_lens.numel()) * int(query.shape[1])
    # A 64-row reduction has exactly two result waves per gfx1201 WGP.
    # Pack those waves into one workgroup so the launch presents one workgroup
    # per WGP, while retaining the lower-overhead one-wave shape for smaller
    # reductions and the established four-wave shape above 128 rows.
    waves_per_block = 1 if total_rows < 64 else (2 if total_rows <= 128 else 4)
    launch = compile_local_reduce(
        head_dim=int(query.shape[2]),
        splits=int(splits),
        num_query_heads=int(query.shape[1]),
        output_dtype=output_dtype,
        split_block_size=int(split_block_size),
        waves_per_block=waves_per_block,
    )
    _run_compiled(
        launch,
        mid_out,
        mid_lse,
        seq_lens,
        query_start_loc,
        output,
        int(seq_lens.numel()),
        int(output.stride(0)),
        int(output.stride(1)),
        *map(int, mid_out.stride()[:3]),
        *map(int, mid_lse.stride()),
        stream,
    )
