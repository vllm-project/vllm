# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch wrappers of the FlyDSL NVFP4 decode index scorers."""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.flydsl_utils import run_compiled
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.host import nvfp4_e4m3_lut
from vllm.utils.math_utils import cdiv

from . import kernel as _kernel

# 158 VGPRs = 3 waves per SIMD; more waves add a second round of workgroups.
TARGET_WAVES = 3072
# Scores are buffered in LDS and stored by whole rows once the batch's score
# footprint reaches this; smaller footprints store directly.
BUFFER_MIN_BYTES = 8 << 20
UNITS_PER_SLOT = 4


def _end_bytes(t: torch.Tensor) -> int:
    """Bytes from the data pointer to the end of the last element."""
    return (
        sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
    ) * t.element_size()


def _unit(max_blocks: int, slots: int, umax: int, footprint: int) -> int:
    """Consecutive blocks per buffered unit (a power of two, 4..umax) giving
    every wave slot >= UNITS_PER_SLOT units, or 0 for direct stores."""
    if footprint < BUFFER_MIN_BYTES:
        return 0
    u = umax
    while u > 4 and cdiv(max_blocks, u) < UNITS_PER_SLOT * slots:
        u //= 2
    return u if cdiv(max_blocks, u) >= UNITS_PER_SLOT * slots else 0


@functools.cache
def _get_kernel(rows: int, heads: int, buffered: bool):
    return _kernel.compile_score_decode(rows, heads, buffered)


def _supported(
    q_idx: torch.Tensor,
    index_cache: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
) -> bool:
    from vllm.platforms.rocm import on_gfx950

    num_heads = q_idx.shape[1]
    return (
        on_gfx950()
        and num_heads in (1, 2, 4)
        and 1 <= num_heads * query_len <= _kernel.MAX_ROWS
        and q_idx.dtype == torch.float8_e4m3fn
        and q_idx.shape[2] == _kernel.D
        and q_idx.stride(2) == 1
        and q_idx.stride(0) % 16 == 0
        and q_idx.stride(1) % 16 == 0
        and q_idx.data_ptr() % 16 == 0
        and index_cache.dtype == torch.uint8
        and index_cache.stride(2) == 1
        and index_cache.stride(1) == _kernel.D // 2 + _kernel.D // 16
        and index_cache.stride(0) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
        and score.dtype == torch.float32
        and score.stride(2) == 1
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1
        and seq_lens.dtype == torch.int32
        and seq_lens.is_contiguous()
    )


def minimax_m3_nvfp4_index_score_decode(
    q_idx: torch.Tensor,  # [num_reqs * query_len, H, 128] e4m3
    index_cache: torch.Tensor,  # [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, num_reqs * query_len, >= max_blocks] fp32
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    max_seq_len: int,
) -> None:
    """Block scores of every decode row for blocks below cdiv(seq_len, 128)."""
    total_q, num_heads, _ = q_idx.shape
    num_reqs = total_q // query_len
    if num_reqs == 0:
        return
    assert _supported(q_idx, index_cache, score, block_table, seq_lens, query_len)
    rows = num_heads * query_len
    max_blocks = max(1, cdiv(max_seq_len, _kernel.BLK))
    # A chunk of NW x (pages per wave) requests; with at most one page per
    # wave, chunks of one request and each wave on its own positions.
    pages_per_wave = cdiv(num_reqs * max_blocks, TARGET_WAVES)
    s_bytes = _end_bytes(score)
    assert s_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    footprint = num_reqs * rows * max_blocks * 4
    chunk = 1 if pages_per_wave <= 1 else _kernel.NW * pages_per_wave
    chunk_cap = _kernel.chunk_max(_kernel.padded_rows(rows))
    chunk = max(1, min(chunk, chunk_cap, num_reqs))
    nch = cdiv(num_reqs, chunk)
    chunk = cdiv(num_reqs, nch)
    per_wave = chunk < _kernel.NW
    wanted = max(1, TARGET_WAVES // (_kernel.NW * nch))
    # Past 16 rows the buffered variant's flush costs more than it saves.
    unit = (
        _unit(
            max_blocks,
            wanted * (_kernel.NW if per_wave else 1),
            _kernel.UMAX,
            footprint,
        )
        if _kernel.padded_rows(rows) <= 16
        else 0
    )
    units = cdiv(max_blocks, max(1, unit))
    positions = cdiv(units, _kernel.NW) if per_wave else units
    splits = max(1, min(positions, wanted))
    run_compiled(
        _get_kernel(rows, num_heads, unit > 0),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(q_idx.device).data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(q_idx.stride(0)),
        int(q_idx.stride(1)),
        _end_bytes(q_idx),
        int(score.stride(0)),
        int(score.stride(1)),
        s_bytes,
        int(block_table.stride(0)),
        _end_bytes(block_table),
        int(num_reqs),
        int(chunk),
        int(nch),
        int(splits),
        int(init_blocks),
        int(local_blocks),
        int(per_wave),
        int(unit),
        torch.cuda.current_stream(),
        # Keep the buffered variant at the plain one's occupancy.
        waves_per_eu=3 if unit > 0 and _kernel.padded_rows(rows) <= 16 else None,
    )


@functools.cache
def _get_kernel_cp_push(query_groups: int):
    from . import kernel_cp_push

    return kernel_cp_push.compile_score_push(query_groups)


def minimax_m3_nvfp4_index_score_push_cp(
    iq: torch.Tensor,  # this rank's index head [num_reqs * query_len, 128] e4m3
    index_cache: torch.Tensor,  # [num_blocks, 128, 72] uint8
    dests: list[int],  # head h's scores [rows, row_stride] fp32 on rank h
    row_stride: int,
    dest_bytes: int,
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    init_blocks: int,
    local_blocks: int,
    rank: int,
    counts: torch.Tensor,  # index_cp's device counters
    flags: list[int],  # rank p's flag words
    iq_bufs: list[int],  # rank p's index-q receive buffer [4, iq_rows, 128] e4m3
    iq_rows: int,
    grid: int = 0,
    query_len: int = 4,
) -> None:
    """This rank's share of all four heads' decode scores, pushed to the heads'
    ranks; returns once every rank's scores have arrived."""
    from . import kernel_cp_push

    rows = iq.shape[0]
    assert query_len in (4, 8) and rows % query_len == 0
    assert rows <= iq_rows
    assert iq.dtype == torch.float8_e4m3fn and iq.shape[1] == kernel_cp_push.D
    assert iq.stride(1) == 1 and iq.stride(0) == kernel_cp_push.D
    # Each four-row tile owns separate query flags and a rebased causal end.
    num_reqs = rows // 4
    assert num_reqs <= kernel_cp_push.MAX_REQS
    if not grid:
        small = num_reqs <= kernel_cp_push.GRID_SMALL_REQS
        grid = kernel_cp_push.GRID_SMALL if small else kernel_cp_push.GRID
    assert 0 < grid <= kernel_cp_push.SC_MAX_WG
    if num_reqs == 0:
        return
    assert dest_bytes < 2**31 * 4, "score offsets must fit in int32 elements"
    run_compiled(
        _get_kernel_cp_push(query_len // 4),
        iq.data_ptr(),
        index_cache.data_ptr(),
        nvfp4_e4m3_lut(iq.device).data_ptr(),
        *dests,
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        counts.data_ptr(),
        *flags,
        *iq_bufs,
        int(iq_rows),
        int(index_cache.stride(0)),
        int(row_stride),
        int(dest_bytes),
        int(block_table.stride(0)),
        _end_bytes(block_table),
        int(num_reqs),
        int(init_blocks),
        int(local_blocks),
        int(rank),
        int(grid),
        torch.cuda.current_stream(),
    )
