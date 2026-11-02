# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch wrapper of the FlyDSL NVFP4 prefill index scorers."""

import functools
import math

import torch

from vllm.models.minimax_m3.amd.ops.flydsl_utils import cu_count, run_compiled
from vllm.utils.math_utils import cdiv

from . import kernel as _kernel

# Workgroups to aim for: more, shorter segments balance the tail of the causal
# windows; a segment keeps >= GROUP blocks.
TARGET_WORKGROUPS = 2048
# Segments are added until the last round of workgroups is at least this full.
MIN_ROUND_FILL = 0.9
WORKGROUPS_PER_CU = 2
MAX_SEGMENTS = 1024
# Small work (<= 128 query rows, or below SMALL_WORK) takes ``kernel_shared``
# with 128-row tiles, aiming at SMALL_TARGET workgroups below MID_WORK.
SMALL_WORK = 4096
MID_WORK = 16384
SMALL_TARGET = 512
# Tests: pin the kernel (True: kernel_shared) or its 32-row tiles per wave.
PIN_SHARED: bool | None = None
PIN_MQ: int | None = None

_NVFP4_E4M3_LUT: dict[torch.device, torch.Tensor] = {}


def nvfp4_e4m3_lut(device: torch.device) -> torch.Tensor:
    """[256] int64: byte j of entry s is e4m3_rn_satfinite(m_j * e4m3(s)) for the
    E2M1 magnitudes m = 0, 0.5, 1, 1.5, 2, 3, 4, 6."""
    lut = _NVFP4_E4M3_LUT.get(device)
    if lut is None:
        # Device ops only, so a first call under graph capture is fine.
        scales = torch.arange(256, dtype=torch.uint8, device=device)
        scales = scales.view(torch.float8_e4m3fn).float()
        code = torch.arange(8, device=device)
        exp, man = code // 2, (code % 2).float()
        mags = torch.where(
            exp == 0, man * 0.5, torch.exp2((exp - 1).float()) * (1.0 + man * 0.5)
        )
        # e2m1 * e4m3 is exact in fp32; saturate, then round once to E4M3.
        values = (scales[:, None] * mags[None, :]).clamp(-448.0, 448.0)
        lut = values.to(torch.float8_e4m3fn).view(torch.int64).reshape(256)
        _NVFP4_E4M3_LUT[device] = lut
    return lut


@functools.cache
def _get_kernel(heads: int):
    return _kernel.compile_score_prefill(heads)


def _kernel_shared():
    # Imported late: the decode host imports this module.
    from . import kernel_shared

    return kernel_shared


@functools.cache
def _get_kernel_shared(heads: int, mq: int):
    return _kernel_shared().compile_score_prefill_shared(heads, mq)


def _supported(q_idx, index_cache, score, *int32_tensors) -> bool:
    from vllm.platforms.rocm import on_gfx950

    heads = q_idx.shape[1]
    return (
        on_gfx950()
        and q_idx.dtype == torch.float8_e4m3fn
        and heads in (1, 2, 4)
        and q_idx.shape[2] == _kernel.D
        and q_idx.stride(2) == 1
        and q_idx.stride(1) == _kernel.D
        and q_idx.stride(0) == heads * _kernel.D
        and score.shape[0] == heads
        and score.stride(2) == 1
        and index_cache.dtype == torch.uint8
        and index_cache.stride(2) == 1
        and index_cache.stride(1) == _kernel.D // 2 + _kernel.D // 16
        and index_cache.stride(0) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
        and all(t.dtype == torch.int32 and t.is_contiguous() for t in int32_tensors)
    )


def minimax_m3_nvfp4_index_score_prefill(
    q_idx: torch.Tensor,  # [total_q, H, 128] e4m3
    index_cache: torch.Tensor,  # [num_blocks, 128, 72] uint8
    score: torch.Tensor,  # [H, total_q, >= cdiv(max_seq_len, 128)] fp32
    block_table: torch.Tensor,  # [batch, >= max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    init_blocks: int,
    local_blocks: int,
    max_query_len: int,
    max_seq_len: int,
) -> None:
    """Block scores of every prefill row for 1, 2 or 4 index heads."""
    batch = cu_seqlens_q.shape[0] - 1
    total_q, heads, _ = q_idx.shape
    if total_q == 0 or block_table.shape[0] == 0:
        return
    assert _supported(q_idx, index_cache, score, block_table, cu_seqlens_q, seq_lens)
    max_blocks = cdiv(max_seq_len, _kernel.BLK)
    assert max_blocks <= score.shape[2]
    work = max_blocks * cdiv(max_query_len, 256) * batch * heads
    small = max_query_len * heads <= 128 or work < SMALL_WORK
    shared = small if PIN_SHARED is None else PIN_SHARED
    mq = PIN_MQ or (1 if small else 2)
    target = SMALL_TARGET if shared and work < MID_WORK else TARGET_WORKGROUPS
    tile_q = _kernel_shared().tile_q(mq) if shared else _kernel.TILE_Q
    qt = cdiv(max_query_len * heads, tile_q)
    min_seg = 1 if shared else _kernel.GROUP
    max_nseg = max(1, min(MAX_SEGMENTS, max_blocks // min_seg))
    nseg = max(1, min(max_nseg, target // max(1, qt * batch)))
    slots = WORKGROUPS_PER_CU * cu_count(q_idx.device)
    while nseg < max_nseg:
        rounds = qt * batch * nseg / slots
        if rounds / math.ceil(rounds) >= MIN_ROUND_FILL:
            break
        nseg += 1
    lut = () if shared else (nvfp4_e4m3_lut(q_idx.device).data_ptr(),)
    run_compiled(
        _get_kernel_shared(heads, mq) if shared else _get_kernel(heads),
        q_idx.data_ptr(),
        index_cache.data_ptr(),
        *lut,
        score.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens.data_ptr(),
        int(index_cache.stride(0)),
        int(block_table.stride(0)),
        int(score.stride(1)),
        int(nseg),
        int(qt),
        int(total_q),
        int(batch),
        int(init_blocks),
        int(local_blocks),
        int(score.stride(0)),
        int(cdiv(qt * nseg * batch, 8) * 8),
        torch.cuda.current_stream(),
    )
