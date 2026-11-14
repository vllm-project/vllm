# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch wrapper of the FlyDSL NVFP4 sparse decode."""

import ctypes
import functools

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    LOG2E,
    cu_count,
    run_compiled,
    unit_scale,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.host import nvfp4_e4m3_lut

from . import kernel as _kernel

logger = init_logger(__name__)

_HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS = 10018
_SUPPORTED_XCD_COUNT = 8

MAX_SPLITS = 8
# Launches with more (row, KV head) workgroup rows than this do not split.
MAX_SPLIT_ROWS = 1024


@functools.cache
def _get_kernel(
    splits: int,
    topk: int,
    xcds: int,
    fp8_out: bool = False,
    place: int = _kernel.PLACE_REQ,
    waves_per_eu: int = 0,
):
    # waves_per_eu is not a kernel argument: it keys the cache because each
    # (kernel, hint) pair is compiled once (run_compiled) on its own launcher.
    return _kernel.compile_nvfp4_sparse_decode(splits, topk, xcds, fp8_out, place)


def _query_xcd_count(device_index: int) -> int:
    """The device's XCD count, or zero when HIP cannot report it."""
    try:
        get_attribute = ctypes.CDLL("libamdhip64.so").hipDeviceGetAttribute
        get_attribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        get_attribute.restype = ctypes.c_int
        value = ctypes.c_int()
        status = get_attribute(
            ctypes.byref(value), _HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS, device_index
        )
        return value.value if status == 0 and value.value > 0 else 0
    except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
        return 0


@functools.cache
def _xcd_count(device_index: int) -> int:
    """8 on gfx950 (the topology the XCD grid was tuned on), else 0."""
    from vllm.platforms.rocm import on_gfx950

    queried = _query_xcd_count(device_index)
    count = queried if on_gfx950() and queried == _SUPPORTED_XCD_COUNT else 0
    logger.info(
        "MiniMax-M3 NVFP4 sparse decode: XCD-aware grid %s on device %d "
        "(reported XCDs=%d)",
        "on" if count else "off",
        device_index,
        queried,
    )
    return count


@functools.cache
def _split_buffers(device: torch.device):
    """Split partials and tickets; allocated in the warm-up, before capture."""
    part = torch.empty(
        MAX_SPLIT_ROWS * MAX_SPLITS * 64 * _kernel.PART_F,
        dtype=torch.float32,
        device=device,
    )
    return part, torch.zeros(MAX_SPLIT_ROWS, dtype=torch.int32, device=device)


def decode_splits(num_rows: int, num_blocks: int, cus: int) -> int:
    """Splits of a row's blocks.

    The kernel is VALU-bound, so the cost model is rounds of workgroups per CU
    times blocks per split plus about one block of fixed cost; the fewest splits
    among the minima, each split at least two blocks.
    """
    best_cost, best = None, 1
    for s in range(1, MAX_SPLITS + 1):
        per = -(-num_blocks // s)
        if s > 1 and per < 2:
            break
        cost = -(-(num_rows * s) // cus) * (per + 1)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, s
    return best


# Splits of the XCD-aware plan; more than 4 never paid off (one round of
# 2-block workgroups costs about what 4-block ones do, plus the wider combine).
MAX_XCD_SPLITS = 4
# Requests past one round on the busiest XCD that run 2 balanced splits on the
# two-workgroups-a-CU build (measured up to here).
MAX_OCC2_REQS = 48


def _xcd_plan(
    num_reqs: int,
    xcds: int,
    topk: int,
    decode_query_len: int,
    num_kv_heads: int,
    cus: int,
) -> tuple[int, int, int] | None:
    """(splits, row placement, waves_per_eu) of the XCD-aware grid, or None.

    The kernel takes 259 VGPRs, so a CU runs one workgroup at a time, and a
    second round of workgroups on an XCD costs a whole workgroup latency chain
    (block-table, page and partial round trips: about 7 us cold) for a few
    blocks saved. So the plan is the most splits (at most `MAX_XCD_SPLITS`)
    for which the busiest XCD runs all its workgroups in one round:
    `PLACE_REQ` (request r on XCD r % xcds) puts ql x ceil(reqs / xcds) rows
    on the busiest XCD, `PLACE_BAL` ceil(rows / xcds) on every XCD;
    `PLACE_REQ` on ties (a request's rows share an L2). With more rows than
    an XCD has CUs, up to `MAX_OCC2_REQS` requests: 2 balanced splits built
    for two workgroups a CU (waves_per_eu 2: 256 VGPRs, 2 spilled outside the
    loop), whose overlapping latency chains beat a second round; past that,
    None (the cost model on the plain grid).

    Measured on MI355X (top-k 16, DSpark 7 = 8 rows a request, one KV head, 64K /
    128K / 256K contexts, cold, graph replay): 1-8 requests 4 splits PLACE_REQ
    (unchanged); 9-10 requests 3 splits PLACE_BAL (14.5-15.7 us vs 16.2-17.1
    with 2); 11-16 2 splits PLACE_REQ (unchanged); 17-32 no split, PLACE_REQ
    (22-24 us; 17-24 took 29-31 us with the former 4 splits, 25-32 already
    ran unsplit); 33-48 2 splits PLACE_BAL occupancy 2 (34.5-36.7 us vs 44-50
    with the cost model's 4 splits). A request's KV heads also share its XCD,
    so with 2 or 4 KV heads per rank (TP2, TP1) the plan does not apply and
    the cost model is kept.
    """
    if xcds <= 0 or topk != 16 or decode_query_len != 8 or num_kv_heads != 1:
        return None  # tuned for DSpark 7, one KV head per rank
    cus_per_xcd = cus // xcds
    rows = num_reqs * decode_query_len
    req_rows = decode_query_len * -(-num_reqs // xcds)
    bal_rows = -(-rows // xcds)
    s_req = min(MAX_XCD_SPLITS, cus_per_xcd // req_rows)
    s_bal = min(MAX_XCD_SPLITS, cus_per_xcd // bal_rows)
    if s_req >= s_bal and s_req:
        return s_req, _kernel.PLACE_REQ, 0
    if s_bal:
        return s_bal, _kernel.PLACE_BAL, 0
    if num_reqs <= MAX_OCC2_REQS:
        return 2, _kernel.PLACE_BAL, 2
    return None


def _grid_x(rows: int, ql: int, xcds: int, place: int) -> int:
    """grid.x of the kernel's row placement (see `kernel.py`)."""
    if not xcds:
        return rows
    if place == _kernel.PLACE_REQ:
        group = xcds * ql
        return -(-rows // group) * group
    sf = (rows // ql // xcds) * ql
    return (sf + -(-(rows - sf * xcds) // xcds)) * xcds


def minimax_m3_nvfp4_sparse_decode_fly(
    q: torch.Tensor,  # [rows, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, rows, topk] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    output: torch.Tensor | None,  # [rows, H, 128] bf16; None with out_q
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    decode_query_len: int = 1,
    splits: int | None = None,
    xcd_count: int | None = None,
    out_q: torch.Tensor | None = None,  # [rows, H * 128] float8_e4m3fn
    out_scale: torch.Tensor | None = None,  # [rows, H * 4] uint8 (E8M0)
    place: int | None = None,
    waves_per_eu: int | None = None,
) -> None:
    """Sparse decode attention over the NVFP4 K/V pages.

    Row i is query row i % decode_query_len of request i // decode_query_len
    and attends to the tokens of its `topk_idx` blocks up to its position.

    With `out_q` and `out_scale` the output is written as MXFP8 (E4M3 values
    and row-major E8M0 scales per 32 dims, equal to `mxfp8_e4m3_quantize` of
    the bf16 output) instead of bf16 into `output`, which is then unused.

    `splits` / `place` / `waves_per_eu` default to the XCD plan (`_xcd_plan`),
    else the cost model; an explicit `splits` keeps `PLACE_REQ`. For a given
    split count the output does not depend on the placement or the occupancy
    hint (the splits combine in split order).
    """
    rows, num_heads, head_dim = q.shape
    if rows == 0:
        return
    num_kv_heads = kv_cache.shape[1] // 2
    topk = topk_idx.shape[-1]
    assert rows == seq_lens.shape[0] * decode_query_len
    assert head_dim == _kernel.D and num_heads == num_kv_heads * _kernel.HG
    fp8_out = out_q is not None
    assert fp8_out == (out_scale is not None) and fp8_out == (output is None)
    assert q.dtype == torch.bfloat16
    assert q.is_contiguous()
    if out_q is not None and out_scale is not None:
        assert out_q.dtype == torch.float8_e4m3fn and out_scale.dtype == torch.uint8
        assert out_q.is_contiguous() and out_scale.is_contiguous()
        assert out_q.shape[0] >= rows and out_q.shape[1] == num_heads * head_dim
        assert out_scale.shape[0] >= rows
        assert out_scale.shape[1] == num_heads * head_dim // 32
        out_ptr, out_q_ptr, out_scale_ptr = (
            q.data_ptr(),
            out_q.data_ptr(),
            out_scale.data_ptr(),
        )
    else:
        assert output is not None
        assert output.dtype == torch.bfloat16 and output.is_contiguous()
        out_ptr, out_q_ptr, out_scale_ptr = (
            output.data_ptr(),
            q.data_ptr(),
            q.data_ptr(),
        )
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(1) == _kernel.BLK * 72
    assert kv_cache.stride(3) == 1 and kv_cache.stride(0) < 2**31
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] >= rows
    assert topk_idx.stride(2) == 1 and topk_idx.stride(1) == topk
    for t in (block_table, seq_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    if xcd_count is None:
        xcd_count = _xcd_count(q.device.index)
    xcds = xcd_count if xcd_count and xcd_count > 1 else 0
    if splits is None and place is None:
        plan = _xcd_plan(
            rows // decode_query_len,
            xcds,
            topk,
            decode_query_len,
            num_kv_heads,
            cu_count(q.device),
        )
        if plan is not None:
            splits, place, plan_wpe = plan
            if waves_per_eu is None:
                waves_per_eu = plan_wpe or None
    if place is None:
        place = _kernel.PLACE_REQ
    assert place == _kernel.PLACE_REQ or xcds
    grid_x = _grid_x(rows, decode_query_len, xcds, place)
    if grid_x * num_kv_heads > MAX_SPLIT_ROWS:
        splits = 1
    elif splits is None:
        splits = decode_splits(rows * num_kv_heads, topk, cu_count(q.device))
    assert splits == 1 or grid_x * num_kv_heads <= MAX_SPLIT_ROWS
    part, cnt = _split_buffers(q.device)
    unit = unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    wpe = waves_per_eu or 0
    run_compiled(
        _get_kernel(splits, topk, xcds, fp8_out, place, wpe),
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk_idx.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        out_ptr,
        out_q_ptr,
        out_scale_ptr,
        ks.data_ptr(),
        vs.data_ptr(),
        nvfp4_e4m3_lut(q.device).data_ptr(),
        part.data_ptr(),
        cnt.data_ptr(),
        part.numel() * 4,
        rows,
        decode_query_len,
        num_heads,
        num_kv_heads,
        topk_idx.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        c,
        grid_x,
        torch.cuda.current_stream(q.device),
        waves_per_eu=wpe or None,
    )
