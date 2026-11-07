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
def _get_kernel(splits: int, topk: int, xcds: int, fp8_out: bool = False):
    return _kernel.compile_nvfp4_sparse_decode(splits, topk, xcds, fp8_out)


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


def _xcd_splits(
    num_reqs: int, xcds: int, topk: int, decode_query_len: int, num_kv_heads: int
) -> int | None:
    """Splits for the XCD-aware grid with up to three requests per XCD, or None.

    A request's rows and splits all run on one XCD (32 CUs), so the cost model of
    `decode_splits`, which spreads the grid over every CU, mispredicts: at 5 requests it
    picks 6 splits for 15.7 us where 4 splits take 11.9 us. Measured on MI355X (top-k
    16, 8 rows a request, 30K-400K contexts, 1-24 requests; the kernel time does not
    depend on the context): 4 splits is best up to 8 requests (11.4-12.4 us, one request
    per XCD) and for 17-24 requests (21 us), 2 splits for 9-16 requests (15.3-16 us).
    Wider grids use the cost model. A request's KV heads also share its XCD, so with 2
    or 4 KV heads per rank (TP2, TP1) the table over-splits and the cost model is kept.
    """
    if xcds <= 0 or topk != 16 or decode_query_len != 8 or num_kv_heads != 1:
        return None  # tuned for DSpark 7, one KV head per rank
    per_xcd = -(-num_reqs // xcds)
    if per_xcd == 2:
        return 2
    return 4 if per_xcd <= 3 else None


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
) -> None:
    """Sparse decode attention over the NVFP4 K/V pages.

    Row i is query row i % decode_query_len of request i // decode_query_len
    and attends to the tokens of its `topk_idx` blocks up to its position.

    With `out_q` and `out_scale` the output is written as MXFP8 (E4M3 values
    and row-major E8M0 scales per 32 dims, equal to `mxfp8_e4m3_quantize` of
    the bf16 output) instead of bf16 into `output`, which is then unused.
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
    if xcds:
        group = xcds * decode_query_len
        grid_x = -(-rows // group) * group
    else:
        grid_x = rows
    if splits is None:
        wg_rows = grid_x * num_kv_heads
        if wg_rows > MAX_SPLIT_ROWS:
            splits = 1
        else:
            splits = _xcd_splits(
                rows // decode_query_len, xcds, topk, decode_query_len, num_kv_heads
            )
            if splits is None:
                splits = decode_splits(rows * num_kv_heads, topk, cu_count(q.device))
    assert splits == 1 or grid_x * num_kv_heads <= MAX_SPLIT_ROWS
    part, cnt = _split_buffers(q.device)
    unit = unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    c = float(np.float32(sm_scale) * np.float32(LOG2E))
    run_compiled(
        _get_kernel(splits, topk, xcds, fp8_out),
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
    )
