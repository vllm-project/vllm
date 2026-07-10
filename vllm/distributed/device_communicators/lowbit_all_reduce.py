# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental low-bit TP all-reduce prototypes."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.logger import init_logger

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

logger = init_logger(__name__)
_INT8_WORKSPACES: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_INT8_BF16PACK_WORKSPACES: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, int]
] = {}
_FP4_WORKSPACES: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_SHARED_ROW_WORKSPACES: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


if triton is not None:

    @triton.jit
    def _fp4_e2m1_code(vals):
        y = tl.minimum(tl.abs(vals), 6.0)
        mag = tl.full(vals.shape, 0, tl.uint8)
        mag += tl.where(y > 0.25, 1, 0).to(tl.uint8)
        mag += tl.where(y > 0.75, 1, 0).to(tl.uint8)
        mag += tl.where(y > 1.25, 1, 0).to(tl.uint8)
        mag += tl.where(y > 1.75, 1, 0).to(tl.uint8)
        mag += tl.where(y > 2.5, 1, 0).to(tl.uint8)
        mag += tl.where(y > 3.5, 1, 0).to(tl.uint8)
        mag += tl.where(y > 5.0, 1, 0).to(tl.uint8)
        sign = tl.where(vals < 0.0, 8, 0).to(tl.uint8)
        return mag | sign

    @triton.jit
    def _fp4_e2m1_decode(code):
        mag_code = code & 7
        mag = tl.where(mag_code == 0, 0.0,
              tl.where(mag_code == 1, 0.5,
              tl.where(mag_code == 2, 1.0,
              tl.where(mag_code == 3, 1.5,
              tl.where(mag_code == 4, 2.0,
              tl.where(mag_code == 5, 3.0,
              tl.where(mag_code == 6, 4.0, 6.0)))))))
        sign = tl.where((code & 8) != 0, -1.0, 1.0)
        return mag * sign

    @triton.jit
    def _quantize_i8_row_kernel(x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr,
                                block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        vals = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.maximum(tl.max(tl.abs(vals), axis=0) / 127.0, 1.0e-8)
        qf = tl.floor(vals / scale + 0.5)
        qf = tl.minimum(tl.maximum(qf, -127.0), 127.0) + 128.0
        tl.store(q_ptr + row * n_cols + offs, qf.to(tl.uint8), mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _dequant_sum_i8_row_kernel(q0_ptr, q1_ptr, q2_ptr, s0_ptr, s1_ptr,
                                   s2_ptr, out_ptr, n_cols: tl.constexpr,
                                   block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        s0 = tl.load(s0_ptr + row)
        s1 = tl.load(s1_ptr + row)
        s2 = tl.load(s2_ptr + row)
        q0 = tl.load(q0_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        q1 = tl.load(q1_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        q2 = tl.load(q2_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        vals = (q0 - 128.0) * s0 + (q1 - 128.0) * s1 + (q2 - 128.0) * s2
        tl.store(out_ptr + row * n_cols + offs, vals, mask=mask)

    @triton.jit
    def _quantize_i8_tensor_kernel(x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr,
                                   block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        vals = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.load(s_ptr)
        qf = tl.floor(vals / scale + 0.5)
        qf = tl.minimum(tl.maximum(qf, -127.0), 127.0) + 128.0
        tl.store(q_ptr + row * n_cols + offs, qf.to(tl.uint8), mask=mask)

    @triton.jit
    def _dequant_sum_i8_tensor_kernel(
        q0_ptr, q1_ptr, q2_ptr, s0_ptr, s1_ptr, s2_ptr, out_ptr,
        n_cols: tl.constexpr, block: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        s0 = tl.load(s0_ptr)
        s1 = tl.load(s1_ptr)
        s2 = tl.load(s2_ptr)
        q0 = tl.load(q0_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        q1 = tl.load(q1_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        q2 = tl.load(q2_ptr + row * n_cols + offs, mask=mask,
                     other=128).to(tl.float32)
        vals = (q0 - 128.0) * s0 + (q1 - 128.0) * s1 + (q2 - 128.0) * s2
        tl.store(out_ptr + row * n_cols + offs, vals, mask=mask)

    @triton.jit
    def _row_i8_scale_kernel(x_ptr, s_ptr, n_cols: tl.constexpr,
                             block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        vals = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.maximum(tl.max(tl.abs(vals), axis=0) / 127.0, 1.0e-8)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _quantize_i8_shared_row_to_i32_kernel(
        x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr, block: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        vals = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.load(s_ptr + row)
        qf = tl.floor(vals / scale + 0.5)
        qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
        tl.store(q_ptr + row * n_cols + offs, qf.to(tl.int32), mask=mask)

    @triton.jit
    def _dequant_i32_shared_row_kernel(
        q_ptr, s_ptr, out_ptr, n_cols: tl.constexpr, block: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        scale = tl.load(s_ptr + row)
        q = tl.load(q_ptr + row * n_cols + offs, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + row * n_cols + offs, q * scale, mask=mask)

    @triton.jit
    def _quantize_fp4_e2m1_block_kernel(
        x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr, block_size: tl.constexpr
    ):
        row = tl.program_id(0)
        block_id = tl.program_id(1)
        offs = tl.arange(0, block_size)
        col = block_id * block_size + offs
        mask = col < n_cols
        vals = tl.load(x_ptr + row * n_cols + col, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.maximum(tl.max(tl.abs(vals), axis=0) / 6.0, 1.0e-8)
        code = _fp4_e2m1_code(vals / scale)
        tl.store(q_ptr + row * n_cols + col, code, mask=mask)
        tl.store(s_ptr + row * tl.cdiv(n_cols, block_size) + block_id, scale)

    @triton.jit
    def _dequant_sum_fp4_e2m1_block_kernel(
        q0_ptr, q1_ptr, q2_ptr, s0_ptr, s1_ptr, s2_ptr, out_ptr,
        n_cols: tl.constexpr, n_blocks: tl.constexpr, block_size: tl.constexpr
    ):
        row = tl.program_id(0)
        block_id = tl.program_id(1)
        offs = tl.arange(0, block_size)
        col = block_id * block_size + offs
        mask = col < n_cols
        s_idx = row * n_blocks + block_id
        s0 = tl.load(s0_ptr + s_idx)
        s1 = tl.load(s1_ptr + s_idx)
        s2 = tl.load(s2_ptr + s_idx)
        q0 = tl.load(q0_ptr + row * n_cols + col, mask=mask, other=0)
        q1 = tl.load(q1_ptr + row * n_cols + col, mask=mask, other=0)
        q2 = tl.load(q2_ptr + row * n_cols + col, mask=mask, other=0)
        vals = (
            _fp4_e2m1_decode(q0) * s0
            + _fp4_e2m1_decode(q1) * s1
            + _fp4_e2m1_decode(q2) * s2
        )
        tl.store(out_ptr + row * n_cols + col, vals, mask=mask)

    @triton.jit
    def _quantize_fp4_e2m1_block_packed_kernel(
        x_ptr, q_ptr, s_ptr, n_cols: tl.constexpr, n_blocks: tl.constexpr,
        block_size: tl.constexpr, pack_size: tl.constexpr
    ):
        row = tl.program_id(0)
        block_id = tl.program_id(1)
        offs = tl.arange(0, block_size)
        col = block_id * block_size + offs
        vals = tl.load(x_ptr + row * n_cols + col).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(vals), axis=0) / 6.0, 1.0e-8)
        pack_offs = tl.arange(0, pack_size)
        lo_vals = tl.load(
            x_ptr + row * n_cols + block_id * block_size + pack_offs * 2
        ).to(tl.float32)
        hi_vals = tl.load(
            x_ptr + row * n_cols + block_id * block_size + pack_offs * 2 + 1
        ).to(tl.float32)
        lo_code = _fp4_e2m1_code(lo_vals / scale)
        hi_code = _fp4_e2m1_code(hi_vals / scale)
        packed = (lo_code & 15) | (hi_code << 4)
        q_base = (row * n_blocks + block_id) * pack_size
        tl.store(q_ptr + q_base + pack_offs, packed)
        tl.store(s_ptr + row * n_blocks + block_id, scale)

    @triton.jit
    def _dequant_sum_fp4_e2m1_block_packed_kernel(
        q0_ptr, q1_ptr, q2_ptr, s0_ptr, s1_ptr, s2_ptr, out_ptr,
        n_cols: tl.constexpr, n_blocks: tl.constexpr, block_size: tl.constexpr,
        pack_size: tl.constexpr
    ):
        row = tl.program_id(0)
        block_id = tl.program_id(1)
        pack_offs = tl.arange(0, pack_size)
        q_base = (row * n_blocks + block_id) * pack_size
        p0 = tl.load(q0_ptr + q_base + pack_offs)
        p1 = tl.load(q1_ptr + q_base + pack_offs)
        p2 = tl.load(q2_ptr + q_base + pack_offs)
        s_idx = row * n_blocks + block_id
        s0 = tl.load(s0_ptr + s_idx).to(tl.float32)
        s1 = tl.load(s1_ptr + s_idx).to(tl.float32)
        s2 = tl.load(s2_ptr + s_idx).to(tl.float32)
        lo_vals = (
            _fp4_e2m1_decode(p0 & 15) * s0
            + _fp4_e2m1_decode(p1 & 15) * s1
            + _fp4_e2m1_decode(p2 & 15) * s2
        )
        hi_vals = (
            _fp4_e2m1_decode(p0 >> 4) * s0
            + _fp4_e2m1_decode(p1 >> 4) * s1
            + _fp4_e2m1_decode(p2 >> 4) * s2
        )
        col = block_id * block_size + pack_offs * 2
        tl.store(out_ptr + row * n_cols + col, lo_vals)
        tl.store(out_ptr + row * n_cols + col + 1, hi_vals)


def should_use_tp3_int8_hidden_reduce(
    input_: torch.Tensor,
    world_size: int,
    min_tokens: int,
) -> bool:
    return (
        triton is not None
        and world_size == 3
        and input_.is_cuda
        and input_.is_contiguous()
        and input_.dim() == 2
        and input_.shape[0] >= min_tokens
        and input_.dtype in (torch.bfloat16, torch.float16)
    )


def should_use_tp3_lowbit_hidden_reduce(
    input_: torch.Tensor,
    world_size: int,
    min_tokens: int,
    mode: str,
) -> bool:
    return (
        mode in (
            "int8",
            "int8_tensor",
            "fp4_block16",
            "fp4_block16_packed",
            "int8_shared_row",
            "int8_bf16pack",
        )
        and should_use_tp3_int8_hidden_reduce(input_, world_size, min_tokens)
    )


def _trace_enabled(input_: torch.Tensor) -> tuple[bool, bool]:
    enabled = (
        os.environ.get("AG2_VLLM_LOWBIT_REDUCE_TRACE") == "1"
        and input_.shape[0]
        >= int(os.environ.get("AG2_VLLM_LOWBIT_REDUCE_TRACE_MIN_TOKENS", "4096"))
    )
    sync = os.environ.get("AG2_VLLM_LOWBIT_REDUCE_TRACE_SYNC") == "1"
    return enabled, sync


def _mark_trace(enabled: bool, sync: bool, device: torch.device) -> float:
    if enabled and sync:
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _int8_workspace(
    device: torch.device,
    world_size: int,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    q_nbytes = rows * cols
    s_nbytes = rows * 4
    total = q_nbytes + s_nbytes
    key = (device_index, world_size, rows, cols)
    cached = _INT8_WORKSPACES.get(key)
    if cached is not None and cached[0].numel() == total:
        return cached
    send = torch.empty(total, device=device, dtype=torch.uint8)
    gathered = torch.empty((world_size, total), device=device, dtype=torch.uint8)
    _INT8_WORKSPACES[key] = (send, gathered)
    return send, gathered


def _int8_bf16pack_workspace(
    device: torch.device,
    world_size: int,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    q_nbytes = rows * cols
    s_nbytes = rows * 4
    total_nbytes = q_nbytes + s_nbytes
    packed_numel = (total_nbytes + 1) // 2
    key = (device_index, world_size, rows, cols)
    cached = _INT8_BF16PACK_WORKSPACES.get(key)
    if cached is not None and cached[0].numel() == packed_numel:
        return cached
    send = torch.empty(packed_numel, device=device, dtype=torch.bfloat16)
    gathered = torch.empty((world_size, packed_numel), device=device,
                           dtype=torch.bfloat16)
    cached = (send, gathered, total_nbytes)
    _INT8_BF16PACK_WORKSPACES[key] = cached
    return cached


def _fp4_workspace(
    device: torch.device,
    world_size: int,
    rows: int,
    cols: int,
    n_blocks: int,
    pack_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    q_nbytes = rows * n_blocks * pack_size
    s_nbytes = rows * n_blocks * 2
    total = q_nbytes + s_nbytes
    key = (device_index, world_size, rows, cols)
    cached = _FP4_WORKSPACES.get(key)
    if cached is not None and cached[0].numel() == total:
        return cached
    send = torch.empty(total, device=device, dtype=torch.uint8)
    gathered = torch.empty((world_size, total), device=device, dtype=torch.uint8)
    _FP4_WORKSPACES[key] = (send, gathered)
    return send, gathered


def _shared_row_workspace(
    device: torch.device,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, rows, cols)
    cached = _SHARED_ROW_WORKSPACES.get(key)
    if cached is not None:
        return cached
    q = torch.empty((rows, cols), device=device, dtype=torch.int32)
    scale = torch.empty((rows,), device=device, dtype=torch.float32)
    _SHARED_ROW_WORKSPACES[key] = (q, scale)
    return q, scale


def tp3_int8_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q_nbytes = rows * cols
    send, gathered = _int8_workspace(input_.device, world_size, rows, cols)
    q_local = send[:q_nbytes].view(rows, cols)
    s_local = send[q_nbytes:].view(torch.float32)
    block = triton.next_power_of_2(cols)
    _quantize_i8_row_kernel[(rows,)](input_, q_local, s_local, cols, block)
    t1 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_gather_into_tensor(gathered, send, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    q_gathered = gathered[:, :q_nbytes].view(world_size, rows, cols)
    s_gathered = gathered[:, q_nbytes:].view(torch.float32).view(world_size, rows)
    _dequant_sum_i8_row_kernel[(rows,)](
        q_gathered[0], q_gathered[1], q_gathered[2],
        s_gathered[0], s_gathered[1], s_gathered[2],
        out, cols, block,
    )
    t3 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=int8 tokens=%d hidden=%d "
            "quant=%.6f gather=%.6f dequant=%.6f total=%.6f payload_mib=%.3f",
            rows, cols, t1 - t0, t2 - t1, t3 - t2, t3 - t0, send.numel() / 2**20,
        )
    return out


def tp3_int8_bf16pack_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Approximate SUM all-reduce with int8 bytes transported as BF16.

    This does not use BF16 arithmetic. The BF16 tensor is only a 16-bit NCCL
    transport container for the existing int8 codes plus fp32 row scales.
    """
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q_nbytes = rows * cols
    send_bf16, gathered_bf16, total_nbytes = _int8_bf16pack_workspace(
        input_.device, world_size, rows, cols
    )
    send = send_bf16.view(torch.uint8)[:total_nbytes]
    gathered = gathered_bf16.view(torch.uint8).view(world_size, -1)[:, :total_nbytes]
    q_local = send[:q_nbytes].view(rows, cols)
    s_local = send[q_nbytes:].view(torch.float32)
    block = triton.next_power_of_2(cols)
    _quantize_i8_row_kernel[(rows,)](input_, q_local, s_local, cols, block)
    t1 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_gather_into_tensor(gathered_bf16, send_bf16, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    q_gathered = gathered[:, :q_nbytes].view(world_size, rows, cols)
    s_gathered = gathered[:, q_nbytes:].view(torch.float32).view(world_size, rows)
    _dequant_sum_i8_row_kernel[(rows,)](
        q_gathered[0], q_gathered[1], q_gathered[2],
        s_gathered[0], s_gathered[1], s_gathered[2],
        out, cols, block,
    )
    t3 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=int8_bf16pack tokens=%d hidden=%d "
            "quant=%.6f gather=%.6f dequant=%.6f total=%.6f payload_mib=%.3f",
            rows,
            cols,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t3 - t0,
            send_bf16.numel() * send_bf16.element_size() / 2**20,
        )
    return out


def tp3_int8_tensor_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Approximate SUM all-reduce via per-tensor int8 all-gather."""
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q_nbytes = rows * cols
    total = q_nbytes + 4
    send = torch.empty(total, device=input_.device, dtype=torch.uint8)
    gathered = torch.empty((world_size, total), device=input_.device, dtype=torch.uint8)
    q_local = send[:q_nbytes].view(rows, cols)
    s_local = send[q_nbytes:].view(torch.float32)
    scale = input_.abs().amax().clamp_min(1.0e-8) / 127.0
    s_local.copy_(scale.reshape(()))
    block = triton.next_power_of_2(cols)
    _quantize_i8_tensor_kernel[(rows,)](input_, q_local, s_local, cols, block)
    t1 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_gather_into_tensor(gathered, send, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    q_gathered = gathered[:, :q_nbytes].view(world_size, rows, cols)
    s_gathered = gathered[:, q_nbytes:].view(torch.float32).view(world_size)
    _dequant_sum_i8_tensor_kernel[(rows,)](
        q_gathered[0], q_gathered[1], q_gathered[2],
        s_gathered[0:1], s_gathered[1:2], s_gathered[2:3],
        out, cols, block,
    )
    t3 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=int8_tensor tokens=%d hidden=%d "
            "quant=%.6f gather=%.6f dequant=%.6f total=%.6f payload_mib=%.3f",
            rows, cols, t1 - t0, t2 - t1, t3 - t2, t3 - t0, send.numel() / 2**20,
        )
    return out


def tp3_fp4_block16_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    block_size = 16
    n_blocks = triton.cdiv(cols, block_size)
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q_local = torch.empty(input_.shape, device=input_.device, dtype=torch.uint8)
    s_local = torch.empty((rows, n_blocks), device=input_.device, dtype=torch.float32)
    _quantize_fp4_e2m1_block_kernel[(rows, n_blocks)](
        input_, q_local, s_local, cols, block_size
    )
    t1 = _mark_trace(trace, trace_sync, input_.device)
    q_gathered = torch.empty(
        (world_size, rows, cols), device=input_.device, dtype=torch.uint8
    )
    s_gathered = torch.empty(
        (world_size, rows, n_blocks), device=input_.device, dtype=torch.float32
    )
    dist.all_gather_into_tensor(q_gathered, q_local, group=group)
    dist.all_gather_into_tensor(s_gathered, s_local, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    _dequant_sum_fp4_e2m1_block_kernel[(rows, n_blocks)](
        q_gathered[0], q_gathered[1], q_gathered[2],
        s_gathered[0], s_gathered[1], s_gathered[2],
        out, cols, n_blocks, block_size,
    )
    t3 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=fp4_block16 tokens=%d hidden=%d "
            "quant=%.6f gather=%.6f dequant=%.6f total=%.6f",
            rows, cols, t1 - t0, t2 - t1, t3 - t2, t3 - t0,
        )
    return out


def tp3_fp4_block16_packed_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    block_size = 16
    pack_size = block_size // 2
    assert cols % block_size == 0
    n_blocks = triton.cdiv(cols, block_size)
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q_nbytes = rows * n_blocks * pack_size
    s_numel = rows * n_blocks
    send, gathered = _fp4_workspace(
        input_.device, world_size, rows, cols, n_blocks, pack_size
    )
    q_local = send[:q_nbytes]
    s_local = send[q_nbytes:].view(torch.float16)
    _quantize_fp4_e2m1_block_packed_kernel[(rows, n_blocks)](
        input_, q_local, s_local, cols, n_blocks, block_size, pack_size
    )
    t1 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_gather_into_tensor(gathered, send, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    q_gathered = gathered[:, :q_nbytes]
    s_gathered = gathered[:, q_nbytes:].view(torch.float16).view(world_size, s_numel)
    _dequant_sum_fp4_e2m1_block_packed_kernel[(rows, n_blocks)](
        q_gathered[0], q_gathered[1], q_gathered[2],
        s_gathered[0], s_gathered[1], s_gathered[2],
        out, cols, n_blocks, block_size, pack_size,
    )
    t3 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=fp4_block16_packed tokens=%d "
            "hidden=%d quant=%.6f gather=%.6f dequant=%.6f total=%.6f "
            "payload_mib=%.3f",
            rows, cols, t1 - t0, t2 - t1, t3 - t2, t3 - t0, send.numel() / 2**20,
        )
    return out


def tp3_int8_shared_row_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
) -> torch.Tensor:
    """Approximate SUM all-reduce using shared row scales and int32 code sum.

    This POC trades the int8 all-gather's three gathered rank buffers for two
    all-reduces: a small per-row max scale and an int32 sum of quantized codes.
    NCCL in the current PyTorch build rejects int16 all-reduce, so int32 is the
    runnable schedule probe even though its payload model is poor.
    """
    assert triton is not None
    world_size = dist.get_world_size(group)
    assert world_size == 3
    rows, cols = input_.shape
    trace, trace_sync = _trace_enabled(input_)
    t0 = _mark_trace(trace, trace_sync, input_.device)
    q, scale = _shared_row_workspace(input_.device, rows, cols)
    block = triton.next_power_of_2(cols)
    _row_i8_scale_kernel[(rows,)](input_, scale, cols, block)
    t1 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_reduce(scale, op=dist.ReduceOp.MAX, group=group)
    t2 = _mark_trace(trace, trace_sync, input_.device)
    _quantize_i8_shared_row_to_i32_kernel[(rows,)](input_, q, scale, cols, block)
    t3 = _mark_trace(trace, trace_sync, input_.device)
    dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group)
    t4 = _mark_trace(trace, trace_sync, input_.device)
    out = torch.empty_like(input_)
    _dequant_i32_shared_row_kernel[(rows,)](q, scale, out, cols, block)
    t5 = _mark_trace(trace, trace_sync, input_.device)
    if trace:
        logger.warning(
            "AG2_LOWBIT_REDUCE_TRACE mode=int8_shared_row tokens=%d hidden=%d "
            "scale=%.6f scale_ar=%.6f quant=%.6f q_ar=%.6f dequant=%.6f "
            "total=%.6f payload_mib=%.3f",
            rows,
            cols,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
            t5 - t0,
            (q.numel() * q.element_size() + scale.numel() * scale.element_size())
            / 2**20,
        )
    return out


def tp3_lowbit_hidden_all_reduce(
    input_: torch.Tensor,
    group: ProcessGroup,
    mode: str,
) -> torch.Tensor:
    if mode == "int8_bf16pack":
        return tp3_int8_bf16pack_hidden_all_reduce(input_, group)
    if mode == "int8_shared_row":
        return tp3_int8_shared_row_hidden_all_reduce(input_, group)
    if mode == "int8_tensor":
        return tp3_int8_tensor_hidden_all_reduce(input_, group)
    if mode == "fp4_block16_packed":
        return tp3_fp4_block16_packed_hidden_all_reduce(input_, group)
    if mode == "fp4_block16":
        return tp3_fp4_block16_hidden_all_reduce(input_, group)
    return tp3_int8_hidden_all_reduce(input_, group)
