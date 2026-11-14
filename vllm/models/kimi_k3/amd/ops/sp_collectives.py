# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce-scatter / all-gather over symmetric memory for Kimi-K3's
sequence-parallel prefill on ROCm (TP8 MI355X, xGMI).

Rank r owns rows [r*R, (r+1)*R) of a [T, N] tensor (R = T / world):

  rs_rows(x [T, N])     -> [R, N]     own rows of the sum over ranks
  rs_cols(x [T, N])     -> [T, N/W]   own column block of the sum, plus the
                                      per-row sum of squares of that block
  ag_rows(x_own [R, N]) -> [T, N]     every owner's rows
  allreduce_vec(v [T])  -> [T]        fp32 vector all-reduce

Reduce-scatters push (remote stores) from an ordinary local tensor into each
owner's slot of its symmetric buffer, then each owner sums its slots locally in
fp32 in a fixed order, so producers need not write into symmetric memory and
every sum is independent of the schedule. All-gathers copy the own rows into
the local symmetric buffer and pull every owner's rows into ordinary memory:
pulling disturbs concurrent compute less than pushing, and storing into the
symmetric buffer from the local GPU is slow. Each collective is a barrier
(every peer is done with the buffer's previous contents), the copy kernel on a
fixed number of workgroups (the rest of the GPU stays free for overlapped
compute), a barrier, and the local reduction. Buffers are per name and hold
max_rows x max_width elements; callers pick names so that one buffer is never
used by two collectives at once.
"""

import torch
import torch.distributed._symmetric_memory as symm_mem

from vllm.triton_utils import tl, triton

BR = 8


@triton.jit
def _peer(ptrs, p, ELEM: tl.constexpr):
    return tl.load(ptrs + p).to(tl.pointer_type(ELEM))


@triton.jit
def _push_rs_rows(
    src_ptr,
    src_stride,
    src_rows,
    dst_ptrs,
    dst_base,
    R,
    N,
    n_tiles,
    rank,
    WORLD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """dst[o] rows dst_base .. = src rows of owner o (src rows o*R .., rows
    past src_rows read as 0); consecutive tiles go to different owners."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_ct = N // BC
    for t in range(pid, n_tiles, nprog):
        o = (rank + 1 + t) % WORLD
        tt = t // WORLD
        rt = tt // n_ct
        ct = tt % n_ct
        rr = rt * BR + tl.arange(0, BR)
        c = ct * BC + tl.arange(0, BC)
        g = o * R + rr
        v = tl.load(
            src_ptr + g[:, None].to(tl.int64) * src_stride + c[None, :],
            mask=(g < src_rows)[:, None],
            other=0.0,
        )
        tl.store(
            _peer(dst_ptrs, o, tl.bfloat16)
            + (dst_base + rr)[:, None].to(tl.int64) * N
            + c[None, :],
            v,
        )


@triton.jit
def _push_rs_cols(
    src_ptr,
    src_stride,
    dst_ptrs,
    rank,
    T,
    CW,
    n_tiles,
    WORLD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """dst[o] slot `rank` ([T, CW] at rank*T) = src[:, o*CW:(o+1)*CW]; one
    tile spans the whole column block (BC >= CW)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    c = tl.arange(0, BC)
    cm = (c < CW)[None, :]
    for t in range(pid, n_tiles, nprog):
        o = (rank + 1 + t) % WORLD
        rr = (t // WORLD) * BR + tl.arange(0, BR)
        v = tl.load(
            src_ptr + rr[:, None].to(tl.int64) * src_stride + (o * CW + c)[None, :],
            mask=cm,
        )
        tl.store(
            _peer(dst_ptrs, o, tl.bfloat16)
            + (rank * T + rr)[:, None].to(tl.int64) * CW
            + c[None, :],
            v,
            mask=cm,
        )


@triton.jit
def _sum_slots(
    buf_ptr,
    out_ptr,
    ss_ptr,
    rows,
    N,
    slot_stride,
    WORLD: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    SUMSQ: tl.constexpr,
):
    """out[i, :] = sum over slots s of buf[s*slot_stride + i*N + :] in fp32;
    SUMSQ (one column tile only) also writes sum(out^2) per row."""
    rt = tl.program_id(0)
    rr = rt * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rr[:, None] < rows) & (c[None, :] < N)
    off = rr[:, None].to(tl.int64) * N + c[None, :]
    acc = tl.zeros([BR, BN], tl.float32)
    for s in tl.static_range(WORLD):
        acc += tl.load(buf_ptr + s * slot_stride + off, mask=m, other=0.0).to(
            tl.float32
        )
    y = acc.to(tl.bfloat16)
    tl.store(out_ptr + off, y, mask=m)
    if SUMSQ:
        yf = y.to(tl.float32)
        tl.store(ss_ptr + rr, tl.sum(yf * yf, axis=1), mask=rr < rows)


@triton.jit
def _pull_ag_rows(
    buf_ptrs,
    out_ptr,
    rank,
    R,
    Cr,
    src0,
    dst0,
    C,
    N,
    n_tiles,
    WORLD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """Rows dst0 + o*C .. of out (C rows per owner o) = o's buffer rows
    o*R + src0 ..; consecutive tiles read different owners. Whole tensor:
    C = R, src0 = dst0 = 0."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    n_ct = N // BC
    for t in range(pid, n_tiles, nprog):
        o = (rank + 1 + t) % WORLD
        tt = t // WORLD
        rt = tt // n_ct
        ct = tt % n_ct
        rr = rt * BR + tl.arange(0, BR)
        c = ct * BC + tl.arange(0, BC)
        src = (o * R + src0 + rr)[:, None].to(tl.int64) * N + c[None, :]
        dst = (dst0 + o * C + rr)[:, None].to(tl.int64) * N + c[None, :]
        tl.store(out_ptr + dst, tl.load(_peer(buf_ptrs, o, tl.bfloat16) + src))


@triton.jit
def _copy_rows(
    src_ptr, src_stride, dst_ptr, rows, N, BR: tl.constexpr, BC: tl.constexpr
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = (r < rows)[:, None] & (c < N)[None, :]
    v = tl.load(src_ptr + r[:, None].to(tl.int64) * src_stride + c[None, :], mask=m)
    tl.store(dst_ptr + r[:, None].to(tl.int64) * N + c[None, :], v, mask=m)


@triton.jit
def _push_vec(src_ptr, dst_ptrs, rank, T, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    """dst[p] slot rank ([T] fp32 at rank*T) = src, for every p."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src_ptr + i, mask=i < T)
    for p in tl.static_range(WORLD):
        tl.store(_peer(dst_ptrs, p, tl.float32) + rank * T + i, v, mask=i < T)


@triton.jit
def _sum_vec(buf_ptr, out_ptr, T, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for s in tl.static_range(WORLD):
        acc += tl.load(buf_ptr + s * T + i, mask=i < T, other=0.0)
    tl.store(out_ptr + i, acc, mask=i < T)


def _bc(n):
    for bc in (512, 256, 128, 64):
        if n % bc == 0:
            return bc
    raise ValueError(f"width {n} is not a multiple of 64")


class _Buf:
    def __init__(self, numel, dtype, dev, group_name):
        self.t = symm_mem.empty((numel,), device=dev, dtype=dtype)
        self.h = symm_mem.rendezvous(self.t, group_name)
        self.ptrs = torch.tensor(self.h.buffer_ptrs, device=dev, dtype=torch.int64)


class SPCollectives:
    """Symmetric buffers are per name, each max_rows * max_width bf16 elements
    (max_rows * world fp32 for allreduce_vec)."""

    def __init__(self, group_name, rank, world, max_rows, dev, max_width=7168):
        self.group_name, self.rank, self.world, self.dev = group_name, rank, world, dev
        self.max_rows, self.max_width = max_rows, max_width
        self.bufs = {}

    def _buf(self, name, numel, dtype=torch.bfloat16):
        b = self.bufs.get(name)
        if b is None:
            cap = self.max_rows * (
                self.max_width if dtype == torch.bfloat16 else self.world
            )
            b = self.bufs[name] = _Buf(cap, dtype, self.dev, self.group_name)
        assert b.t.numel() >= numel and b.t.dtype == dtype, (name, numel, b.t.numel())
        return b

    def _barrier(self, b):
        b.h.barrier(channel=0)

    def rs_rows(self, x, name, ctas, rows=None, out=None):
        """x: [T, N] partial, any local tensor; T may be below `rows` (the
        padded row count), the missing rows count as 0."""
        N = x.shape[1]
        rows = x.shape[0] if rows is None else rows
        R = rows // self.world
        assert rows % (self.world * BR) == 0 and x.stride(1) == 1 and x.shape[0] <= rows
        b = self._buf(name, rows * N)
        bc = _bc(N)
        self._barrier(b)
        _push_rs_rows[(ctas,)](
            x,
            x.stride(0),
            x.shape[0],
            b.ptrs,
            self.rank * R,
            R,
            N,
            (R // BR) * (N // bc) * self.world,
            self.rank,
            WORLD=self.world,
            BR=BR,
            BC=bc,
            num_warps=4,
        )
        self._barrier(b)
        out = (
            torch.empty(R, N, device=self.dev, dtype=torch.bfloat16)
            if out is None
            else out
        )
        _sum_slots[(R // BR, triton.cdiv(N, 512))](
            b.t,
            out,
            out,
            R,
            N,
            R * N,
            WORLD=self.world,
            BR=BR,
            BN=512,
            SUMSQ=False,
            num_warps=4,
        )
        return out

    def rs_cols(self, x, name, ctas, out=None, ss=None):
        """-> (own column block [T, N / world] of the sum, fp32 per-row sum of
        squares of that block)."""
        T, N = x.shape
        CW = N // self.world
        assert T % BR == 0 and x.stride(1) == 1 and CW <= 1024
        b = self._buf(name, T * N)
        self._barrier(b)
        _push_rs_cols[(ctas,)](
            x,
            x.stride(0),
            b.ptrs,
            self.rank,
            T,
            CW,
            (T // BR) * self.world,
            WORLD=self.world,
            BR=BR,
            BC=triton.next_power_of_2(CW),
            num_warps=4,
        )
        self._barrier(b)
        out = (
            torch.empty(T, CW, device=self.dev, dtype=torch.bfloat16)
            if out is None
            else out
        )
        ss = torch.empty(T, device=self.dev, dtype=torch.float32) if ss is None else ss
        _sum_slots[(T // BR, 1)](
            b.t,
            out,
            ss,
            T,
            CW,
            T * CW,
            WORLD=self.world,
            BR=BR,
            BN=triton.next_power_of_2(CW),
            SUMSQ=True,
            num_warps=4,
        )
        return out, ss

    def ag_rows(self, x_own, name, ctas, out=None):
        """-> out [T, N], every owner's rows: own rows are copied into the
        local symmetric buffer, then every rank pulls all owners' rows into
        `out` (ordinary memory, which concurrent compute tolerates better than
        pushes, and stores into the symmetric buffer are slow)."""
        R, N = x_own.shape
        assert R % BR == 0 and x_own.stride(1) == 1
        T = R * self.world
        b = self._buf(name, T * N)
        full = b.t[: T * N].view(T, N)
        bc = _bc(N)
        out = (
            torch.empty(T, N, device=self.dev, dtype=torch.bfloat16)
            if out is None
            else out
        )
        self._barrier(b)  # every peer is done reading the previous contents
        self._stage(x_own, full[self.rank * R : (self.rank + 1) * R])
        self._barrier(b)  # every owner's rows are in place
        _pull_ag_rows[(ctas,)](
            b.ptrs,
            out,
            self.rank,
            R,
            R,
            0,
            0,
            R,
            N,
            (R // BR) * (N // bc) * self.world,
            WORLD=self.world,
            BR=BR,
            BC=bc,
            num_warps=4,
        )
        return out

    @staticmethod
    def _stage(x, dst):
        rows, N = x.shape
        _copy_rows[(triton.cdiv(rows, 8), triton.cdiv(N, 512))](
            x, x.stride(0), dst, rows, N, BR=8, BC=512, num_warps=4
        )

    # Block-cyclic chunks: the batch is k chunks of C = T / k rows; rank r owns
    # rows [j C + r Cr, j C + (r+1) Cr) of chunk j (Cr = C / world), stored
    # locally as rows [j Cr, (j+1) Cr) of an [R, N] tensor.

    def rs_chunk(self, x, name, ctas, j, k, R, out):
        """x: chunk j's [<= C, N] partial in token order -> out[j Cr:(j+1) Cr]
        (own rows of the sum). Chunk 0 also waits until every peer is done with
        the buffer's previous contents."""
        N = x.shape[1]
        Cr = R // k
        assert x.stride(1) == 1 and Cr % BR == 0
        b = self._buf(name, R * self.world * N)
        bc = _bc(N)
        if j == 0:
            self._barrier(b)
        _push_rs_rows[(ctas,)](
            x,
            x.stride(0),
            x.shape[0],
            b.ptrs,
            self.rank * R + j * Cr,
            Cr,
            N,
            (Cr // BR) * (N // bc) * self.world,
            self.rank,
            WORLD=self.world,
            BR=BR,
            BC=bc,
            num_warps=4,
        )
        self._barrier(b)
        _sum_slots[(Cr // BR, triton.cdiv(N, 512))](
            b.t[j * Cr * N :],
            out[j * Cr : (j + 1) * Cr],
            out,
            Cr,
            N,
            R * N,
            WORLD=self.world,
            BR=BR,
            BN=512,
            SUMSQ=False,
            num_warps=4,
        )

    def ag_stage(self, x_own, name):
        """Own rows [R, N] into the local buffer for ag_pull (one barrier each side)."""
        R, N = x_own.shape
        b = self._buf(name, R * self.world * N)
        self._barrier(b)
        self._stage(x_own, b.t[self.rank * R * N : (self.rank + 1) * R * N].view(R, N))
        self._barrier(b)

    def ag_pull(self, name, out, j, k, R, ctas):
        """out[j C:(j+1) C] (token order) = every owner's local rows
        [j Cr, (j+1) Cr) staged by ag_stage."""
        N = out.shape[1]
        Cr, C = R // k, R // k * self.world
        b = self._buf(name, R * self.world * N)
        bc = _bc(N)
        _pull_ag_rows[(ctas,)](
            b.ptrs,
            out,
            self.rank,
            R,
            Cr,
            j * Cr,
            j * C,
            Cr,
            N,
            (Cr // BR) * (N // bc) * self.world,
            WORLD=self.world,
            BR=BR,
            BC=bc,
            num_warps=4,
        )

    def allreduce_vec(self, v, name, out=None):
        T = v.numel()
        b = self._buf(name, T * self.world, torch.float32)
        self._barrier(b)
        _push_vec[(triton.cdiv(T, 1024),)](
            v, b.ptrs, self.rank, T, WORLD=self.world, BLOCK=1024, num_warps=4
        )
        self._barrier(b)
        out = torch.empty_like(v) if out is None else out
        _sum_vec[(triton.cdiv(T, 1024),)](
            b.t, out, T, WORLD=self.world, BLOCK=1024, num_warps=4
        )
        return out
