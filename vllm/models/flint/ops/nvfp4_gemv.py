# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M BF16 x NVFP4 (W4A16) and BF16 x BF16 GEMM for SM100 in CuTe DSL.

``out[M, N] = (x[M, K] @ dequant(W)[N, K]^T) * global_scale`` with the weight in
the ModelOpt layout used by the FlashInfer CuTe-DSL W4A16 kernel: ``weight``
uint8 ``[N, K / 2]`` (two E2M1 codes per byte, low nibble first) and
``weight_scale`` the 128x4-swizzled E4M3 block scales (one per 16 values), so
both kernels share the same tensors.

Numerics: every weight is dequantized exactly to BF16 (``e2m1 * e4m3`` via
``cvt.rn.bf16x2.e2m1x2`` and ``mul.rn.bf16x2``), multiplied with the BF16
activations on tensor cores (``mma.sync`` m16n8k16, FP32 accumulate), and the
FP32 global scale is applied to the FP32 accumulator before the single output
rounding to BF16.

Each CTA owns 16 output features (the MMA M dimension, swap-AB) and ``8 * nt``
tokens (``nt`` = 1, 2 or 4, i.e. up to 32); its warps (and, with ``split > 1``,
the CTAs of a cluster) split K and reduce through shared memory / DSMEM. All
weight loads are issued before ``griddepcontrol.wait`` so they overlap the
previous kernel under PDL.
"""

from dataclasses import dataclass
from typing import Any

import torch

_CHUNK = 64  # K values per warp step (four m16n8k16 k-tiles)
_NACC = 8  # accumulator sets per lane, rotated over k-tiles


@dataclass(frozen=True)
class GemvConfig:
    warps: int
    split: int


def _default_config(n: int, k: int, bf16: bool = False) -> GemvConfig:
    chunks = k // _CHUNK
    warps = 8 if chunks % 8 == 0 else 4
    blocks = n // 16
    split = 2 if blocks < 256 and chunks % (2 * warps) == 0 else 1
    if bf16 and chunks % (2 * split * warps) == 0:
        split *= 2  # keep the 4x larger per-lane weight fragments in registers
    return GemvConfig(warps, split)


_C = GemvConfig
_Table = dict[tuple[int, int], dict[int, GemvConfig]]
# Tuned on GB200 (CUDA-graph replay, weights rotated past L2): config per
# number of 8-token MMA tiles per CTA (``nt``). A shape / ``nt`` is listed only
# where the GEMV beats the alternative (FlashInfer CuTe-DSL for NVFP4, cuBLAS
# for BF16) at ``M <= 8 * nt``; the model uses the GEMV only for listed ones.
# Gate/up projections always take the SiLU-and-mul tables.
_TUNED: _Table = {
    (5120, 2048): {1: _C(8, 1), 2: _C(8, 2)},
    (2048, 4096): {1: _C(4, 4), 2: _C(8, 2), 4: _C(4, 4)},
    (2048, 6144): {1: _C(8, 4), 2: _C(8, 2), 4: _C(4, 8)},
    (2048, 1024): {1: _C(4, 1), 2: _C(4, 1), 4: _C(8, 1)},
}
_TUNED_SILU_MUL: _Table = {
    (12288, 2048): {1: _C(4, 1), 2: _C(2, 2)},
    (2048, 2048): {1: _C(2, 4), 2: _C(4, 2), 4: _C(8, 2)},
}
# The BF16 qkv and dense down GEMVs are no faster than cuBLAS.
_TUNED_BF16: _Table = {
    (2048, 4096): {1: _C(4, 8), 2: _C(8, 4)},
    (2048, 1024): {1: _C(4, 4), 2: _C(4, 4)},
}
_TUNED_BF16_SILU_MUL: _Table = {
    (12288, 2048): {1: _C(2, 1)},
    (2048, 2048): {1: _C(8, 4), 2: _C(4, 4), 4: _C(8, 2)},
}


def _table(silu_mul: bool, bf16: bool) -> _Table:
    if bf16:
        return _TUNED_BF16_SILU_MUL if silu_mul else _TUNED_BF16
    return _TUNED_SILU_MUL if silu_mul else _TUNED


def _configs(n: int, k: int, silu_mul: bool, bf16: bool) -> dict[int, GemvConfig]:
    tuned = _table(silu_mul, bf16).get((n, k))
    if tuned is not None:
        return tuned
    cfg = _default_config(n // 2 if silu_mul else n, k, bf16)
    return {1: cfg, 2: cfg}


def tuned(n: int, k: int, silu_mul: bool = False, bf16: bool = False) -> bool:
    """Whether the GEMV is the fastest choice for this shape at some M."""
    return (n, k) in _table(silu_mul, bf16)


def pick_nt(
    n: int, k: int, m: int, silu_mul: bool = False, bf16: bool = False
) -> int | None:
    """Smallest tuned token tile count covering ``m``, or None if the GEMV
    is not the fastest choice for this shape and M."""
    nts = [nt for nt in _table(silu_mul, bf16).get((n, k), ()) if 8 * nt >= m]
    return min(nts) if nts else None


def max_m(n: int, k: int, silu_mul: bool = False, bf16: bool = False) -> int:
    return 8 * max(_configs(n, k, silu_mul, bf16))


def supports(n: int, k: int, silu_mul: bool = False, bf16: bool = False) -> bool:
    if n % 128 or k % _CHUNK or (silu_mul and n % 32):
        return False
    return all(
        (k // _CHUNK) % (cfg.warps * cfg.split) == 0
        for cfg in _configs(n, k, silu_mul, bf16).values()
    )


def _build_kernel_class():
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass import Float32, Int32, Int64, Uint32, const_expr

    from .cute_primitives import (
        decode,
        ld_u8,
        ld_w8,
        ld_w16,
        ld_x16,
        mma,
        round_bf16,
        scale_pair,
        silu_mul,
        st_dsmem_f32,
        st_out,
    )

    class Nvfp4Gemv:
        def __init__(self, k, nt, warps, split, pdl, silu_mul, bf16):
            self.bf16 = bf16
            self.kc = k // _CHUNK
            assert self.kc % (warps * split) == 0
            self.cpw = self.kc // (warps * split)
            self.nt = nt
            self.warps = warps
            self.split = split
            self.pdl = pdl
            # silu_mul: each CTA also owns the matching "up" row block N/2
            # rows below and writes silu(gate) * up.
            self.rb = 2 if silu_mul else 1
            self.nacc = _NACC // self.rb

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            out: cute.Tensor,
            M: Int32,
            N: Int32,
            stride_x: Int32,
            stride_o: Int32,
            stream: CUstream,
        ):
            tokens = 8 * self.nt
            n_out = N // self.rb
            self.kernel(x, w, s, gs, out, M, n_out, stride_x, stride_o).launch(
                grid=[(n_out // 16) * self.split, (M + tokens - 1) // tokens, 1],
                block=[32 * self.warps, 1, 1],
                cluster=[self.split, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

        @cute.kernel
        def kernel(
            self,
            x: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            out: cute.Tensor,
            M: Int32,
            n_out: Int32,
            stride_x: Int32,
            stride_o: Int32,
        ):
            NT: cutlass.Constexpr = self.nt
            CPW: cutlass.Constexpr = self.cpw
            KC: cutlass.Constexpr = self.kc
            WARPS: cutlass.Constexpr = self.warps
            SPLIT: cutlass.Constexpr = self.split
            NACC: cutlass.Constexpr = self.nacc
            RB: cutlass.Constexpr = self.rb
            NA: cutlass.Constexpr = RB * NT * 4  # accumulators per lane
            BF16W: cutlass.Constexpr = self.bf16
            KB: cutlass.Constexpr = self.kc * (128 if self.bf16 else 32)
            QW: cutlass.Constexpr = 16 if self.bf16 else 4  # words per chunk
            tidx, _, _ = cute.arch.thread_idx()
            bx, mt, _ = cute.arch.block_idx()
            nb = bx // SPLIT
            rank = bx % SPLIT
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4
            c_begin = (rank * WARPS + warp) * CPW

            # Lane (g, t) of k-tile j covers k = 64c + 16t + 4j + {0..3} of
            # rows g and g + 8, i.e. bytes 8t + 2j + {0, 1} of the 32-byte
            # chunk row, and scale group 4c + t of both rows.
            qs = cute.make_rmem_tensor((RB * CPW * QW,), Uint32)
            ss = cute.make_rmem_tensor((RB * CPW * 2,), Uint32)
            for rb in cutlass.range_constexpr(RB):
                row_g = nb * 16 + g + rb * n_out
                row_g8 = row_g + 8
                if const_expr(BF16W):
                    # 16 BF16 weights of rows g and g + 8 per chunk.
                    wbase = (
                        w.iterator.toint() + Int64(row_g) * KB + c_begin * 128 + t * 32
                    )
                    for c in cutlass.range_constexpr(CPW):
                        q0 = (rb * CPW + c) * QW
                        for h in cutlass.range_constexpr(4):
                            off = (h // 2) * 8 * KB + c * 128 + (h % 2) * 16
                            wv = ld_w16(wbase + off)
                            for i in cutlass.range_constexpr(4):
                                qs[q0 + h * 4 + i] = wv[i]
                else:
                    wbase = (
                        w.iterator.toint() + Int64(row_g) * KB + c_begin * 32 + t * 8
                    )
                    sbase_g = (
                        s.iterator.toint()
                        + Int64(row_g // 128) * (KC * 512)
                        + (row_g % 32) * 16
                        + ((row_g % 128) // 32) * 4
                        + t
                        + c_begin * 512
                    )
                    sbase_g8 = (
                        s.iterator.toint()
                        + Int64(row_g8 // 128) * (KC * 512)
                        + (row_g8 % 32) * 16
                        + ((row_g8 % 128) // 32) * 4
                        + t
                        + c_begin * 512
                    )
                    for c in cutlass.range_constexpr(CPW):
                        qa = ld_w8(wbase + c * 32)
                        qb = ld_w8(wbase + 8 * KB + c * 32)
                        q0 = (rb * CPW + c) * QW
                        qs[q0 + 0] = qa[0]
                        qs[q0 + 1] = qa[1]
                        qs[q0 + 2] = qb[0]
                        qs[q0 + 3] = qb[1]
                        ss[(rb * CPW + c) * 2] = ld_u8(sbase_g + c * 512)
                        ss[(rb * CPW + c) * 2 + 1] = ld_u8(sbase_g8 + c * 512)
            gsv = Float32(1.0)
            if const_expr(not BF16W):
                gsv = gs[0]
            if const_expr(SPLIT > 1):
                # Peers must be running before DSMEM stores; waited on below.
                cute.arch.cluster_arrive_relaxed()

            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            xbase = x.iterator.toint() + (Int64(c_begin) * _CHUNK + t * 16) * 2
            xall = []
            for c in cutlass.range_constexpr(CPW):
                xs = []
                for nt in cutlass.range_constexpr(NT):
                    tok = mt * (8 * NT) + nt * 8 + g
                    addr = xbase + Int64(tok) * stride_x * 2 + c * 128
                    xs.append(ld_x16(addr, Int32(tok < M)))
                xall.append(xs)

            acc = cute.make_rmem_tensor((NACC * NA,), Float32)
            acc.fill(0.0)
            for rb in cutlass.range_constexpr(RB):
                for c in cutlass.range_constexpr(CPW):
                    q0 = (rb * CPW + c) * QW
                    if const_expr(not BF16W):
                        sg = scale_pair(ss[(rb * CPW + c) * 2])
                        sg8 = scale_pair(ss[(rb * CPW + c) * 2 + 1])
                    for j in cutlass.range_constexpr(4):
                        if const_expr(BF16W):
                            # rows g / g + 8, k pairs (4j, 4j+1) and (4j+2, 4j+3)
                            a = [
                                qs[q0 + 2 * j],
                                qs[q0 + 8 + 2 * j],
                                qs[q0 + 2 * j + 1],
                                qs[q0 + 8 + 2 * j + 1],
                            ]
                        else:
                            a = decode(qs[q0 + j // 2], qs[q0 + 2 + j // 2], j, sg, sg8)
                        for nt in cutlass.range_constexpr(NT):
                            xv = xall[c][nt]
                            ab = ((c * 4 + j) % NACC) * NA + (rb * NT + nt) * 4
                            r = mma(
                                [acc[ab], acc[ab + 1], acc[ab + 2], acc[ab + 3]],
                                a,
                                xv[2 * j],
                                xv[2 * j + 1],
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[ab + i] = r[i]
            for h in cutlass.range_constexpr(1, NACC):
                for i in cutlass.range_constexpr(NA):
                    acc[i] = acc[i] + acc[h * NA + i]
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_launch_dependents()

            smem = cutlass.memory.SmemAllocator()
            red = smem.allocate_tensor(
                Float32,
                cute.make_layout((WARPS, NA, 32), stride=(NA * 32, 32, 1)),
                byte_alignment=16,
            )
            part = smem.allocate_tensor(
                Float32,
                cute.make_layout((SPLIT, NA, 32), stride=(NA * 32, 32, 1)),
                byte_alignment=16,
            )
            if const_expr(WARPS > 1):
                for i in cutlass.range_constexpr(NA):
                    red[warp, i, lane] = acc[i]
                cute.arch.sync_threads()
            if const_expr(SPLIT > 1):
                cute.arch.cluster_wait()
            if warp == 0:
                for i in cutlass.range_constexpr(NA):
                    v = acc[i]
                    for w2 in cutlass.range_constexpr(1, WARPS):
                        v = v + red[w2, i, lane]
                    acc[i] = v
                if const_expr(SPLIT > 1):
                    for i in cutlass.range_constexpr(NA):
                        slot = part.iterator + (rank * NA + i) * 32 + lane
                        st_dsmem_f32(slot, acc[i])
            if const_expr(SPLIT > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            if warp == 0 and rank == 0:
                if const_expr(SPLIT > 1):
                    for i in cutlass.range_constexpr(NA):
                        v = part[0, i, lane]
                        for r2 in cutlass.range_constexpr(1, SPLIT):
                            v = v + part[r2, i, lane]
                        acc[i] = v
                obase = out.iterator.toint()
                for nt in cutlass.range_constexpr(NT):
                    for i in cutlass.range_constexpr(4):
                        v = acc[nt * 4 + i] * gsv
                        if const_expr(RB == 2):
                            u = acc[(NT + nt) * 4 + i] * gsv
                            v = silu_mul(round_bf16(v), round_bf16(u))
                        row = nb * 16 + g + 8 * (i // 2)
                        tok = mt * (8 * NT) + nt * 8 + 2 * t + (i % 2)
                        st_out(
                            obase + (Int64(tok) * stride_o + row) * 2,
                            v,
                            Int32(tok < M),
                        )

    return Nvfp4Gemv


_kernel_cls: Any = None
_compiled: dict[tuple, Any] = {}


def _compile(key: tuple):
    global _kernel_cls
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from quack.compile_utils import make_fake_tensor

    k, nt, cfg, pdl, silu_mul, bf16 = key
    if _kernel_cls is None:
        _kernel_cls = _build_kernel_class()
    sym = cute.sym_int
    x = make_fake_tensor(cutlass.BFloat16, (sym(), k), divisibility=8)
    w = make_fake_tensor(cutlass.Uint8, (sym(),), divisibility=16)
    s = make_fake_tensor(cutlass.Uint8, (sym(),), divisibility=4)
    gs = make_fake_tensor(cutlass.Float32, (1,), divisibility=1)
    out = make_fake_tensor(cutlass.BFloat16, (sym(), sym()), divisibility=1)
    kernel = _kernel_cls(k, nt, cfg.warps, cfg.split, pdl, silu_mul, bf16)
    stream = CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel, x, w, s, gs, out, 1, 128, k, 128, stream, options="--enable-tvm-ffi"
    )


def _get_compiled(n: int, k: int, nt: int, silu_mul: bool, pdl: bool, bf16: bool):
    cfg = _configs(n, k, silu_mul, bf16)[nt]
    key = (k, nt, cfg, pdl, silu_mul, bf16)
    fn = _compiled.get(key)
    if fn is None:
        fn = _compiled[key] = _compile(key)
    return fn


def _launch(x, w, s, gs, n, silu_mul, pdl, bf16) -> torch.Tensor:
    from cuda.bindings.driver import CUstream

    m, k = x.shape
    assert m > 0 and x.stride(1) == 1 and x.dtype == torch.bfloat16
    assert supports(n, k, silu_mul, bf16), (n, k, silu_mul, bf16)
    out = torch.empty(m, n // 2 if silu_mul else n, dtype=x.dtype, device=x.device)
    nts = _configs(n, k, silu_mul, bf16)
    # Beyond the largest tile: re-read the weight per tile of tokens.
    nt = min((nt for nt in nts if 8 * nt >= m), default=max(nts))
    fn = _get_compiled(n, k, nt, silu_mul, pdl, bf16)
    fn(
        x,
        w,
        s,
        gs,
        out,
        m,
        n,
        x.stride(0),
        out.stride(0),
        CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


def nvfp4_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    silu_mul: bool = False,
    pdl: bool = True,
) -> torch.Tensor:
    """Small-M W4A16 GEMM: ``x @ dequant(weight).T * global_scale``.

    Args:
        x: BF16 ``[M, K]`` activations with unit inner stride. Tuned for
            ``M <= max_m(N, K)``; larger M re-reads the weight per token tile.
        weight: uint8 ``[N, K / 2]`` packed E2M1 codes (ModelOpt layout).
        weight_scale: 128x4-swizzled E4M3 block scales (``swizzle_blockscale``).
        global_scale: FP32 scalar tensor (one element).
        silu_mul: Treat the output as ``[gate | up]`` halves and return the
            BF16 ``silu(gate) * up`` (``[M, N / 2]``) of the BF16 gate / up
            outputs (exact SiLU, one BF16 rounding).
        pdl: Launch with programmatic dependent launch: weight loads start
            while the previous kernel drains. Disable after kernels that need
            the whole GPU to make progress (e.g. attention), whose remaining
            CTAs the early, waiting GEMV CTAs would starve.

    Returns:
        The BF16 ``[M, N]`` (``[M, N / 2]`` with ``silu_mul``) output.

    """
    assert weight.shape[1] * 2 == x.shape[1]
    return _launch(
        x,
        weight.view(-1),
        weight_scale.view(torch.uint8).view(-1),
        global_scale.reshape(1),
        weight.shape[0],
        silu_mul,
        pdl,
        bf16=False,
    )


def bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    silu_mul: bool = False,
    pdl: bool = True,
) -> torch.Tensor:
    """Small-M BF16 GEMM ``x @ weight.T`` (FP32 accumulate); same kernel and
    arguments as ``nvfp4_gemv`` with a row-major BF16 ``[N, K]`` weight."""
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert weight.shape[1] == x.shape[1]
    return _launch(
        x,
        weight.view(torch.uint8).view(-1),
        weight.view(torch.uint8).view(-1),
        _one(x.device),
        weight.shape[0],
        silu_mul,
        pdl,
        bf16=True,
    )


_ONES: dict[torch.device, torch.Tensor] = {}


def _one(device: torch.device) -> torch.Tensor:
    one = _ONES.get(device)
    if one is None:
        one = _ONES[device] = torch.ones(1, dtype=torch.float32, device=device)
    return one


def warmup(
    n: int,
    k: int,
    silu_mul: bool = False,
    pdl: bool = True,
    bf16: bool = False,
) -> None:
    """Compile every variant the GEMV may pick for this shape."""
    for nt in _configs(n, k, silu_mul, bf16):
        _get_compiled(n, k, nt, silu_mul, pdl, bf16)


def _dequant_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        w_ptr, s_ptr, out_ptr, row0, K: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr
    ):
        r = row0 + tl.program_id(0) * BR + tl.arange(0, BR)
        kb = tl.program_id(1) * BK
        b = tl.load(
            w_ptr
            + r[:, None].to(tl.int64) * (K // 2)
            + (kb // 2 + tl.arange(0, BK // 2))[None, :]
        )
        # fp4 bits -> fp32 bits of e2m1 * 2^-126 (exact), sign included.
        lo = ((b & 7).to(tl.int32) << 22) | ((b & 8).to(tl.int32) << 28)
        hi = ((b & 0x70).to(tl.int32) << 18) | ((b & 0x80).to(tl.int32) << 24)
        v = tl.reshape(
            tl.interleave(lo, hi).to(tl.float32, bitcast=True), (BR, BK // 16, 16)
        )
        g = kb // 16 + tl.arange(0, BK // 16)
        sidx = ((r[:, None] // 128) * (K // 64) + g[None, :] // 4).to(tl.int64) * 512
        sidx += ((r % 32) * 16 + ((r % 128) // 32) * 4)[:, None] + g[None, :] % 4
        sc = tl.load(s_ptr + sidx).to(tl.float32)
        v = tl.reshape(v * 8.507059173023462e37 * sc[:, :, None], (BR, BK))
        out = (
            out_ptr
            + (r - row0)[:, None].to(tl.int64) * K
            + (kb + tl.arange(0, BK))[None, :]
        )
        tl.store(out, v.to(tl.bfloat16))

    return kernel


_dequant = None


def dequantize_rows(
    weight: torch.Tensor, weight_scale: torch.Tensor, row0: int, rows: int
) -> torch.Tensor:
    """Exact BF16 ``e2m1 * e4m3`` of weight rows ``[row0, row0 + rows)``
    (global scale not applied); ``weight_scale`` is 128x4-swizzled."""
    global _dequant
    if _dequant is None:
        _dequant = _dequant_kernel()
    k = weight.shape[1] * 2
    assert rows % 16 == 0 and k % 256 == 0
    out = torch.empty(rows, k, dtype=torch.bfloat16, device=weight.device)
    _dequant[(rows // 16, k // 256)](
        weight,
        weight_scale.view(torch.float8_e4m3fn).view(-1),
        out,
        row0,
        k,
        16,
        256,
        num_warps=4,
    )
    return out
