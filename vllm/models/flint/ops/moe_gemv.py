# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-indexed small-M NVFP4 GEMV kernels (CuTe DSL) for the low-latency
decode MoE block, built on the dense GEMV design of ``nvfp4_gemv``: each CTA
owns 16 output rows (the m16n8k16 M dimension, swap-AB), its warps split K and
reduce through shared memory, weights are dequantized exactly to BF16
(``cvt.rn.bf16x2.e2m1x2`` + ``mul.rn.bf16x2``) and the fp32 global scale is
applied to the fp32 accumulator.

``gate_up``: one (token, expert) pair per CTA column of the grid; writes
``bf16(silu(g) * u)`` from fp32 gate/up with FlashInfer's approximate sigmoid
(the shared expert rounds its gate/up to BF16 first and uses the exact SiLU,
as its dense GEMM does). Expert weights use FlashInfer's W4A16
layout ([up; gate] in 64-row groups, 6D MMA scales).

``down_combine``: per token and 16 output rows, the down GEMVs of the token's
top-k experts and of the shared expert, all weights prefetched before
``griddepcontrol.wait``; ``out = bf16(bf16(sum_j tw_j * y_j) + y_shared)``
with ``y = bf16(acc * global_scale)``.
"""

from typing import Any

import torch

_CHUNK = 64
_WARPS = 4
_SPLIT = 2

_classes: Any = None
_compiled: dict[tuple, Any] = {}


def _build():
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass import Float32, Int32, Int64, Uint32, const_expr
    from cutlass.cutlass_dsl import dsl_user_op

    from .cute_primitives import (
        decode,
        ld_u8,
        ld_x16,
        mma,
        ptx,
        round_bf16,
        scale_pair,
        silu_mul,
        st_dsmem_f32,
        st_out,
    )

    @dsl_user_op
    def silu_mul_approx(g: Float32, u: Float32, *, loc=None, ip=None):
        # g * rcp.approx(1 + ex2.approx(-g * log2(e))) * u, the instructions of
        # FlashInfer's W4A16 MoE epilogue (routed experts).
        return ptx(
            "{\n.reg .f32 e, d;\n"
            "mul.f32 e, $1, 0fBFB8AA3B;\nex2.approx.ftz.f32 e, e;\n"
            "add.f32 d, e, 0f3F800000;\nrcp.approx.ftz.f32 d, d;\n"
            "mul.f32 d, $1, d;\nmul.f32 $0, d, $2;\n}",
            "=f,f,f",
            [g, u],
            Float32,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_w8p(addr: Int64, *, loc=None, ip=None):
        # Weight words with a 128-byte L2 prefetch: a lane's later chunks of
        # the same row follow in 32-byte steps.
        return list(
            ptx(
                "ld.global.nc.L1::no_allocate.L2::128B.v2.u32 {$0, $1}, [$2];",
                "=r,=r,l",
                [addr],
                (Uint32,) * 2,
                loc=loc,
                ip=ip,
            )
        )

    @cute.jit
    def load_rows(
        w_base,
        s_base,
        row,
        KC: cutlass.Constexpr,
        CPW: cutlass.Constexpr,
        c_begin,
        t,
        qs,
        ss,
        q_off: cutlass.Constexpr,
        s_off: cutlass.Constexpr,
    ):
        """Weight words / scale bytes of rows (row, row + 8) for CPW chunks."""
        kb = KC * 32
        wbase = w_base + Int64(row) * kb + c_begin * 32 + t * 8
        row8 = row + 8
        sb_g = (
            s_base
            + Int64(row // 128) * (KC * 512)
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + t
            + c_begin * 512
        )
        sb_g8 = (
            s_base
            + Int64(row8 // 128) * (KC * 512)
            + (row8 % 32) * 16
            + ((row8 % 128) // 32) * 4
            + t
            + c_begin * 512
        )
        for c in cutlass.range_constexpr(CPW):
            qa = ld_w8p(wbase + c * 32)
            qb = ld_w8p(wbase + 8 * kb + c * 32)
            q0 = q_off + c * 4
            qs[q0 + 0] = qa[0]
            qs[q0 + 1] = qa[1]
            qs[q0 + 2] = qb[0]
            qs[q0 + 3] = qb[1]
            ss[s_off + c * 2] = ld_u8(sb_g + c * 512)
            ss[s_off + c * 2 + 1] = ld_u8(sb_g8 + c * 512)

    @cute.jit
    def gemv_pass(
        acc,
        a_off: cutlass.Constexpr,
        qs,
        ss,
        q_off: cutlass.Constexpr,
        s_off: cutlass.Constexpr,
        x_base,
        CPW: cutlass.Constexpr,
        valid,
    ):
        """acc[a_off:a_off+4] += W_rows (16 x 64*CPW) @ x (token column 0)."""
        for c in cutlass.range_constexpr(CPW):
            xv = ld_x16(x_base + c * 128, valid)
            sg = scale_pair(ss[s_off + c * 2])
            sg8 = scale_pair(ss[s_off + c * 2 + 1])
            q0 = q_off + c * 4
            for j in cutlass.range_constexpr(4):
                a = decode(qs[q0 + j // 2], qs[q0 + 2 + j // 2], j, sg, sg8)
                r = mma(
                    [acc[a_off], acc[a_off + 1], acc[a_off + 2], acc[a_off + 3]],
                    a,
                    xv[2 * j],
                    xv[2 * j + 1],
                )
                for i in cutlass.range_constexpr(4):
                    acc[a_off + i] = r[i]

    def alloc_reduce(smem, NA, WARPS, SPLIT):
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
        return red, part

    @cute.jit
    def cta_reduce(
        acc,
        NA: cutlass.Constexpr,
        WARPS: cutlass.Constexpr,
        SPLIT: cutlass.Constexpr,
        warp,
        lane,
        rank,
        red,
        part,
    ):
        """Sum each lane's NA accumulators over the CTA's warps and the
        cluster's CTAs (fixed order) into warp 0 of cluster rank 0."""
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
                    st_dsmem_f32(part.iterator + (rank * NA + i) * 32 + lane, acc[i])
        if const_expr(SPLIT > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if warp == 0 and rank == 0:
                for i in cutlass.range_constexpr(NA):
                    v = part[0, i, lane]
                    for r2 in cutlass.range_constexpr(1, SPLIT):
                        v = v + part[r2, i, lane]
                    acc[i] = v

    class GateUp:
        def __init__(self, h, inter, si, topk):
            self.h, self.inter, self.si, self.topk = h, inter, si, topk
            assert si % inter == 0
            self.kc = h // _CHUNK
            assert self.kc % (_WARPS * _SPLIT) == 0
            self.cpw = self.kc // (_WARPS * _SPLIT)

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            ids: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            act: cute.Tensor,
            wsh: cute.Tensor,
            ssh: cute.Tensor,
            gssh: cute.Tensor,
            sact: cute.Tensor,
            P: Int32,
            NSH: Int32,
            stream: CUstream,
        ):
            self.kernel(x, ids, w, s, gs, act, wsh, ssh, gssh, sact, P).launch(
                grid=[self.inter // 16 * _SPLIT, P + NSH, 1],
                block=[32 * _WARPS, 1, 1],
                cluster=[_SPLIT, 1, 1],
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            x: cute.Tensor,
            ids: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            act: cute.Tensor,
            wsh: cute.Tensor,
            ssh: cute.Tensor,
            gssh: cute.Tensor,
            sact: cute.Tensor,
            P: Int32,
        ):
            H: cutlass.Constexpr = self.h
            INTER: cutlass.Constexpr = self.inter
            KC: cutlass.Constexpr = self.kc
            CPW: cutlass.Constexpr = self.cpw
            WARPS: cutlass.Constexpr = _WARPS
            tidx, _, _ = cute.arch.thread_idx()
            SPLIT: cutlass.Constexpr = _SPLIT
            bx, p, _ = cute.arch.block_idx()
            nb = bx // SPLIT
            rank = bx % SPLIT
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4
            c_begin = (rank * WARPS + warp) * CPW
            SI: cutlass.Constexpr = self.si
            # Programs p >= P are the shared expert: token (p - P) // halves,
            # columns ((p - P) % halves) * INTER + 16 * nb.
            HALVES: cutlass.Constexpr = self.si // self.inter
            row_g = Int32(0)
            row_u = Int32(0)
            w_base = Int64(0)
            s_base = Int64(0)
            gsv = Float32(0.0)
            tok = Int32(0)
            o_off = Int64(0)
            qs = cute.make_rmem_tensor((2 * CPW * 4,), Uint32)
            ss = cute.make_rmem_tensor((2 * CPW * 2,), Uint32)
            if p < P:
                cute.arch.griddepcontrol_wait()
                e = Int64(ids[p])
                # FlashInfer GEMM1 rows: [up 64; gate 64] per 128-row group.
                row_u = (nb * 16) // 64 * 128 + (nb * 16) % 64 + g
                row_g = row_u + 64
                w_base = w.iterator.toint() + e * (2 * INTER * KC * 32)
                s_base = s.iterator.toint() + e * ((2 * INTER // 128) * KC * 512)
                gsv = gs[e]
                tok = p // self.topk
                o_off = Int64(p) * INTER + nb * 16
                load_rows(w_base, s_base, row_g, KC, CPW, c_begin, t, qs, ss, 0, 0)
                load_rows(
                    w_base, s_base, row_u, KC, CPW, c_begin, t, qs, ss, CPW * 4, CPW * 2
                )
            else:
                # Shared expert: static weights, loaded before the wait.
                v = p - P
                tok = v // HALVES
                c0 = (v % HALVES) * INTER + nb * 16
                row_g = c0 + g
                row_u = SI + c0 + g
                w_base = wsh.iterator.toint()
                s_base = ssh.iterator.toint()
                gsv = gssh[0]
                o_off = Int64(tok) * SI + c0
                load_rows(w_base, s_base, row_g, KC, CPW, c_begin, t, qs, ss, 0, 0)
                load_rows(
                    w_base, s_base, row_u, KC, CPW, c_begin, t, qs, ss, CPW * 4, CPW * 2
                )
                cute.arch.griddepcontrol_wait()
            if const_expr(SPLIT > 1):
                cute.arch.cluster_arrive_relaxed()
            x_base = x.iterator.toint() + (Int64(tok) * H + c_begin * 64 + t * 16) * 2
            valid = Int32(g == 0)
            acc = cute.make_rmem_tensor((8,), Float32)
            acc.fill(0.0)
            gemv_pass(acc, 0, qs, ss, 0, 0, x_base, CPW, valid)
            gemv_pass(acc, 4, qs, ss, CPW * 4, CPW * 2, x_base, CPW, valid)
            cute.arch.griddepcontrol_launch_dependents()
            smem = cutlass.memory.SmemAllocator()
            red, part = alloc_reduce(smem, 8, WARPS, SPLIT)
            cta_reduce(acc, 8, WARPS, SPLIT, warp, lane, rank, red, part)
            if warp == 0 and rank == 0 and t == 0:
                a_base = act.iterator.toint()
                if p >= P:
                    a_base = sact.iterator.toint()
                # Routed experts keep gate/up in fp32 (TRTLLM MoE rule); the
                # shared expert is a dense MLP whose gate/up GEMM output is BF16.
                is_shared = p >= P
                for i in cutlass.range_constexpr(0, 4, 2):
                    gv = acc[i] * gsv
                    uv = acc[4 + i] * gsv
                    gv = cutlass.select_(is_shared, round_bf16(gv), gv)
                    uv = cutlass.select_(is_shared, round_bf16(uv), uv)
                    v = cutlass.select_(
                        is_shared, silu_mul(gv, uv), silu_mul_approx(gv, uv)
                    )
                    off = o_off + g + 8 * (i // 2)
                    st_out(a_base + off * 2, v, Int32(1))

    class DownCombine:
        def __init__(self, h, inter, si, topk):
            self.h, self.inter, self.si, self.topk = h, inter, si, topk
            self.kc = inter // _CHUNK
            self.kcs = si // _CHUNK
            parts = _WARPS * _SPLIT
            assert self.kc % parts == 0 and self.kcs % parts == 0
            self.cpw = self.kc // parts
            self.cpws = self.kcs // parts

        @cute.jit
        def __call__(
            self,
            act: cute.Tensor,
            ids: cute.Tensor,
            tw: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            sact: cute.Tensor,
            wsd: cute.Tensor,
            ssd: cute.Tensor,
            gssd: cute.Tensor,
            out: cute.Tensor,
            M: Int32,
            stream: CUstream,
        ):
            self.kernel(act, ids, tw, w, s, gs, sact, wsd, ssd, gssd, out).launch(
                grid=[self.h // 16 * _SPLIT, M, 1],
                block=[32 * _WARPS, 1, 1],
                cluster=[_SPLIT, 1, 1],
                stream=stream,
                use_pdl=True,
            )

        @cute.kernel
        def kernel(
            self,
            act: cute.Tensor,
            ids: cute.Tensor,
            tw: cute.Tensor,
            w: cute.Tensor,
            s: cute.Tensor,
            gs: cute.Tensor,
            sact: cute.Tensor,
            wsd: cute.Tensor,
            ssd: cute.Tensor,
            gssd: cute.Tensor,
            out: cute.Tensor,
        ):
            H: cutlass.Constexpr = self.h
            INTER: cutlass.Constexpr = self.inter
            SI: cutlass.Constexpr = self.si
            TOPK: cutlass.Constexpr = self.topk
            KC: cutlass.Constexpr = self.kc
            KCS: cutlass.Constexpr = self.kcs
            CPW: cutlass.Constexpr = self.cpw
            CPWS: cutlass.Constexpr = self.cpws
            WARPS: cutlass.Constexpr = _WARPS
            tidx, _, _ = cute.arch.thread_idx()
            SPLIT: cutlass.Constexpr = _SPLIT
            bx, tok, _ = cute.arch.block_idx()
            nb = bx // SPLIT
            rank = bx % SPLIT
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4
            part_id = rank * WARPS + warp
            row = nb * 16 + g
            qs = cute.make_rmem_tensor(((TOPK * CPW + CPWS) * 4,), Uint32)
            ss = cute.make_rmem_tensor(((TOPK * CPW + CPWS) * 2,), Uint32)
            # Expert ids / weights come from the router stage, which completed
            # before the previous kernel passed its dependency wait.
            gsv = cute.make_rmem_tensor((TOPK,), Float32)
            twv = cute.make_rmem_tensor((TOPK,), Float32)
            for j in cutlass.range_constexpr(TOPK):
                e = Int64(ids[tok * TOPK + j])
                load_rows(
                    w.iterator.toint() + e * (H * KC * 32),
                    s.iterator.toint() + e * ((H // 128) * KC * 512),
                    row,
                    KC,
                    CPW,
                    part_id * CPW,
                    t,
                    qs,
                    ss,
                    j * CPW * 4,
                    j * CPW * 2,
                )
                gsv[j] = gs[e]
                twv[j] = tw[tok * TOPK + j]
            load_rows(
                wsd.iterator.toint(),
                ssd.iterator.toint(),
                row,
                KCS,
                CPWS,
                part_id * CPWS,
                t,
                qs,
                ss,
                TOPK * CPW * 4,
                TOPK * CPW * 2,
            )
            gss = gssd[0]
            if const_expr(SPLIT > 1):
                cute.arch.cluster_arrive_relaxed()
            cute.arch.griddepcontrol_wait()
            valid = Int32(g == 0)
            NA: cutlass.Constexpr = (TOPK + 1) * 4
            acc = cute.make_rmem_tensor((NA,), Float32)
            acc.fill(0.0)
            for j in cutlass.range_constexpr(TOPK):
                x_base = (
                    act.iterator.toint()
                    + (Int64(tok * TOPK + j) * INTER + part_id * CPW * 64 + t * 16) * 2
                )
                gemv_pass(
                    acc, j * 4, qs, ss, j * CPW * 4, j * CPW * 2, x_base, CPW, valid
                )
            x_base = (
                sact.iterator.toint()
                + (Int64(tok) * SI + part_id * CPWS * 64 + t * 16) * 2
            )
            gemv_pass(
                acc,
                TOPK * 4,
                qs,
                ss,
                TOPK * CPW * 4,
                TOPK * CPW * 2,
                x_base,
                CPWS,
                valid,
            )
            cute.arch.griddepcontrol_launch_dependents()
            smem = cutlass.memory.SmemAllocator()
            red, part = alloc_reduce(smem, NA, WARPS, SPLIT)
            cta_reduce(acc, NA, WARPS, SPLIT, warp, lane, rank, red, part)
            if warp == 0 and rank == 0 and t == 0:
                o_base = out.iterator.toint()
                for i in cutlass.range_constexpr(0, 4, 2):
                    routed = Float32(0.0)
                    for j in cutlass.range_constexpr(TOPK):
                        routed = routed + twv[j] * round_bf16(acc[j * 4 + i] * gsv[j])
                    shared = round_bf16(acc[TOPK * 4 + i] * gss)
                    v = round_bf16(routed) + shared
                    r = nb * 16 + g + 8 * (i // 2)
                    st_out(o_base + (Int64(tok) * H + r) * 2, v, Int32(1))

    return GateUp, DownCombine


def _fake(dtype, shape, div):
    from quack.compile_utils import make_fake_tensor

    return make_fake_tensor(dtype, shape, divisibility=div)


def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
    """The contiguous storage behind a (possibly permuted) scale view."""
    t = t.view(torch.uint8)
    return torch.as_strided(t, (t.numel(),), (1,), t.storage_offset())


def _stream():
    from cuda.bindings.driver import CUstream

    return CUstream(torch.cuda.current_stream().cuda_stream)


def gate_up(
    x: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    alpha: torch.Tensor,
    act: torch.Tensor,
    wsh: torch.Tensor,
    ssh: torch.Tensor,
    gssh: torch.Tensor,
    sact: torch.Tensor,
) -> None:
    """act[p] = silu_and_mul((x[p // topk] @ W1[ids[p]]^T) * alpha) on fp32
    gate/up for the FlashInfer-layout NVFP4 experts (w1 uint8 [E, 2I, H/2]),
    and in the same launch sact[t] = silu_and_mul(bf16((x[t] @ Wsh^T) * gssh))
    for the shared expert (ModelOpt [gate; up] uint8 [2 SI, H/2], flat
    128x4-swizzled scales)."""
    global _classes
    import cutlass
    import cutlass.cute as cute

    if _classes is None:
        _classes = _build()
    m, h = x.shape
    p, topk = ids.numel(), ids.shape[1]
    inter = w1.shape[1] // 2
    si = wsh.shape[0] // 2
    key = ("gu", h, inter, si, topk)
    fn = _compiled.get(key)
    if fn is None:
        sym = cute.sym_int
        kern = _classes[0](h, inter, si, topk)
        fn = _compiled[key] = cute.compile(
            kern,
            _fake(cutlass.BFloat16, (sym(),), 8),
            _fake(cutlass.Int32, (sym(),), 1),
            _fake(cutlass.Uint8, (sym(),), 16),
            _fake(cutlass.Uint8, (sym(),), 4),
            _fake(cutlass.Float32, (sym(),), 1),
            _fake(cutlass.BFloat16, (sym(),), 1),
            _fake(cutlass.Uint8, (sym(),), 16),
            _fake(cutlass.Uint8, (sym(),), 4),
            _fake(cutlass.Float32, (1,), 1),
            _fake(cutlass.BFloat16, (sym(),), 1),
            1,
            1,
            _stream(),
            options="--enable-tvm-ffi",
        )
    fn(
        x.view(-1),
        ids.view(-1),
        w1.view(-1),
        _flat_bytes(w1_sf),
        alpha,
        act.view(-1),
        wsh.view(torch.uint8).view(-1),
        ssh.view(-1),
        gssh.reshape(1),
        sact.view(-1),
        p,
        m * (si // inter),
        _stream(),
    )


def down_combine(
    act: torch.Tensor,
    ids: torch.Tensor,
    tw: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    alpha2: torch.Tensor,
    sact: torch.Tensor,
    wsd: torch.Tensor,
    ssd: torch.Tensor,
    gssd: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """out[t] = bf16(bf16(sum_j tw * bf16(act_j @ W2[e_j]^T * alpha2)) +
    bf16(sact[t] @ Wsd^T * gssd))."""
    global _classes
    import cutlass
    import cutlass.cute as cute

    if _classes is None:
        _classes = _build()
    m, topk = ids.shape
    h = w2.shape[1]
    inter = w2.shape[2] * 2
    si = wsd.shape[1] * 2
    key = ("down", h, inter, si, topk)
    fn = _compiled.get(key)
    if fn is None:
        sym = cute.sym_int
        kern = _classes[1](h, inter, si, topk)
        fn = _compiled[key] = cute.compile(
            kern,
            _fake(cutlass.BFloat16, (sym(),), 8),
            _fake(cutlass.Int32, (sym(),), 1),
            _fake(cutlass.Float32, (sym(),), 1),
            _fake(cutlass.Uint8, (sym(),), 16),
            _fake(cutlass.Uint8, (sym(),), 4),
            _fake(cutlass.Float32, (sym(),), 1),
            _fake(cutlass.BFloat16, (sym(),), 8),
            _fake(cutlass.Uint8, (sym(),), 16),
            _fake(cutlass.Uint8, (sym(),), 4),
            _fake(cutlass.Float32, (1,), 1),
            _fake(cutlass.BFloat16, (sym(),), 1),
            1,
            _stream(),
            options="--enable-tvm-ffi",
        )
    fn(
        act.view(-1),
        ids.view(-1),
        tw.view(-1),
        w2.view(-1),
        _flat_bytes(w2_sf),
        alpha2,
        sact.view(-1),
        wsd.view(-1),
        ssd.view(-1),
        gssd.reshape(1),
        out.view(-1),
        m,
        _stream(),
    )
