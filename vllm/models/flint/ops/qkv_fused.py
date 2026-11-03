# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode qkv for Flint at M <= 8 tokens (CuTe DSL), two kernels:

1. ``fused_add_rms_norm`` (or ``rms_norm`` for the first layer) writing the
   normalized activations ``xn`` and the new residual; it releases its PDL
   dependents immediately so the second kernel's weights stream in meanwhile.
2. the qkv GEMV (same fragment math, K split and summation order as
   ``nvfp4_gemv`` at M <= 8) whose 16-row CTAs form one cluster per head and
   send their BF16 rows to rank 0 through DSMEM, which applies per-head q/k
   RMSNorm, NeoX RoPE (FP32 cos/sin table, FP32 math) and the paged KV-cache
   write like ``qk_norm_rope_cache``.

Numerics are those of ``norm.fused_add_rms_norm`` / ``norm.rms_norm`` ->
GEMV -> ``qk_norm_rope_cache``; the per-head q/k norm sums its squares in a
different order, so q/k can differ from that path by one BF16 ulp.
"""

from functools import cache

import torch

from .nvfp4_gemv import _one

_K = 2048
_HEAD = 128
_CHUNK = 64
_WARPS = 8
MAX_M = 8


@cache
def _kernels():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint32, const_expr
    from cutlass.cutlass_dsl import dsl_user_op

    from .cute_primitives import (
        bf16x2,
        decode,
        fma,
        ld_u8,
        ld_w8,
        ld_w16,
        ld_x16,
        lds_f32,
        mma,
        ptx,
        round_bf16,
        scale_pair,
        smem_addr,
        stg_bf16,
        stg_u32x4,
        sts_f32,
    )

    @dsl_user_op
    def ldg128(addr, *, loc=None, ip=None):
        return ptx(
            "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,l",
            [Int64(addr)],
            (Uint32,) * 4,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ldg128_nc(addr, *, loc=None, ip=None):
        return ptx(
            "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];",
            "=r,=r,=r,=r,l",
            [Int64(addr)],
            (Uint32,) * 4,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def unpack(w, *, loc=None, ip=None):
        return ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.bf16 $0, l; "
            "cvt.f32.bf16 $1, h; }",
            "=f,=f,r",
            [Uint32(w)],
            (Float32, Float32),
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def rsqrt(x, *, loc=None, ip=None):
        return ptx(
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            [Float32(x)],
            Float32,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_bf16(addr, *, loc=None, ip=None):
        return ptx(
            "{ .reg .b16 h; ld.global.nc.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,l",
            [Int64(addr)],
            Float32,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_f32(addr, *, loc=None, ip=None):
        return ptx(
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            [Int64(addr)],
            Float32,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_i64(addr, *, loc=None, ip=None):
        return ptx(
            "ld.global.nc.s64 $0, [$1];",
            "=l,l",
            [Int64(addr)],
            Int64,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def ld_i64_pred(addr, pred, *, loc=None, ip=None):
        return ptx(
            "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b64 $0, -1; "
            "@p ld.global.nc.s64 $0, [$1]; }",
            "=l,l,r",
            [Int64(addr), Int32(pred)],
            Int64,
            effects=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def st_cluster0_f32(addr, v, *, loc=None, ip=None):
        ptx(
            "{ .reg .b32 r; mapa.shared::cluster.u32 r, $0, 0; "
            "st.shared::cluster.f32 [r], $1; }",
            "r,f,~{memory}",
            [Uint32(addr), Float32(v)],
            loc=loc,
            ip=ip,
        )

    class Norm:
        """Grid (M,): one CTA of 256 threads per token row, thread i owns
        elements 8i .. 8i + 7."""

        def __init__(self, has_residual):
            self.has_residual = has_residual

        @cute.jit
        def __call__(self, x, res_in, res_out, xn, norm_w, M: Int32, eps: Float32,
                     stream):  # fmt: skip
            self.kernel(x, res_in, res_out, xn, norm_w, eps).launch(
                grid=[M, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, x, res_in, res_out, xn, norm_w, eps: Float32):
            tidx, _, _ = cute.arch.thread_idx()
            tok, _, _ = cute.arch.block_idx()
            warp = tidx // 32
            lane = tidx % 32
            smem = cutlass.memory.SmemAllocator()
            var_ptr = smem.allocate_array(Float32, 8, byte_alignment=16)
            varb = smem_addr(var_ptr)
            nwv = ldg128_nc(norm_w.iterator.toint() + tidx * 16)
            cute.arch.griddepcontrol_wait()
            # The GEMV only needs our output after its own grid-dependency
            # wait: let it launch and stream its weights now.
            cute.arch.griddepcontrol_launch_dependents()
            off = Int64(tok) * (_K * 2) + tidx * 16
            # s = x (+ residual) in fp32; the variance and the output use the
            # unrounded s, the new residual is bf16(s).
            hv = ldg128(x.iterator.toint() + off)
            sv = [v for i in range(4) for v in unpack(hv[i])]
            if const_expr(self.has_residual):
                rv = ldg128(res_in.iterator.toint() + off)
                rf = [v for i in range(4) for v in unpack(rv[i])]
                sv = [sv[j] + rf[j] for j in range(8)]
                stg_u32x4(
                    res_out.iterator.toint() + off,
                    *[bf16x2(sv[2 * i], sv[2 * i + 1]) for i in range(4)],
                )
            acc_s = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                acc_s = acc_s + sv[2 * i] * sv[2 * i] + sv[2 * i + 1] * sv[2 * i + 1]
            for o in cutlass.range_constexpr(5):
                acc_s = acc_s + cute.arch.shuffle_sync_bfly(acc_s, 1 << o)
            if lane == 0:
                sts_f32(varb + warp * 4, acc_s)
            cute.arch.sync_threads()
            v = Float32(0.0)
            for w2 in cutlass.range_constexpr(_WARPS):
                v = v + lds_f32(varb + w2 * 4)
            rs = rsqrt(v / Float32(_K) + eps)
            outw = []
            for i in cutlass.range_constexpr(4):
                wlo, whi = unpack(nwv[i])
                outw.append(bf16x2(sv[2 * i] * rs * wlo, sv[2 * i + 1] * rs * whi))
            stg_u32x4(xn.iterator.toint() + off, *outw)

    class QkvRope:
        """Grid (N / 16) CTAs in clusters of 8 (one 128-row head each)."""

        def __init__(self, bf16w, rope, num_q_heads, num_kv_heads):
            self.bf16w = bf16w
            self.rope = rope
            self.hq = num_q_heads
            self.hkv = num_kv_heads

        @cute.jit
        def __call__(self, xn, w, s, gs, qn_w, kn_w, cos_sin, positions, slots,
                     q_out, k_cache, v_cache, M: Int32, num_slots: Int32,
                     qk_eps: Float32, stride_block: Int64, stride_slot: Int64,
                     stride_head: Int64, block_size: Int32, stream):  # fmt: skip
            n = (self.hq + 2 * self.hkv) * _HEAD
            self.kernel(
                xn, w, s, gs, qn_w, kn_w, cos_sin, positions, slots, q_out,
                k_cache, v_cache, M, num_slots, qk_eps, stride_block,
                stride_slot, stride_head, block_size,
            ).launch(
                grid=[n // 16, 1, 1],
                block=[32 * _WARPS, 1, 1],
                cluster=[8, 1, 1],
                stream=stream,
                use_pdl=True,
            )  # fmt: skip

        @cute.kernel
        def kernel(self, xn, w, s, gs, qn_w, kn_w, cos_sin, positions, slots,
                   q_out, k_cache, v_cache, M: Int32, num_slots: Int32,
                   qk_eps: Float32, stride_block: Int64, stride_slot: Int64,
                   stride_head: Int64, block_size: Int32):  # fmt: skip
            BF16W: cutlass.Constexpr = self.bf16w
            KC: cutlass.Constexpr = _K // _CHUNK
            CPW: cutlass.Constexpr = KC // _WARPS
            NACC: cutlass.Constexpr = 8
            KB: cutlass.Constexpr = KC * (128 if self.bf16w else 32)
            QW: cutlass.Constexpr = 16 if self.bf16w else 4
            tidx, _, _ = cute.arch.thread_idx()
            nb, _, _ = cute.arch.block_idx()
            head = nb // 8
            rank = nb % 8
            warp = tidx // 32
            lane = tidx % 32
            g = lane // 4
            t = lane % 4
            c_begin = warp * CPW

            smem = cutlass.memory.SmemAllocator()
            red_ptr = smem.allocate_array(Float32, _WARPS * 4 * 32, byte_alignment=16)
            red = smem_addr(red_ptr)
            hb_ptr = smem.allocate_array(Float32, _HEAD * 8, byte_alignment=16)
            hb = smem_addr(hb_ptr)

            # ---- weights and epilogue inputs (before the grid dependency) ----
            qs = cute.make_rmem_tensor((CPW * QW,), Uint32)
            ss = cute.make_rmem_tensor((CPW * 2,), Uint32)
            row_g = nb * 16 + g
            row_g8 = row_g + 8
            if const_expr(BF16W):
                wbase = w.iterator.toint() + Int64(row_g) * KB + c_begin * 128 + t * 32
                for c in cutlass.range_constexpr(CPW):
                    for h in cutlass.range_constexpr(4):
                        off = (h // 2) * 8 * KB + c * 128 + (h % 2) * 16
                        wv = ld_w16(wbase + off)
                        for i in cutlass.range_constexpr(4):
                            qs[c * QW + h * 4 + i] = wv[i]
            else:
                wbase = w.iterator.toint() + Int64(row_g) * KB + c_begin * 32 + t * 8
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
                    qs[c * QW + 0] = qa[0]
                    qs[c * QW + 1] = qa[1]
                    qs[c * QW + 2] = qb[0]
                    qs[c * QW + 3] = qb[1]
                    ss[c * 2] = ld_u8(sbase_g + c * 512)
                    ss[c * 2 + 1] = ld_u8(sbase_g8 + c * 512)
            gsv = Float32(1.0)
            if const_expr(not BF16W):
                gsv = gs[0]
            is_q = head < self.hq
            is_v = head >= self.hq + self.hkv
            wptr = cutlass.select_(is_q, qn_w.iterator.toint(), kn_w.iterator.toint())
            hw = [ld_bf16(Int64(wptr) + (lane + 32 * e) * 2) for e in range(4)]
            etc = cutlass.min(warp, M - 1)
            pos = Int64(0)
            if const_expr(self.rope):
                pos = ld_i64(positions.iterator.toint() + Int64(etc) * 8)
            slot = ld_i64_pred(
                slots.iterator.toint() + Int64(etc) * 8, Int32(etc < num_slots)
            )
            cs_v = [Float32(1.0)] * 2
            sn_v = [Float32(0.0)] * 2
            if const_expr(self.rope):
                # FP32 cos | sin table rows of 128 floats.
                cbase = cos_sin.iterator.toint() + pos * (_HEAD * 4)
                for e in cutlass.range_constexpr(2):
                    cs_v[e] = ld_f32(cbase + (lane + 32 * e) * 4)
                    sn_v[e] = ld_f32(cbase + (64 + lane + 32 * e) * 4)
            cute.arch.cluster_arrive_relaxed()

            cute.arch.griddepcontrol_wait()
            xbase = xn.iterator.toint() + (Int64(c_begin) * _CHUNK + t * 16) * 2
            xall = [
                ld_x16(xbase + Int64(g) * (_K * 2) + c * 128, Int32(g < M))
                for c in range(CPW)
            ]

            # ---- GEMV ----
            acc = cute.make_rmem_tensor((NACC * 4,), Float32)
            acc.fill(0.0)
            for c in cutlass.range_constexpr(CPW):
                xv = xall[c]
                q0 = c * QW
                if const_expr(not BF16W):
                    sg = scale_pair(ss[c * 2])
                    sg8 = scale_pair(ss[c * 2 + 1])
                for j in cutlass.range_constexpr(4):
                    if const_expr(BF16W):
                        a = [
                            qs[q0 + 2 * j],
                            qs[q0 + 8 + 2 * j],
                            qs[q0 + 2 * j + 1],
                            qs[q0 + 8 + 2 * j + 1],
                        ]
                    else:
                        a = decode(qs[q0 + j // 2], qs[q0 + 2 + j // 2], j, sg, sg8)
                    ab = ((c * 4 + j) % NACC) * 4
                    r = mma(
                        [acc[ab], acc[ab + 1], acc[ab + 2], acc[ab + 3]],
                        a,
                        xv[2 * j],
                        xv[2 * j + 1],
                    )
                    for i in cutlass.range_constexpr(4):
                        acc[ab + i] = r[i]
            for h in cutlass.range_constexpr(1, NACC):
                for i in cutlass.range_constexpr(4):
                    acc[i] = acc[i] + acc[h * 4 + i]
            cute.arch.griddepcontrol_launch_dependents()
            for i in cutlass.range_constexpr(4):
                sts_f32(red + ((warp * 4 + i) * 32 + lane) * 4, acc[i])
            cute.arch.sync_threads()
            cute.arch.cluster_wait()
            if warp == 0:
                for i in cutlass.range_constexpr(4):
                    v = acc[i]
                    for w2 in cutlass.range_constexpr(1, _WARPS):
                        v = v + lds_f32(red + ((w2 * 4 + i) * 32 + lane) * 4)
                    v = round_bf16(v * gsv)
                    row = rank * 16 + g + 8 * (i // 2)
                    tok = 2 * t + (i % 2)
                    st_cluster0_f32(hb + (row * 8 + tok) * 4, v)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # ---- per-head epilogue on rank 0: warp = token, lane covers rows
            # lane + 32 * r; RoPE pairs (lane, lane + 64), (lane + 32, lane + 96).
            if (rank == 0) & (warp < M):
                y = [lds_f32(hb + ((lane + 32 * e) * 8 + warp) * 4) for e in range(4)]
                sq = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]
                for o in cutlass.range_constexpr(5):
                    sq = sq + cute.arch.shuffle_sync_bfly(sq, 1 << o)
                rs = rsqrt(sq / Float32(_HEAD) + qk_eps)
                nv = [round_bf16(y[e] * rs * hw[e]) for e in range(4)]
                if const_expr(self.rope):
                    rot = [None] * 4
                    for e in cutlass.range_constexpr(2):
                        rot[e] = fma(nv[e], cs_v[e], -(nv[e + 2] * sn_v[e]))
                        rot[e + 2] = fma(nv[e + 2], cs_v[e], nv[e] * sn_v[e])
                    nv = rot
                out = [cutlass.select_(is_v, y[e], nv[e]) for e in range(4)]
                if is_q:
                    qrow = (
                        q_out.iterator.toint()
                        + (Int64(warp) * (self.hq * _HEAD) + head * _HEAD) * 2
                    )
                    for e in cutlass.range_constexpr(4):
                        stg_bf16(qrow + (lane + 32 * e) * 2, out[e])
                elif slot >= 0:
                    kvh = cutlass.select_(
                        is_v, head - self.hq - self.hkv, head - self.hq
                    )
                    base = cutlass.select_(
                        is_v, v_cache.iterator.toint(), k_cache.iterator.toint()
                    )
                    bs = Int64(block_size)
                    dst = (
                        Int64(base)
                        + (
                            (slot // bs) * stride_block
                            + (slot % bs) * stride_slot
                            + Int64(kvh) * stride_head
                        )
                        * 2
                    )
                    for e in cutlass.range_constexpr(4):
                        stg_bf16(dst + (lane + 32 * e) * 2, out[e])

    return Norm, QkvRope


@cache
def _compile_norm(has_residual):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from quack.compile_utils import make_fake_tensor

    sym = cute.sym_int
    bf16 = cutlass.BFloat16
    args = [make_fake_tensor(bf16, (sym(), _K), divisibility=8) for _ in range(4)]
    args.append(make_fake_tensor(bf16, (_K,), divisibility=8))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _kernels()[0](has_residual), *args, Int32(1), Float32(1e-6), stream,
        options="--enable-tvm-ffi",
    )  # fmt: skip


@cache
def _compile_qkv(bf16w, rope, num_q_heads, num_kv_heads):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64
    from quack.compile_utils import make_fake_tensor

    sym = cute.sym_int
    bf16 = cutlass.BFloat16
    args = [
        make_fake_tensor(bf16, (sym(), _K), divisibility=8),  # xn
        make_fake_tensor(cutlass.Uint8, (sym(),), divisibility=16),  # weight
        make_fake_tensor(cutlass.Uint8, (sym(),), divisibility=4),  # scales
        make_fake_tensor(Float32, (1,), divisibility=1),  # global scale
        make_fake_tensor(bf16, (_HEAD,), divisibility=8),  # q norm w
        make_fake_tensor(bf16, (_HEAD,), divisibility=8),  # k norm w
        make_fake_tensor(Float32, (sym(), sym()), divisibility=4),  # cos | sin
        make_fake_tensor(cutlass.Int64, (sym(),), divisibility=1),  # positions
        make_fake_tensor(cutlass.Int64, (sym(),), divisibility=1),  # slots
        make_fake_tensor(bf16, (sym(), num_q_heads * _HEAD), divisibility=8),
        make_fake_tensor(bf16, (sym(),), divisibility=8),  # k cache (flat)
        make_fake_tensor(bf16, (sym(),), divisibility=8),  # v cache (flat)
    ]
    kernel = _kernels()[1](bf16w, rope, num_q_heads, num_kv_heads)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel, *args, Int32(1), Int32(1), Float32(1e-6), Int64(1), Int64(1),
        Int64(1), Int32(16), stream, options="--enable-tvm-ffi",
    )  # fmt: skip


def flint_qkv_fused(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    qk_eps: float,
    cos_sin_cache: torch.Tensor | None,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual-add RMSNorm + qkv GEMV + q/k norm, RoPE, cache write for
    ``M <= 8`` tokens.

    Args:
        x: [M, 2048] bf16 attention input (contiguous).
        residual: [M, 2048] bf16 residual, or None for the first layer.
        norm_weight: Input RMSNorm weight.
        eps: Input RMSNorm epsilon.
        weight: qkv weight, NVFP4 (uint8 [N, 1024]) or BF16 [N, 2048].
        weight_scale: NVFP4 128x4-swizzled block scales (None for BF16).
        global_scale: NVFP4 fp32 global scale (None for BF16).
        q_norm_weight: Per-head q RMSNorm weight.
        k_norm_weight: Per-head k RMSNorm weight.
        qk_eps: q/k RMSNorm epsilon.
        cos_sin_cache: [max_pos, 128] fp32 (cos | sin), or None (no RoPE).
        positions: [M] int64.
        k_cache: [blocks, block_size, kv_heads, 128] view.
        v_cache: Same shape and strides as ``k_cache``.
        slot_mapping: [<= M] int64; negative slots and tokens beyond it are
            not written.
        num_q_heads: Number of q heads.
        num_kv_heads: Number of KV heads.

    Returns:
        (q [M, num_q_heads * 128] bf16, residual_out [M, 2048] bf16).

    """
    m = x.shape[0]
    assert 0 < m <= MAX_M and x.shape[1] == _K and x.is_contiguous()
    bf16w = weight.dtype == torch.bfloat16
    n = (num_q_heads + 2 * num_kv_heads) * _HEAD
    assert weight.shape[0] == n and weight.is_contiguous()
    assert k_cache.stride() == v_cache.stride() and k_cache.stride(-1) == 1
    has_residual = residual is not None
    if residual is not None:
        assert residual.is_contiguous()
        res_out = torch.empty_like(x)
    else:
        residual = res_out = x
    xn = torch.empty_like(x)
    _compile_norm(has_residual)(x, residual, res_out, xn, norm_weight, m, eps)
    q = torch.empty(m, num_q_heads * _HEAD, dtype=x.dtype, device=x.device)
    w = weight.view(torch.uint8).view(-1)
    s = w if weight_scale is None else weight_scale.view(torch.uint8).view(-1)
    gs = global_scale.reshape(1) if global_scale is not None else _one(x.device)
    rope = cos_sin_cache is not None
    if cos_sin_cache is not None:
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[1] == _HEAD
        cs = cos_sin_cache
    else:
        cs = no_rope_table(x.device)
    _compile_qkv(bf16w, rope, num_q_heads, num_kv_heads)(
        xn, w, s, gs, q_norm_weight, k_norm_weight, cs, positions, slot_mapping,
        q, _base(k_cache), _base(v_cache), m, slot_mapping.shape[0], qk_eps,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.shape[1],
    )  # fmt: skip
    return q, res_out


def _base(t: torch.Tensor) -> torch.Tensor:
    return torch.as_strided(t, (_HEAD,), (1,))


_NO_ROPE: dict[torch.device, torch.Tensor] = {}


def no_rope_table(device: torch.device) -> torch.Tensor:
    """Placeholder fp32 cos|sin argument for kernels compiled without RoPE."""
    t = _NO_ROPE.get(device)
    if t is None:
        t = _NO_ROPE[device] = torch.zeros(1, _HEAD, device=device)
    return t
