# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An MFMA co-runner for gfx950 kernel tests: on MI355X, v_pk_mul_f32 can
return wrong products while another kernel's MFMA waves share the CU."""

import torch


def mfma_spinner(n_wg):
    """``n_wg`` workgroups looping v_mfma_f32_16x16x32_bf16 for ``ticks``
    (s_memrealtime, 100 per us).

    Launch it on a high-priority stream: on a normal one it can share a hardware
    queue with the kernel under test and never overlap it.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    @fx.struct
    class Shared:
        a: fx.Array[fx.Int32, 4096, 16]

    def now():
        return fx.Int64(
            llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
        )

    @flyc.kernel(name="test_mfma_spinner", known_block_size=[256, 1, 1])
    def spinner(arg_out: fx.Int64, i64_ticks: fx.Int64):
        sm = fx.SharedAllocator().allocate(Shared).peek()
        tx = fx.thread_idx.x
        sm.a.ptr[tx] = tx
        a4 = Vec.from_elements([tx, tx + 1, tx + 2, tx + 3], fx.Int32)
        acc = Vec.filled(4, 0.0, fx.Float32)
        t0 = now()
        t = now()
        while t - t0 < i64_ticks:
            for _ in range_constexpr(16):
                ab = Vec(a4).bitcast(fx.BFloat16)
                acc = Vec(
                    fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [ab, ab, acc, 0, 0, 0]
                    )
                )
            t = now()
        f32p = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )
        out = fx.inttoptr(f32p, arg_out)
        out[fx.block_idx.x * 256 + tx] = acc[0] + fx.Float32(sm.a.ptr[255 - tx])

    @flyc.jit
    def launch(arg_out: fx.Int64, i64_ticks: fx.Int64, stream: fx.Stream):
        spinner(arg_out, i64_ticks).launch(
            grid=(n_wg, 1, 1), block=(256, 1, 1), stream=stream
        )

    buf = torch.zeros(n_wg * 256, device="cuda")
    compiled = flyc.compile(launch, buf.data_ptr(), 1, torch.cuda.current_stream())
    return lambda ticks, stream: compiled(buf.data_ptr(), ticks, stream)
