# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 small-batch MXFP8 WO-A with inverse RoPE and quantization."""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op

from vllm.cute_utils import _tcgen05, mbarrier, recast_val, simple_tma_copy
from vllm.cute_utils.cvt import bf16x2_to_fp32x2, fp32x4_to_fp8x4
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)


@dsl_user_op
def _ue8m0_rp(value, *, loc=None, ip=None):
    result = llvm.inline_asm(
        Uint32.mlir_type,
        [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 sf; cvt.rp.satfinite.ue8m0x2.f32 sf, $1, $1; "
        "cvt.u32.u16 $0, sf; and.b32 $0, $0, 255; }",
        "=r,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _scale(amax: Float32):
    exponent = _ue8m0_rp(amax * Float32(1.0 / 448.0))
    inv = recast_val((Uint32(254) - exponent) << 23, Float32)
    if exponent == 0:
        inv = Float32(0)
    return exponent, inv


class FusedWoAKernel(VllmCuTeDSLJitKernel["FusedWoAKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        tokens: int
        x_stride: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        tokens = compile_key.tokens
        acc_cols = 1 << (tokens - 1).bit_length()
        tile_m = max(8, acc_cols)
        sf_col = max(16, tile_m)
        tmem_cols = sf_col * 2
        threads = min(acc_cols * 128, 512)

        @cute.kernel
        def device_kernel(
            x: cute.Tensor,
            positions: cute.Tensor,
            rope: cute.Tensor,
            w: cpasync.TmaInfo,
            ws: cute.Tensor,
            q: cute.Tensor,
            qs: cute.Tensor,
        ):
            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(tid // 32)
            lane = tid % 32
            split = bid % 8
            tile = bid // 8
            group = tile // 8
            smem = utils.SmemAllocator()
            sw = smem.allocate_tensor(
                Float8E4M3FN,
                w.smem_layout.outer,
                byte_alignment=128,
                swizzle=w.smem_layout.inner,
            )
            sx = smem.allocate_tensor(
                Float8E4M3FN,
                cute.make_layout((tile_m, 128, 4), stride=(128, 1, tile_m * 128)),
                byte_alignment=1024,
            )
            ssw = smem.allocate_tensor(Int32, cute.make_layout((128, 4)), 128)
            ssx = smem.allocate_tensor(Int32, cute.make_layout((128, 4)), 128)
            partial = smem.allocate_tensor(
                Float32, cute.make_layout((128, (tokens + 7) // 8, 8)), 128
            )
            loaded = smem.allocate_array(Int64, 4)
            done = smem.allocate_array(Int64, 1)
            reduced = smem.allocate_array(Int64, 1)
            taddr = smem.allocate(Int32, 4)
            if tid == 0:
                for stage in cutlass.range_constexpr(4):
                    cute.arch.mbarrier_init(loaded + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(reduced, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
            if warp == 0:
                tiles = cute.zipped_divide(w.tma_tensor, (128, 128))
                for stage in cutlass.range_constexpr(4):
                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(loaded + stage, 16384)
                    simple_tma_copy(
                        w.atom,
                        tiles[None, (tile, split * 4 + stage)],
                        sw[None, None, stage],
                        loaded + stage,
                    )
            elif warp == 1:
                cute.arch.alloc_tmem(tmem_cols, taddr)
                cute.arch.relinquish_tmem_alloc_permit()
            if tid < 128:
                for stage in cutlass.range_constexpr(4):
                    ssw[(tid % 32) * 4 + tid // 32, stage] = ws[
                        group, (tile % 8) * 128 + tid, split * 4 + stage
                    ]
                    ssx[tid, stage] = Int32(0)
                    sx32 = cute.recast_tensor(sx, Int32)
                    for row in cutlass.range_constexpr(tile_m // 4):
                        sx32[tid // 32 + row * 4, tid % 32, stage] = Int32(0)
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_wait()
            x64 = cute.recast_tensor(x, Uint64)
            sx32 = cute.recast_tensor(sx, Uint32)
            sfbytes = cute.recast_tensor(ssx, Uint8)
            for iteration in cutlass.range_constexpr(
                (tokens * 128 + threads - 1) // threads
            ):
                token = iteration * (threads // 128) + tid // 128
                qid = tid % 128
                k = qid * 4
                if cutlass.const_expr(tokens % (threads // 128) == 0) or token < tokens:
                    packed = x64[token, group * 8 + split, qid]
                    v0, v1 = bf16x2_to_fp32x2(Uint32(packed))
                    v2, v3 = bf16x2_to_fp32x2(Uint32(packed >> 32))
                    if k >= 448:
                        pos = positions[token]
                        c = Float32(rope[pos, (k - 448) // 2])
                        s = Float32(rope[pos, (k - 448) // 2 + 32])
                        v0, v1 = v0 * c + v1 * s, v1 * c - v0 * s
                        c = Float32(rope[pos, (k - 448) // 2 + 1])
                        s = Float32(rope[pos, (k - 448) // 2 + 33])
                        v2, v3 = v2 * c + v3 * s, v3 * c - v2 * s
                    amax = cute.arch.fmax(
                        cute.arch.fmax(cute.abs(v0), cute.abs(v1)),
                        cute.arch.fmax(cute.abs(v2), cute.abs(v3)),
                    )
                    amax = cute.arch.warp_reduction_max(amax, threads_in_group=8)
                    exponent, inv = _scale(cute.arch.fmax(amax, Float32(1e-10)))
                    # Recasting drops the pointer swizzle; address it explicitly.
                    sx32[token, (qid % 32) ^ ((token % 8) * 4), qid // 32] = (
                        fp32x4_to_fp8x4(v0 * inv, v1 * inv, v2 * inv, v3 * inv)
                    )
                    if lane % 8 == 0:
                        sfbytes[token * 16 + k % 128 // 32 + k // 128 * 512] = Uint8(
                            exponent
                        )
            cute.arch.sync_threads()
            cute.arch.fence_proxy("async.shared", space="cta")
            _tcgen05.fence_after_thread_sync()
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()
            base = cute.make_tensor(taddr, cute.make_layout(1))[0]
            if warp == 0:
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                sfdesc = Uint64((8 << 32) | (1 << 46))
                idesc = _tcgen05.make_mxfp8_idesc(128, tile_m)
                for stage in cutlass.range_constexpr(4):
                    cute.arch.mbarrier_wait(loaded + stage, 0)
                    adesc = sdesc | (sw[None, None, stage].iterator.toint() >> 4)
                    bdesc = sdesc | (sx[None, None, stage].iterator.toint() >> 4)
                    _tcgen05.cp(
                        base + sf_col,
                        sfdesc | (ssw[None, stage].iterator.toint() >> 4),
                        "32x128b",
                        "warpx4",
                    )
                    _tcgen05.cp(
                        base + sf_col + 4,
                        sfdesc | (ssx[None, stage].iterator.toint() >> 4),
                        "32x128b",
                        "warpx4",
                    )
                    for kk in cutlass.range_constexpr(4):
                        _tcgen05.mma_mxfp8(
                            base,
                            adesc + kk * 2,
                            bdesc + kk * 2,
                            idesc + ((kk << 4) | (kk << 29)),
                            base + sf_col,
                            base + sf_col + 4,
                            stage > 0 or kk > 0,
                        )
                _tcgen05.commit(done)
            cute.arch.mbarrier_wait(done, 0)
            _tcgen05.fence_after_thread_sync()
            if tid == 0:
                owned = (tokens + 7 - split) // 8
                mbarrier.arrive_expect_tx(reduced, owned * 128 * 8 * 4)
            cute.arch.cluster_wait()
            if tid < 128:
                acc = cute.make_rmem_tensor(acc_cols, Float32)
                if cutlass.const_expr(tokens == 1):
                    acc[0] = _tcgen05.ld(warp * 32, base, "32x32b", 1)
                else:
                    acc.store(_tcgen05.ld(warp * 32, base, "32x32b", acc_cols))
                _tcgen05.wait_ld()
                for token in cutlass.range_constexpr(tokens):
                    ptr = cute.domain_offset((tid, token // 8, split), partial).iterator
                    cute.arch.store_async_dsmem(
                        ptr, recast_val(acc[token], Int32), reduced, token % 8
                    )
                if split < tokens:
                    cute.arch.mbarrier_wait(reduced, 0)
                    for i in cutlass.range_constexpr((tokens + 7) // 8):
                        token = split + i * 8
                        if (
                            cutlass.const_expr(tokens <= 8 or tokens % 8 == 0)
                            or token < tokens
                        ):
                            value = Float32(0)
                            for peer in cutlass.range_constexpr(8):
                                value += partial[tid, i, peer]
                            value = Float32(BFloat16(value))
                            amax = cute.arch.warp_reduction_max(cute.abs(value))
                            exponent, inv = _scale(amax)
                            q[token, tile * 128 + tid] = Float8E4M3FN(value * inv)
                            if lane == 0:
                                qs[tile * 512 + token * 16 + warp] = Uint8(exponent)
                # Every scale tile owns its padding, with no second memset kernel.
                for i in cutlass.range_constexpr(4):
                    offset = i * 128 + tid
                    row = offset // 16 + (offset % 16 // 4) * 32
                    if split == 0 and row >= tokens:
                        qs[tile * 512 + offset] = Uint8(0)
            # Each receiver drained all writes into its own inbox. Only local
            # TMEM readers must finish before deallocation; no peer reads SMEM.
            cute.arch.sync_threads()
            if warp == 0:
                cute.arch.dealloc_tmem(
                    cute.make_ptr(Float32, base, cute.AddressSpace.tmem), tmem_cols
                )

        @cute.jit
        def host_entrypoint(
            x: cute.Tensor,
            positions: cute.Tensor,
            rope: cute.Tensor,
            w: cute.Tensor,
            ws: cute.Tensor,
            q: cute.Tensor,
            qs: cute.Tensor,
            stream: CUstream,
        ):
            layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((128, 128, 4), stride=(128, 1, 16384)),
            )
            tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.ONE),
                w,
                layout,
                (128, 128),
            )
            device_kernel(x, positions, rope, tma, ws, q, qs).launch(
                grid=(128, 1, 1),
                block=(threads, 1, 1),
                cluster=(8, 1, 1),
                stream=stream,
                use_pdl=True,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self, *, tokens: int, x_stride: int
    ) -> CompileKey:
        return self.CompileKey(tokens=tokens, x_stride=x_stride)

    def get_warmup_keys(
        self, vllm_config: Any, *, max_tokens: int, x_stride: int
    ) -> list[CompileKey]:
        compilation_config = vllm_config.compilation_config
        # Graphs pad batches up to a capture size; eager batches keep theirs.
        padded = 0
        if compilation_config.cudagraph_mode.mixed_mode():
            padded = compilation_config.max_cudagraph_capture_size or 0
        captured = compilation_config.cudagraph_capture_sizes or []
        return self._trace_dispatch(self.dispatch)(
            tokens=[
                *(size for size in captured if size <= min(padded, max_tokens)),
                *range(padded + 1, max_tokens + 1),
            ],
            x_stride=x_stride,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        tokens = compile_key.tokens
        return (
            make_fake_tensor(
                BFloat16,
                (tokens, 16, 512),
                (compile_key.x_stride, 512, 1),
                assumed_align=16,
            ),
            make_fake_tensor(Int64, (tokens,), (1,)),
            make_fake_tensor(Float32, (cute.sym_int(), 64), (64, 1)),
            make_fake_tensor(Float8E4M3FN, (2048, 4096), (4096, 1), assumed_align=16),
            make_fake_tensor(Int32, (2, 1024, 32), (32768, 1, 1024)),
            make_fake_tensor(Float8E4M3FN, (tokens, 2048), (2048, 1)),
            make_fake_tensor(Uint8, (8192,), (1,)),
        )

    @kernel_launcher
    def __call__(
        self,
        *,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> CuTeDSLLaunchSpec["FusedWoAKernel.CompileKey"]:
        """Return FP8 WO-A output and FlashInfer F8_128x4 scale bytes."""
        tokens = x.shape[0]
        q = torch.empty((tokens, 2048), device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(8192, device=x.device, dtype=torch.uint8)
        launch_args = (
            x,
            positions,
            rope,
            weight.view(2048, 4096),
            weight_scale,
            q,
            scales,
        )
        compile_key = self.dispatch(tokens=tokens, x_stride=x.stride(0))
        return compile_key, launch_args, (q, scales)


_FUSED_WO_A_KERNEL = FusedWoAKernel()
