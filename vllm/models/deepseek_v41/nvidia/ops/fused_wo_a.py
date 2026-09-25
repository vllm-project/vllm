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
from vllm.model_executor.warmup.jit_warmup import WarmupIntRange, kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)

# Each head is [nope | rope] as in fused_inv_rope_fp8_quant: only the last
# _ROPE_DIM columns carry RoPE.
_HEAD_DIM = 512
_ROPE_DIM = 64


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
        n_groups: int
        heads_per_group: int
        o_lora_rank: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        tokens = compile_key.tokens
        acc_cols = 1 << (tokens - 1).bit_length()
        tile_m = max(8, acc_cols)
        tile_n = 128
        tile_k = 128
        # Each group's heads split K across one cluster, one head per CTA.
        heads_per_group = compile_key.heads_per_group
        tiles_per_group = compile_key.o_lora_rank // tile_n
        n_tiles = compile_key.n_groups * tiles_per_group
        # Each CTA keeps one whole head of K in SMEM, one TMA box per stage.
        num_stages = _HEAD_DIM // tile_k
        w_stage_bytes = tile_n * tile_k
        nope_dim = _HEAD_DIM - _ROPE_DIM
        threads_per_token = _HEAD_DIM // 4
        sf_col = max(16, tile_m)
        tmem_cols = sf_col * 2
        threads = min(acc_cols * threads_per_token, 512)
        tokens_per_iter = threads // threads_per_token

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
            split = bid % heads_per_group
            tile = bid // heads_per_group
            group = tile // tiles_per_group
            smem = utils.SmemAllocator()
            sW = smem.allocate_tensor(
                Float8E4M3FN,
                w.smem_layout.outer,
                byte_alignment=128,
                swizzle=w.smem_layout.inner,
            )
            sX = smem.allocate_tensor(
                Float8E4M3FN,
                cute.make_layout(
                    (tile_m, tile_k, num_stages), stride=(tile_k, 1, tile_m * tile_k)
                ),
                byte_alignment=1024,
            )
            sW_SF = smem.allocate_tensor(
                Int32, cute.make_layout((tile_n, num_stages)), 128
            )
            sX_SF = smem.allocate_tensor(
                Int32, cute.make_layout((tile_n, num_stages)), 128
            )
            partial = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (tile_n, cute.ceil_div(tokens, heads_per_group), heads_per_group)
                ),
                128,
            )
            mbar_loaded = smem.allocate_array(Int64, num_stages)
            mbar_ws_loaded = smem.allocate_array(Int64, 1)
            mbar_done = smem.allocate_array(Int64, 1)
            mbar_reduced = smem.allocate_array(Int64, 1)
            taddr = smem.allocate(Int32, 4)
            if tid == 0:
                for stage in cutlass.range_constexpr(num_stages):
                    # Wait for weight TMA and this stage's quantization threads.
                    cute.arch.mbarrier_init(mbar_loaded + stage, 1 + threads // 4)
                cute.arch.mbarrier_init(mbar_ws_loaded, tile_n)
                cute.arch.mbarrier_init(mbar_done, 1)
                cute.arch.mbarrier_init(mbar_reduced, 1)
                cute.arch.mbarrier_init_fence()
            # Local users of the barriers (warp 0's TMA below) need the init too.
            cute.arch.sync_threads()
            # Paired with cluster_wait() before the DSMEM exchange, so peers see
            # mbar_reduced initialized before their st.async. The init fence
            # above orders the init before this relaxed arrive, and splitting
            # arrive / wait hides the cluster barrier behind the prologue.
            cute.arch.cluster_arrive_relaxed()
            if warp == 0:
                tiles = cute.zipped_divide(w.tma_tensor, (tile_n, tile_k))
                for stage in cutlass.range_constexpr(num_stages):
                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar_loaded + stage, w_stage_bytes)
                    simple_tma_copy(
                        w.atom,
                        tiles[None, (tile, split * num_stages + stage)],
                        sW[None, None, stage],
                        mbar_loaded + stage,
                    )
            if tid < tile_n:
                for stage in cutlass.range_constexpr(num_stages):
                    # TMA cannot scatter 4-byte scales into tcgen05.cp
                    # 32x128b.warpx4 order; a 4-byte cp.async can, without
                    # stalling the thread on the load.
                    cute.arch.cp_async_shared_global(
                        cute.domain_offset(
                            ((tid % 32) * 4 + tid // 32, stage), sW_SF
                        ).iterator,
                        cute.domain_offset(
                            (
                                group,
                                (tile % tiles_per_group) * tile_n + tid,
                                split * num_stages + stage,
                            ),
                            ws,
                        ).iterator,
                        4,
                        "ca",
                    )
                    sX_SF[tid, stage] = Int32(0)
                    sX32 = cute.recast_tensor(sX, Int32)
                    for row in cutlass.range_constexpr(tile_m // 4):
                        sX32[tid // 32 + row * 4, tid % 32, stage] = Int32(0)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ws_loaded)
            # Order the zeroing before other threads' quantized stores.
            cute.arch.sync_threads()
            # x and positions come from the PDL predecessor; everything above
            # only touches weights.
            cute.arch.griddepcontrol_wait()
            # TMEM is allocated at runtime, and a PDL predecessor on this SM may
            # hold it until it exits; allocating earlier can deadlock.
            if warp == 0:
                cute.arch.alloc_tmem(tmem_cols, taddr)
                cute.arch.relinquish_tmem_alloc_permit()
            x64 = cute.recast_tensor(x, Uint64)
            sX32 = cute.recast_tensor(sX, Uint32)
            sX_SF_bytes = cute.recast_tensor(sX_SF, Uint8)
            for iteration in cutlass.range_constexpr(
                cute.ceil_div(tokens, tokens_per_iter)
            ):
                token = iteration * tokens_per_iter + tid // threads_per_token
                qid = (tid + 32) % threads_per_token
                k = qid * 4
                if cutlass.const_expr(tokens % tokens_per_iter == 0) or token < tokens:
                    packed = x64[token, group * heads_per_group + split, qid]
                    v0, v1 = bf16x2_to_fp32x2(Uint32(packed))
                    v2, v3 = bf16x2_to_fp32x2(Uint32(packed >> 32))
                    if k >= nope_dim:
                        # Inverse RoPE on the interleaved pairs (k, k+1) and
                        # (k+2, k+3); each rope row is cos || sin.
                        pos = positions[token]
                        c = Float32(rope[pos, (k - nope_dim) // 2])
                        s = Float32(rope[pos, (k - nope_dim) // 2 + _ROPE_DIM // 2])
                        v0, v1 = v0 * c + v1 * s, v1 * c - v0 * s
                        c = Float32(rope[pos, (k - nope_dim) // 2 + 1])
                        s = Float32(rope[pos, (k - nope_dim) // 2 + _ROPE_DIM // 2 + 1])
                        v2, v3 = v2 * c + v3 * s, v3 * c - v2 * s
                    amax = cute.arch.fmax(
                        cute.arch.fmax(cute.abs(v0), cute.abs(v1)),
                        cute.arch.fmax(cute.abs(v2), cute.abs(v3)),
                    )
                    amax = cute.arch.warp_reduction_max(amax, threads_in_group=8)
                    exponent, inv = _scale(cute.arch.fmax(amax, Float32(1e-10)))
                    # Recasting drops the pointer swizzle; address it explicitly.
                    sX32[token, (qid % 32) ^ ((token % 8) * 4), qid // 32] = (
                        fp32x4_to_fp8x4(v0 * inv, v1 * inv, v2 * inv, v3 * inv)
                    )
                    if lane % 8 == 0:
                        sX_SF_bytes[
                            token * 16 + k % tile_k // 32 + k // tile_k * 512
                        ] = Uint8(exponent)
            cute.arch.fence_proxy("async.shared", space="cta")
            mbarrier.arrive(mbar_loaded + qid // 32, order="release")
            if warp == 0:
                base = cute.make_tensor(taddr, cute.make_layout(1))[0]
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                sfdesc = Uint64((8 << 32) | (1 << 46))
                idesc = _tcgen05.make_mxfp8_idesc(tile_n, tile_m)
                # cp.async writes through the generic proxy; tcgen05.cp reads
                # through the async proxy.
                cute.arch.mbarrier_wait(mbar_ws_loaded, 0)
                cute.arch.fence_proxy("async.shared", space="cta")
                for stage in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_wait(mbar_loaded + stage, 0)
                    _tcgen05.fence_after_thread_sync()
                    if cutlass.const_expr(stage == num_stages - 1):
                        with cute.arch.elect_one():
                            cute.arch.griddepcontrol_launch_dependents()
                    adesc = sdesc | (sW[None, None, stage].iterator.toint() >> 4)
                    bdesc = sdesc | (sX[None, None, stage].iterator.toint() >> 4)
                    _tcgen05.cp(
                        base + sf_col,
                        sfdesc | (sW_SF[None, stage].iterator.toint() >> 4),
                        "32x128b",
                        "warpx4",
                    )
                    _tcgen05.cp(
                        base + sf_col + 4,
                        sfdesc | (sX_SF[None, stage].iterator.toint() >> 4),
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
                _tcgen05.commit(mbar_done)
                # Only the MMA warp waits; the CTA barrier releases the rest.
                cute.arch.mbarrier_wait(mbar_done, 0)
            cute.arch.sync_threads()
            base = cute.make_tensor(taddr, cute.make_layout(1))[0]
            _tcgen05.fence_after_thread_sync()
            # Tokens t with t % heads_per_group == split arrive from every peer
            # (this CTA included), tile_n Float32 rows each.
            if tid == 0:
                owned = cute.ceil_div(tokens - split, heads_per_group)
                mbarrier.arrive_expect_tx(
                    mbar_reduced, owned * tile_n * heads_per_group * 4
                )
            # Pairs with cluster_arrive_relaxed() above: every peer's
            # mbar_reduced is initialized before the st.async below.
            cute.arch.cluster_wait()
            # quack's mixed const_expr-if rewrite (installed process-wide once
            # quack is imported) can leave these unbound after the dynamic path
            # above; bind them so the regions below join on one type.
            amax, exponent, inv = Float32(0), Uint32(0), Float32(0)
            if tid < tile_n:
                acc = cute.make_rmem_tensor(acc_cols, Float32)
                if cutlass.const_expr(tokens == 1):
                    acc[0] = _tcgen05.ld(warp * 32, base, "32x32b", 1)
                else:
                    acc.store(_tcgen05.ld(warp * 32, base, "32x32b", acc_cols))
                _tcgen05.wait_ld()
                for token in cutlass.range_constexpr(tokens):
                    ptr = cute.domain_offset(
                        (tid, token // heads_per_group, split), partial
                    ).iterator
                    cute.arch.store_async_dsmem(
                        ptr,
                        recast_val(acc[token], Int32),
                        mbar_reduced,
                        token % heads_per_group,
                    )
                if split < tokens:
                    # One warp waits for the peers' bytes; named barrier 1 (0 is
                    # sync_threads) releases the other tile_n epilogue threads.
                    if warp == 0:
                        cute.arch.mbarrier_wait(mbar_reduced, 0)
                    cute.arch.barrier(barrier_id=1, number_of_threads=tile_n)
                    for i in cutlass.range_constexpr(
                        cute.ceil_div(tokens, heads_per_group)
                    ):
                        token = split + i * heads_per_group
                        if (
                            cutlass.const_expr(
                                tokens <= heads_per_group
                                or tokens % heads_per_group == 0
                            )
                            or token < tokens
                        ):
                            value = Float32(0)
                            for peer in cutlass.range_constexpr(heads_per_group):
                                value += partial[tid, i, peer]
                            value = Float32(BFloat16(value))
                            amax = cute.arch.warp_reduction_max(cute.abs(value))
                            exponent, inv = _scale(amax)
                            q[token, tile * tile_n + tid] = Float8E4M3FN(value * inv)
                            if lane == 0:
                                qs[tile * 512 + token * 16 + warp] = Uint8(exponent)
                # Every scale tile owns its padding, with no second memset kernel.
                for i in cutlass.range_constexpr(4):
                    offset = i * tile_n + tid
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
                cute.make_layout(
                    (tile_n, tile_k, num_stages), stride=(tile_k, 1, w_stage_bytes)
                ),
            )
            tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.ONE),
                w,
                layout,
                (tile_n, tile_k),
            )
            device_kernel(x, positions, rope, tma, ws, q, qs).launch(
                grid=(n_tiles * heads_per_group, 1, 1),
                block=(threads, 1, 1),
                cluster=(heads_per_group, 1, 1),
                stream=stream,
                use_pdl=True,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self, *, tokens: int, n_groups: int, heads_per_group: int, o_lora_rank: int
    ) -> CompileKey:
        return self.CompileKey(
            tokens=tokens,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            o_lora_rank=o_lora_rank,
        )

    def get_warmup_keys(
        self, *, max_tokens: int, n_groups: int, heads_per_group: int, o_lora_rank: int
    ) -> list[CompileKey]:
        # Target graphs, draft graphs and eager batches use different token
        # counts, so warm every count the dispatcher accepts.
        return self._trace_dispatch(self.dispatch)(
            tokens=WarmupIntRange(1, max_tokens + 1),
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            o_lora_rank=o_lora_rank,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        tokens, groups = compile_key.tokens, compile_key.n_groups
        rank = compile_key.o_lora_rank
        n = groups * rank
        k = compile_key.heads_per_group * _HEAD_DIM
        return (
            make_fake_tensor(
                BFloat16,
                (tokens, groups * compile_key.heads_per_group, _HEAD_DIM),
                (cute.sym_int(divisibility=8), _HEAD_DIM, 1),
                assumed_align=16,
            ),
            make_fake_tensor(Int64, (tokens,), (1,)),
            make_fake_tensor(Float32, (cute.sym_int(), _ROPE_DIM), (_ROPE_DIM, 1)),
            make_fake_tensor(Float8E4M3FN, (n, k), (k, 1), assumed_align=16),
            make_fake_tensor(
                Int32, (groups, rank, k // 128), (rank * k // 128, 1, rank)
            ),
            make_fake_tensor(Float8E4M3FN, (tokens, n), (n, 1)),
            make_fake_tensor(Uint8, (128 * n // 32,), (1,)),
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
        groups, rank = weight_scale.shape[:2]
        n = groups * rank
        q = torch.empty((tokens, n), device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(128 * n // 32, device=x.device, dtype=torch.uint8)
        launch_args = (
            x,
            positions,
            rope,
            weight.view(n, -1),
            weight_scale,
            q,
            scales,
        )
        compile_key = self.dispatch(
            tokens=tokens,
            n_groups=groups,
            heads_per_group=x.shape[1] // groups,
            o_lora_rank=rank,
        )
        return compile_key, launch_args, (q, scales)


_FUSED_WO_A_KERNEL = FusedWoAKernel()
