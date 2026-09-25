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
        # tcgen05 MXFP8 MMA consumes K = 32, one MX block, per instruction.
        mma_k = 32
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
            ws: cpasync.TmaInfo,
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
                swizzle=cute.make_swizzle(3, 4, 3),
            )
            # tcgen05.cp 32x128b.warpx4 order: row r at byte (r % 32) * 16 +
            # (r // 32) * 4, one 4-byte cell of MX-block scales per row.
            sW_SF = smem.allocate_tensor(
                Int32,
                cute.make_layout(((32, 4), num_stages), stride=((4, 1), tile_n)),
                128,
            )
            sW_SF_raw = smem.allocate_tensor(
                Int32, cute.make_layout((tile_n, num_stages)), 128
            )
            sX_SF = smem.allocate_tensor(
                Uint8,
                cute.make_layout(
                    ((32, 4), tile_k // mma_k, num_stages),
                    stride=((16, 4), 1, tile_n * 4),
                ),
                128,
            )
            copy_fp8x4 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float8E4M3FN, num_bits_per_copy=32
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
                cute.arch.mbarrier_init(mbar_ws_loaded, 1)
                cute.arch.mbarrier_init(mbar_done, 1)
                cute.arch.mbarrier_init(mbar_reduced, 1)
                cute.arch.mbarrier_init_fence()
            # Local users of the barriers (warp 0's TMA below) need the init too.
            cute.arch.sync_threads()
            # Paired with cluster_wait() below: peers must see mbar_reduced
            # initialized before their st.async; the split hides the barrier.
            cute.arch.cluster_arrive_relaxed()
            if warp == 0:
                # Raw DeepGEMM MN-major scales; the MMA warp transposes them to
                # the UTCCP order, which TMA cannot scatter at 4-byte granularity.
                ws_tiles = cute.zipped_divide(ws.tma_tensor, (tile_n, num_stages))
                with cute.arch.elect_one():
                    mbarrier.arrive_expect_tx(mbar_ws_loaded, tile_n * num_stages * 4)
                simple_tma_copy(
                    ws.atom,
                    ws_tiles[None, (tile % tiles_per_group, split, group)],
                    sW_SF_raw,
                    mbar_ws_loaded,
                )
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
            # x and positions come from the PDL predecessor; everything above
            # only touches weights. Padded sX rows only feed unread MMA columns.
            cute.arch.griddepcontrol_wait()
            # TMEM is allocated at runtime, and a PDL predecessor on this SM may
            # hold it until it exits; allocating earlier can deadlock.
            if warp == 0:
                cute.arch.alloc_tmem(tmem_cols, taddr)
                cute.arch.relinquish_tmem_alloc_permit()
            copy_x = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=64
            )
            n_iters = cute.ceil_div(tokens, tokens_per_iter)
            qid = (tid + 32) % threads_per_token
            k = qid * 4
            # Issue every token's x load before any math so their latencies overlap.
            x_bf16 = cute.make_rmem_tensor((4, n_iters), BFloat16)
            for iteration in cutlass.range_constexpr(n_iters):
                token = iteration * tokens_per_iter + tid // threads_per_token
                if cutlass.const_expr(tokens % tokens_per_iter == 0) or token < tokens:
                    x_src = cute.local_tile(
                        x[token, group * heads_per_group + split, None], (4,), (qid,)
                    )
                    cute.copy(copy_x, x_src, x_bf16[None, iteration])
            for iteration in cutlass.range_constexpr(n_iters):
                token = iteration * tokens_per_iter + tid // threads_per_token
                if cutlass.const_expr(tokens % tokens_per_iter == 0) or token < tokens:
                    x_f32 = cute.make_rmem_tensor((4,), Float32)
                    x_f32.store(x_bf16[None, iteration].load().to(Float32))
                    if k >= nope_dim:
                        # Inverse RoPE on the interleaved pairs (k, k+1) and
                        # (k+2, k+3); each rope row is cos || sin.
                        pos = positions[token]
                        for pair in cutlass.range_constexpr(2):
                            freq = (k - nope_dim) // 2 + pair
                            cos = Float32(rope[pos, freq])
                            sin = Float32(rope[pos, freq + _ROPE_DIM // 2])
                            even, odd = x_f32[2 * pair], x_f32[2 * pair + 1]
                            x_f32[2 * pair] = even * cos + odd * sin
                            x_f32[2 * pair + 1] = odd * cos - even * sin
                    amax = cute.arch.fmax(
                        cute.arch.fmax(cute.abs(x_f32[0]), cute.abs(x_f32[1])),
                        cute.arch.fmax(cute.abs(x_f32[2]), cute.abs(x_f32[3])),
                    )
                    amax = cute.arch.warp_reduction_max(amax, threads_in_group=8)
                    exponent, inv = _scale(cute.arch.fmax(amax, Float32(1e-10)))
                    x_fp8 = cute.make_rmem_tensor((4,), Float8E4M3FN)
                    x_fp8.store((x_f32.load() * inv).to(Float8E4M3FN))
                    cute.copy(
                        copy_fp8x4,
                        x_fp8,
                        cute.local_tile(
                            sX[token, None, k // tile_k], (4,), (qid % 32,)
                        ),
                    )
                    if lane % 8 == 0:
                        sX_SF[token, k % tile_k // mma_k, k // tile_k] = Uint8(exponent)
            cute.arch.fence_proxy("async.shared", space="cta")
            mbarrier.arrive(mbar_loaded + qid // 32, order="release")
            if warp == 0:
                base = cute.make_tensor(taddr, cute.make_layout(1))[0]
                # sW / sX are 128B-swizzled K-major; one stage's K fills the
                # swizzle atom, so the leading byte offset is unused.
                sdesc = _tcgen05.make_sdesc_128B_swizzle(LBO=0)
                # Unswizzled 32 x 16 B scale tiles: SBO = one 8-row core matrix
                # (16 B units at bit 32); bit 46 is the SM100 descriptor version.
                sf_sbo = 8 * 16
                sfdesc = Uint64((sf_sbo >> 4 << 32) | (1 << 46))
                # M = tile_n weight rows (A), N = tile_m token rows (B).
                idesc = _tcgen05.make_mxfp8_idesc(tile_n, tile_m)
                # sW_SF's layout is the UTCCP order, so a plain copy transposes;
                # fence the generic-proxy writes before tcgen05.cp reads them.
                cute.arch.mbarrier_wait(mbar_ws_loaded, 0)
                for stage in cutlass.range_constexpr(num_stages):
                    for i in cutlass.range_constexpr(4):
                        sW_SF[i * 32 + lane, stage] = sW_SF_raw[i * 32 + lane, stage]
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
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
                        sfdesc | (sX_SF[None, None, stage].iterator.toint() >> 4),
                        "32x128b",
                        "warpx4",
                    )
                    # Advance mma_k FP8 bytes (16 B units) and pick MX block kk of
                    # each 4-byte scale (B sf_id at bit 4, A sf_id at bit 29).
                    for kk in cutlass.range_constexpr(tile_k // mma_k):
                        _tcgen05.mma_mxfp8(
                            base,
                            adesc + kk * mma_k // 16,
                            bdesc + kk * mma_k // 16,
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
            # quack's process-wide const_expr-if rewrite can leave these unbound
            # after the dynamic path above; bind them so the regions join.
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
            ws_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.ONE),
                cute.make_tensor(ws.iterator, cute.select(ws.layout, mode=[1, 2, 0])),
                cute.make_layout((tile_n, num_stages)),
                (tile_n, num_stages),
            )
            device_kernel(x, positions, rope, tma, ws_tma, q, qs).launch(
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
                Int32,
                (groups, rank, k // 128),
                (rank * k // 128, 1, rank),
                assumed_align=16,
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
