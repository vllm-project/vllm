# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL sparse-attention compressor for DeepSeek V4.

The public wrappers provide the C4 fused and C128 split kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import torch_to_cute_dtype
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import compile_cutedsl
from vllm.utils.math_utils import round_up


@dsl_user_op
def _recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_to_fp32(data: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [data.ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _fp32x2_to_fp8e4m3x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint16:
    out = llvm.inline_asm(
        T.i16(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint16(out)


class SparseAttnCompressNormRopeStoreC4Kernel(
    VllmJitKernel["SparseAttnCompressNormRopeStoreC4Kernel.CompileKey"]
):
    min_scale = 1.0e-4
    rcp_ln2 = 1.4426950408889634

    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        state_width: int
        rope_head_dim: int
        fp8_max: float
        quant_block: int
        token_stride: int
        scale_dim: int
        compress_ratio: int
        overlap: bool
        norm_weight_dtype: type[cutlass.Numeric]

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        min_scale = SparseAttnCompressNormRopeStoreC4Kernel.min_scale
        rcp_ln2 = SparseAttnCompressNormRopeStoreC4Kernel.rcp_ln2
        head_dim = compile_key.head_size
        state_width = compile_key.state_width
        rope_dim = compile_key.rope_head_dim
        nope_dim = head_dim - rope_dim
        fp8_max = compile_key.fp8_max
        quant_block = compile_key.quant_block
        token_stride = compile_key.token_stride
        scale_dim = compile_key.scale_dim
        elems_per_lane = 8
        copy_elems = 4
        copy_chunks = elems_per_lane // copy_elems
        lanes_per_group = quant_block // elems_per_lane
        groups_per_warp = 32 // lanes_per_group
        scale_reduce_steps = lanes_per_group.bit_length() - 1
        scale_reduce_offset = lanes_per_group // 2
        num_warps = (head_dim // quant_block) // groups_per_warp
        nope_blocks = nope_dim // quant_block
        tb_size = num_warps * 32
        compress_ratio = compile_key.compress_ratio
        overlap = compile_key.overlap
        window = (1 + int(overlap)) * compress_ratio

        @cute.kernel
        def device_kernel(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            block_size: Int64,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
        ):
            token_idx, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            warp_id = cute.arch.make_warp_uniform(tid // 32)
            lane_id = tid % 32
            group_lane = lane_id % lanes_per_group
            group_idx = warp_id * groups_per_warp + lane_id // lanes_per_group
            elem_base = group_idx * quant_block + group_lane * elems_per_lane

            slot_id = slot_mapping[token_idx]
            has_position = token_idx < positions.shape[0]
            position = Int64(0)
            if has_position:
                position = positions[token_idx]
            boundary = has_position and (
                (position + Int64(1)) % Int64(compress_ratio) == Int64(0)
            )
            has_req_idx = token_idx < token_to_req_indices.shape[0]
            has_kv_slot_idx = token_idx < kv_slot_mapping.shape[0]
            kv_slot_idx = Int64(-1)
            if has_kv_slot_idx:
                kv_slot_idx = kv_slot_mapping[token_idx]
            active = (
                slot_id >= Int64(0)
                and has_req_idx
                and boundary
                and kv_slot_idx >= Int64(0)
            )

            if active:
                req_idx = token_to_req_indices[token_idx]
                start = position - Int64(window - 1)

                smem = cutlass.utils.SmemAllocator()
                s_block_numbers = smem.allocate_tensor(
                    Int32, cute.make_layout((window,)), byte_alignment=4
                )
                partial_sums = smem.allocate_tensor(
                    Float32, cute.make_layout((num_warps,)), byte_alignment=4
                )
                rrms_shared = smem.allocate_tensor(
                    Float32, cute.make_layout((1,)), byte_alignment=4
                )

                for row in cutlass.range_constexpr(window):
                    pos = start + Int64(row)
                    if tid == row:
                        block_number_i32 = Int32(0)
                        if pos >= Int64(0):
                            block_index = pos // block_size
                            block_number_i32 = block_table[req_idx, block_index]
                        s_block_numbers[row] = block_number_i32
                cute.arch.sync_threads()

                local_max = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_sum = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_product = cute.make_rmem_tensor((elems_per_lane,), Float32)

                for e in cutlass.range_constexpr(elems_per_lane):
                    local_max[e] = -Float32.inf
                    local_sum[e] = Float32(0.0)
                    local_product[e] = Float32(0.0)

                cp_f32x4 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                )
                copy_layout = cute.make_layout(
                    (copy_chunks, copy_elems),
                    stride=(copy_elems, 1),
                )
                kv_vals = cute.make_rmem_tensor(copy_layout, Float32)
                score_vals = cute.make_rmem_tensor(copy_layout, Float32)

                for row in cutlass.range_constexpr(window):
                    pos = start + Int64(row)
                    if pos >= Int64(0):
                        block_index = pos // block_size
                        block_offset = pos - block_index * block_size
                        block_number = s_block_numbers[row].to(Int64)
                        head_offset = Int64((row // compress_ratio) * head_dim)
                        row_tensor = state_cache[block_number, block_offset, None]
                        for chunk in cutlass.range_constexpr(copy_chunks):
                            copy_elem = const_expr(chunk * copy_elems)
                            col_tile = (
                                head_offset + (elem_base + Int32(copy_elem)).to(Int64)
                            ) // Int64(copy_elems)
                            kv_src = cute.local_tile(
                                row_tensor,
                                tiler=(copy_elems,),
                                coord=(col_tile,),
                            )
                            score_src = cute.local_tile(
                                row_tensor,
                                tiler=(copy_elems,),
                                coord=(col_tile + Int64(state_width // copy_elems),),
                            )
                            cute.copy(cp_f32x4, kv_src, kv_vals[chunk, None])
                            cute.copy(cp_f32x4, score_src, score_vals[chunk, None])

                        for e in cutlass.range_constexpr(elems_per_lane):
                            chunk = const_expr(e // copy_elems)
                            copy_elem = const_expr(e % copy_elems)
                            score = score_vals[chunk, copy_elem]
                            kv = kv_vals[chunk, copy_elem]
                            new_max = cute.arch.fmax(local_max[e], score)
                            old_scale = cute.math.exp2(
                                (local_max[e] - new_max) * Float32(rcp_ln2),
                                fastmath=True,
                            )
                            new_scale = cute.math.exp2(
                                (score - new_max) * Float32(rcp_ln2),
                                fastmath=True,
                            )
                            local_sum[e] = local_sum[e] * old_scale + new_scale
                            local_product[e] = (
                                local_product[e] * old_scale + kv * new_scale
                            )
                            local_max[e] = new_max

                x = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_sumsq = Float32(0.0)
                for e in cutlass.range_constexpr(elems_per_lane):
                    x[e] = local_product[e] / local_sum[e]
                    local_sumsq += x[e] * x[e]

                warp_sum = local_sumsq
                for step in cutlass.range_constexpr(5):
                    offset = const_expr(16 >> step)
                    warp_sum += cute.arch.shuffle_sync_bfly(warp_sum, offset)

                if lane_id == 0:
                    partial_sums[warp_id] = warp_sum
                cute.arch.sync_threads()
                if tid == 0:
                    total = Float32(0.0)
                    for i in cutlass.range_constexpr(num_warps):
                        total += partial_sums[i]
                    rrms_shared[0] = cute.math.rsqrt(
                        total / Float32(head_dim) + rms_norm_eps, fastmath=True
                    )
                cute.arch.sync_threads()

                rrms = rrms_shared[0]
                for e in cutlass.range_constexpr(elems_per_lane):
                    elem = elem_base + e
                    x[e] = x[e] * rrms * rms_norm_weight[elem].to(Float32)

                k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
                k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
                page = kv_slot_idx // kv_cache_block_size
                kv_offset = kv_slot_idx - page * kv_cache_block_size
                value_base = page * k_cache.stride[0] + kv_offset * Int64(token_stride)
                scale_base = (
                    page * k_cache.stride[0]
                    + kv_cache_block_size * Int64(token_stride)
                    + kv_offset * Int64(scale_dim)
                )

                if group_idx == nope_blocks:
                    compressed_pos = (position // Int64(compress_ratio)) * Int64(
                        compress_ratio
                    )
                    for pair in cutlass.range_constexpr(elems_per_lane // 2):
                        elem = const_expr(pair * 2)
                        pair_idx = (elem_base - nope_dim) // 2 + Int32(pair)
                        cos_v = cos_sin_cache[compressed_pos, pair_idx]
                        sin_v = cos_sin_cache[
                            compressed_pos, pair_idx + Int32(rope_dim // 2)
                        ]
                        real = x[elem] * cos_v - x[elem + 1] * sin_v
                        imag = x[elem] * sin_v + x[elem + 1] * cos_v
                        packed = _fp32x2_to_bf16x2(real, imag)
                        out_base = (
                            value_base
                            + Int64(nope_dim)
                            + ((elem_base - nope_dim + Int32(elem)) * 2).to(Int64)
                        )
                        k_cache_u32.iterator[out_base // Int64(4)] = packed
                else:
                    q = cute.make_rmem_tensor((elems_per_lane,), Float32)
                    local_absmax = Float32(0.0)
                    for pair in cutlass.range_constexpr(elems_per_lane // 2):
                        elem = const_expr(pair * 2)
                        q_packed = _fp32x2_to_bf16x2(x[elem], x[elem + 1])
                        q0, q1 = _bf16x2_to_fp32(q_packed)
                        q[elem] = q0
                        q[elem + 1] = q1
                        local_absmax = cute.arch.fmax(
                            local_absmax,
                            cute.arch.fmax(cute.math.absf(q0), cute.math.absf(q1)),
                        )
                    absmax = local_absmax
                    group_mask_and_clamp = const_expr(
                        (cute.arch.WARP_SIZE - lanes_per_group) << 8
                        | (cute.arch.WARP_SIZE - 1)
                    )
                    for step in cutlass.range_constexpr(scale_reduce_steps):
                        offset = const_expr(scale_reduce_offset >> step)
                        absmax = cute.arch.fmax(
                            absmax,
                            cute.arch.shuffle_sync_bfly(
                                absmax,
                                offset=offset,
                                mask_and_clamp=group_mask_and_clamp,
                            ),
                        )
                    scale_raw = cute.arch.fmax(
                        Float32(min_scale),
                        absmax / Float32(fp8_max),
                    )
                    bits = _recast_val(scale_raw, Uint32)
                    ue8m0 = ((bits + Uint32(0x7FFFFF)) >> Uint32(23)) & Uint32(0xFF)
                    inv_scale = _recast_val(
                        (Uint32(254) - ue8m0) << Uint32(23), Float32
                    )
                    for pair in cutlass.range_constexpr(elems_per_lane // 2):
                        elem = const_expr(pair * 2)
                        y0 = cutlass.min(
                            cute.arch.fmax(q[elem] * inv_scale, Float32(-fp8_max)),
                            Float32(fp8_max),
                        )
                        y1 = cutlass.min(
                            cute.arch.fmax(q[elem + 1] * inv_scale, Float32(-fp8_max)),
                            Float32(fp8_max),
                        )
                        packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                        out_base = value_base + (elem_base + Int32(elem)).to(Int64)
                        k_cache_u16.iterator[out_base // Int64(2)] = packed_fp8
                    if group_lane == 0:
                        k_cache.iterator[scale_base + group_idx.to(Int64)] = ue8m0.to(
                            Uint8
                        )
                        if group_idx == 0:
                            k_cache.iterator[scale_base + Int64(nope_blocks)] = Uint8(0)

        @cute.jit
        def host_entrypoint(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            block_size: Int64,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
            stream: CUstream,
        ):
            grid = (slot_mapping.shape[0], 1, 1)
            device_kernel(
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                block_size,
                rms_norm_weight,
                rms_norm_eps,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
                kv_cache_block_size,
            ).launch(grid=grid, block=(tb_size, 1, 1), stream=stream)

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        norm_weight_dtype: type[cutlass.Numeric],
        head_size: int,
        rope_head_dim: int,
    ) -> CompileKey:
        overlap = compress_ratio == 4
        state_width = head_size * (2 if overlap else 1)
        return self.CompileKey(
            head_size=head_size,
            state_width=state_width,
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            compress_ratio=compress_ratio,
            overlap=overlap,
            norm_weight_dtype=norm_weight_dtype,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=4,
            norm_weight_dtype=torch_to_cute_dtype(vllm_config.model_config.dtype),
            head_size=hf_config.head_dim,
            rope_head_dim=hf_config.qk_rope_head_dim,
            enabled=(
                cache_config.block_size > 0
                and 4 in hf_config.compress_ratios
                and cache_config.cache_dtype == "fp8_ds_mla"
            ),
            _when=lambda *, enabled: enabled,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        head_size = compile_key.head_size
        state_width = compile_key.state_width
        rope_head_dim = compile_key.rope_head_dim
        quant_block = compile_key.quant_block
        token_stride = compile_key.token_stride
        scale_dim = compile_key.scale_dim
        compress_ratio = compile_key.compress_ratio
        overlap = compile_key.overlap
        norm_weight_dtype = compile_key.norm_weight_dtype
        if compress_ratio != 4 or not overlap:
            raise ValueError("CuTe DSL C4 fused sparse-attn requires C4 overlap.")
        if head_size != 512:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires head_size=512.",
            )
        if state_width != 2 * head_size:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn requires state_width=2*head_size.",
            )
        if quant_block != 64:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires quant_block=64.",
            )
        if rope_head_dim != 64:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires rope_head_dim=64.",
            )
        if token_stride < head_size + rope_head_dim:
            raise ValueError("token_stride is too small for the packed FP8/BF16 row.")
        expected_scale_dim = (head_size - rope_head_dim) // quant_block + 1
        if scale_dim < expected_scale_dim:
            raise ValueError("scale_dim is too small for the UE8M0 scale row.")
        num_positions = cute.sym_int()
        num_slots = cute.sym_int()
        num_req_indices = cute.sym_int()
        num_kv_slots = cute.sym_int()
        num_state_blocks = cute.sym_int()
        num_kv_blocks = cute.sym_int()
        state_cache_block_size = cute.sym_int()
        block_table_width = cute.sym_int()
        max_pos = cute.sym_int()
        state_cache_width = state_width * 2
        state_cache = cute.runtime.make_fake_tensor(
            Float32,
            (num_state_blocks, state_cache_block_size, state_cache_width),
            stride=(
                cute.sym_int64(divisibility=16),
                cute.sym_int64(divisibility=16),
                1,
            ),
            assumed_align=16,
        )
        token_to_req_indices = make_fake_tensor(
            Int32,
            (num_req_indices,),
            divisibility=4,
        )
        positions = make_fake_tensor(Int64, (num_positions,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_slots,), divisibility=8)
        block_table = make_fake_tensor(
            Int32,
            (cute.sym_int(), block_table_width),
            divisibility=1,
        )
        rms_norm_weight = make_fake_tensor(
            norm_weight_dtype,
            (head_size,),
            divisibility=4,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            Float32,
            (max_pos, rope_head_dim),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=4,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (num_kv_blocks, cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(divisibility=16), cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        kv_slot_mapping = make_fake_tensor(Int64, (num_kv_slots,), divisibility=8)
        host_entrypoint = self.kernel(compile_key)

        compiled = compile_cutedsl(
            host_entrypoint,
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            Int64(0),
            rms_norm_weight,
            Float32(0.0),
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            Int64(0),
        )
        self._compiled_cache[compile_key] = compiled

    def __call__(
        self,
        *,
        state_cache: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
    ) -> Any:
        compile_key = self.dispatch(
            compress_ratio=compress_ratio,
            norm_weight_dtype=torch_to_cute_dtype(rms_norm_weight.dtype),
            head_size=head_dim,
            rope_head_dim=rope_head_dim,
        )
        runtime_context = {
            "kv_cache_shape": tuple(kv_cache.shape),
            "kv_cache_stride": tuple(kv_cache.stride()),
            "compress_ratio": compress_ratio,
        }
        kernel = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        return kernel(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache,
            kv_slot_mapping,
            kv_cache.shape[1],
        )


class SparseAttnCompressNormRopeStoreFullC4Kernel(
    VllmJitKernel["SparseAttnCompressNormRopeStoreFullC4Kernel.CompileKey"]
):
    rcp_ln2 = 1.4426950408889634

    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        state_width: int
        rope_head_dim: int
        fp8_max: float
        quant_block: int
        compress_ratio: int
        overlap: bool
        store_full_fp8: bool
        norm_weight_dtype: type[cutlass.Numeric]

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        rcp_ln2 = SparseAttnCompressNormRopeStoreFullC4Kernel.rcp_ln2
        head_dim = compile_key.head_size
        state_width = compile_key.state_width
        rope_dim = compile_key.rope_head_dim
        nope_dim = head_dim - rope_dim
        fp8_max = compile_key.fp8_max
        quant_block = compile_key.quant_block
        elems_per_lane = 8
        copy_elems = 4
        copy_chunks = elems_per_lane // copy_elems
        lanes_per_group = quant_block // elems_per_lane
        groups_per_warp = 32 // lanes_per_group
        num_warps = (head_dim // quant_block) // groups_per_warp
        nope_blocks = nope_dim // quant_block
        tb_size = num_warps * 32
        compress_ratio = compile_key.compress_ratio
        overlap = compile_key.overlap
        window = (1 + int(overlap)) * compress_ratio
        store_full_fp8 = compile_key.store_full_fp8

        @cute.kernel
        def device_kernel(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            block_size: Int64,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
            fp8_scale: cute.Tensor,
        ):
            token_idx, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            warp_id = cute.arch.make_warp_uniform(tid // 32)
            lane_id = tid % 32
            group_lane = lane_id % lanes_per_group
            group_idx = warp_id * groups_per_warp + lane_id // lanes_per_group
            elem_base = group_idx * quant_block + group_lane * elems_per_lane

            slot_id = slot_mapping[token_idx]
            has_position = token_idx < positions.shape[0]
            position = Int64(0)
            if has_position:
                position = positions[token_idx]
            boundary = has_position and (
                (position + Int64(1)) % Int64(compress_ratio) == Int64(0)
            )
            has_req_idx = token_idx < token_to_req_indices.shape[0]
            has_kv_slot_idx = token_idx < kv_slot_mapping.shape[0]
            kv_slot_idx = Int64(-1)
            if has_kv_slot_idx:
                kv_slot_idx = kv_slot_mapping[token_idx]
            active = (
                slot_id >= Int64(0)
                and has_req_idx
                and boundary
                and kv_slot_idx >= Int64(0)
            )

            if active:
                req_idx = token_to_req_indices[token_idx]
                start = position - Int64(window - 1)

                smem = cutlass.utils.SmemAllocator()
                s_block_numbers = smem.allocate_tensor(
                    Int32, cute.make_layout((window,)), byte_alignment=4
                )
                partial_sums = smem.allocate_tensor(
                    Float32, cute.make_layout((num_warps,)), byte_alignment=4
                )
                rrms_shared = smem.allocate_tensor(
                    Float32, cute.make_layout((1,)), byte_alignment=4
                )

                for row in cutlass.range_constexpr(window):
                    pos = start + Int64(row)
                    if tid == row:
                        block_number_i32 = Int32(0)
                        if pos >= Int64(0):
                            block_index = pos // block_size
                            block_number_i32 = block_table[req_idx, block_index]
                        s_block_numbers[row] = block_number_i32
                cute.arch.sync_threads()

                local_max = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_sum = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_product = cute.make_rmem_tensor((elems_per_lane,), Float32)

                for e in cutlass.range_constexpr(elems_per_lane):
                    local_max[e] = -Float32.inf
                    local_sum[e] = Float32(0.0)
                    local_product[e] = Float32(0.0)

                cp_f32x4 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                )
                copy_layout = cute.make_layout(
                    (copy_chunks, copy_elems),
                    stride=(copy_elems, 1),
                )
                kv_vals = cute.make_rmem_tensor(copy_layout, Float32)
                score_vals = cute.make_rmem_tensor(copy_layout, Float32)

                for row in cutlass.range_constexpr(window):
                    pos = start + Int64(row)
                    if pos >= Int64(0):
                        block_index = pos // block_size
                        block_offset = pos - block_index * block_size
                        block_number = s_block_numbers[row].to(Int64)
                        head_offset = Int64((row // compress_ratio) * head_dim)
                        row_tensor = state_cache[block_number, block_offset, None]
                        for chunk in cutlass.range_constexpr(copy_chunks):
                            copy_elem = const_expr(chunk * copy_elems)
                            col_tile = (
                                head_offset + (elem_base + Int32(copy_elem)).to(Int64)
                            ) // Int64(copy_elems)
                            kv_src = cute.local_tile(
                                row_tensor,
                                tiler=(copy_elems,),
                                coord=(col_tile,),
                            )
                            score_src = cute.local_tile(
                                row_tensor,
                                tiler=(copy_elems,),
                                coord=(col_tile + Int64(state_width // copy_elems),),
                            )
                            cute.copy(cp_f32x4, kv_src, kv_vals[chunk, None])
                            cute.copy(cp_f32x4, score_src, score_vals[chunk, None])

                        for e in cutlass.range_constexpr(elems_per_lane):
                            chunk = const_expr(e // copy_elems)
                            copy_elem = const_expr(e % copy_elems)
                            score = score_vals[chunk, copy_elem]
                            kv = kv_vals[chunk, copy_elem]
                            new_max = cute.arch.fmax(local_max[e], score)
                            old_scale = cute.math.exp2(
                                (local_max[e] - new_max) * Float32(rcp_ln2),
                                fastmath=True,
                            )
                            new_scale = cute.math.exp2(
                                (score - new_max) * Float32(rcp_ln2),
                                fastmath=True,
                            )
                            local_sum[e] = local_sum[e] * old_scale + new_scale
                            local_product[e] = (
                                local_product[e] * old_scale + kv * new_scale
                            )
                            local_max[e] = new_max

                x = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_sumsq = Float32(0.0)
                for e in cutlass.range_constexpr(elems_per_lane):
                    x[e] = local_product[e] / local_sum[e]
                    local_sumsq += x[e] * x[e]

                warp_sum = local_sumsq
                for step in cutlass.range_constexpr(5):
                    offset = const_expr(16 >> step)
                    warp_sum += cute.arch.shuffle_sync_bfly(warp_sum, offset)

                if lane_id == 0:
                    partial_sums[warp_id] = warp_sum
                cute.arch.sync_threads()
                if tid == 0:
                    total = Float32(0.0)
                    for i in cutlass.range_constexpr(num_warps):
                        total += partial_sums[i]
                    rrms_shared[0] = cute.math.rsqrt(
                        total / Float32(head_dim) + rms_norm_eps, fastmath=True
                    )
                cute.arch.sync_threads()

                rrms = rrms_shared[0]
                for e in cutlass.range_constexpr(elems_per_lane):
                    elem = elem_base + e
                    x[e] = x[e] * rrms * rms_norm_weight[elem].to(Float32)

                page = kv_slot_idx // kv_cache_block_size
                kv_offset = kv_slot_idx - page * kv_cache_block_size
                value_base = page * k_cache.stride[0] + kv_offset * k_cache.stride[1]

                if const_expr(store_full_fp8):
                    k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
                    inv_fp8 = Float32(1.0) / fp8_scale[0]
                    if group_idx == nope_blocks:
                        compressed_pos = (position // Int64(compress_ratio)) * Int64(
                            compress_ratio
                        )
                        for pair in cutlass.range_constexpr(elems_per_lane // 2):
                            elem = const_expr(pair * 2)
                            pair_idx = (elem_base - nope_dim) // 2 + Int32(pair)
                            cos_v = cos_sin_cache[compressed_pos, pair_idx]
                            sin_v = cos_sin_cache[
                                compressed_pos, pair_idx + Int32(rope_dim // 2)
                            ]
                            real = x[elem] * cos_v - x[elem + 1] * sin_v
                            imag = x[elem] * sin_v + x[elem + 1] * cos_v
                            packed_bf16 = _fp32x2_to_bf16x2(real, imag)
                            b0, b1 = _bf16x2_to_fp32(packed_bf16)
                            y0 = cutlass.min(
                                cutlass.max(b0 * inv_fp8, Float32(-fp8_max)),
                                Float32(fp8_max),
                            )
                            y1 = cutlass.min(
                                cutlass.max(b1 * inv_fp8, Float32(-fp8_max)),
                                Float32(fp8_max),
                            )
                            packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                            out_base = value_base + (elem_base + Int32(elem)).to(Int64)
                            k_cache_u16.iterator[out_base // Int64(2)] = packed_fp8
                    else:
                        for pair in cutlass.range_constexpr(elems_per_lane // 2):
                            elem = const_expr(pair * 2)
                            packed_bf16 = _fp32x2_to_bf16x2(x[elem], x[elem + 1])
                            b0, b1 = _bf16x2_to_fp32(packed_bf16)
                            y0 = cutlass.min(
                                cutlass.max(b0 * inv_fp8, Float32(-fp8_max)),
                                Float32(fp8_max),
                            )
                            y1 = cutlass.min(
                                cutlass.max(b1 * inv_fp8, Float32(-fp8_max)),
                                Float32(fp8_max),
                            )
                            packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                            out_base = value_base + (elem_base + Int32(elem)).to(Int64)
                            k_cache_u16.iterator[out_base // Int64(2)] = packed_fp8
                else:
                    k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
                    if group_idx == nope_blocks:
                        compressed_pos = (position // Int64(compress_ratio)) * Int64(
                            compress_ratio
                        )
                        for pair in cutlass.range_constexpr(elems_per_lane // 2):
                            elem = const_expr(pair * 2)
                            pair_idx = (elem_base - nope_dim) // 2 + Int32(pair)
                            cos_v = cos_sin_cache[compressed_pos, pair_idx]
                            sin_v = cos_sin_cache[
                                compressed_pos, pair_idx + Int32(rope_dim // 2)
                            ]
                            real = x[elem] * cos_v - x[elem + 1] * sin_v
                            imag = x[elem] * sin_v + x[elem + 1] * cos_v
                            packed_bf16 = _fp32x2_to_bf16x2(real, imag)
                            out_base = value_base + ((elem_base + Int32(elem)) * 2).to(
                                Int64
                            )
                            k_cache_u32.iterator[out_base // Int64(4)] = packed_bf16
                    else:
                        for pair in cutlass.range_constexpr(elems_per_lane // 2):
                            elem = const_expr(pair * 2)
                            packed_bf16 = _fp32x2_to_bf16x2(x[elem], x[elem + 1])
                            out_base = value_base + ((elem_base + Int32(elem)) * 2).to(
                                Int64
                            )
                            k_cache_u32.iterator[out_base // Int64(4)] = packed_bf16

        @cute.jit
        def host_entrypoint(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            block_size: Int64,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
            fp8_scale: cute.Tensor,
            stream: CUstream,
        ):
            grid = (slot_mapping.shape[0], 1, 1)
            device_kernel(
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                block_size,
                rms_norm_weight,
                rms_norm_eps,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
                kv_cache_block_size,
                fp8_scale,
            ).launch(grid=grid, block=(tb_size, 1, 1), stream=stream)

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        store_full_fp8: bool,
        norm_weight_dtype: type[cutlass.Numeric],
        head_size: int,
        rope_head_dim: int,
    ) -> CompileKey:
        overlap = compress_ratio == 4
        return self.CompileKey(
            head_size=head_size,
            state_width=head_size * (2 if overlap else 1),
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=compress_ratio,
            overlap=overlap,
            store_full_fp8=store_full_fp8,
            norm_weight_dtype=norm_weight_dtype,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=4,
            store_full_fp8=cache_config.cache_dtype in ("fp8", "fp8_e4m3"),
            norm_weight_dtype=torch_to_cute_dtype(vllm_config.model_config.dtype),
            head_size=hf_config.head_dim,
            rope_head_dim=hf_config.qk_rope_head_dim,
            enabled=(
                cache_config.block_size > 0
                and 4 in hf_config.compress_ratios
                and cache_config.cache_dtype != "fp8_ds_mla"
            ),
            _when=lambda *, enabled: enabled,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        head_size = compile_key.head_size
        state_width = compile_key.state_width
        rope_head_dim = compile_key.rope_head_dim
        quant_block = compile_key.quant_block
        compress_ratio = compile_key.compress_ratio
        overlap = compile_key.overlap
        norm_weight_dtype = compile_key.norm_weight_dtype
        if compress_ratio != 4 or not overlap:
            raise ValueError("CuTe DSL C4 fused sparse-attn requires C4 overlap.")
        if head_size != 512:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires head_size=512.",
            )
        if state_width != 2 * head_size:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn requires state_width=2*head_size.",
            )
        if quant_block != 64:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires quant_block=64.",
            )
        if rope_head_dim != 64:
            raise ValueError(
                "CuTe DSL C4 fused sparse-attn currently requires rope_head_dim=64.",
            )
        num_positions = cute.sym_int()
        num_slots = cute.sym_int()
        num_req_indices = cute.sym_int()
        num_kv_slots = cute.sym_int()
        num_state_blocks = cute.sym_int()
        num_kv_blocks = cute.sym_int()
        state_cache_block_size = cute.sym_int()
        block_table_width = cute.sym_int()
        max_pos = cute.sym_int()
        state_cache_width = state_width * 2
        state_cache = cute.runtime.make_fake_tensor(
            Float32,
            (num_state_blocks, state_cache_block_size, state_cache_width),
            stride=(
                cute.sym_int64(divisibility=16),
                cute.sym_int64(divisibility=16),
                1,
            ),
            assumed_align=16,
        )
        token_to_req_indices = make_fake_tensor(
            Int32,
            (num_req_indices,),
            divisibility=4,
        )
        positions = make_fake_tensor(Int64, (num_positions,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_slots,), divisibility=8)
        block_table = make_fake_tensor(
            Int32,
            (cute.sym_int(), block_table_width),
            divisibility=1,
        )
        rms_norm_weight = make_fake_tensor(
            norm_weight_dtype,
            (head_size,),
            divisibility=4,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            Float32,
            (max_pos, rope_head_dim),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=4,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (num_kv_blocks, cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(divisibility=16), cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        kv_slot_mapping = make_fake_tensor(Int64, (num_kv_slots,), divisibility=8)
        fp8_scale = make_fake_tensor(Float32, (1,), divisibility=1)
        host_entrypoint = self.kernel(compile_key)

        compiled = compile_cutedsl(
            host_entrypoint,
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            Int64(0),
            rms_norm_weight,
            Float32(0.0),
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            Int64(0),
            fp8_scale,
        )
        self._compiled_cache[compile_key] = compiled

    def __call__(
        self,
        *,
        state_cache: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        compress_ratio: int,
        store_full_fp8: bool,
        fp8_scale: torch.Tensor | None,
        head_dim: int,
        rope_head_dim: int,
    ) -> Any:
        compile_key = self.dispatch(
            compress_ratio=compress_ratio,
            store_full_fp8=store_full_fp8,
            norm_weight_dtype=torch_to_cute_dtype(rms_norm_weight.dtype),
            head_size=head_dim,
            rope_head_dim=rope_head_dim,
        )
        runtime_context = {
            "kv_cache_shape": tuple(kv_cache.shape),
            "kv_cache_stride": tuple(kv_cache.stride()),
            "compress_ratio": compress_ratio,
            "store_full_fp8": store_full_fp8,
        }
        kernel = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        if fp8_scale is None:
            fp8_scale = torch.ones(1, dtype=torch.float32, device=kv_cache.device)
        return kernel(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache.view(torch.uint8),
            kv_slot_mapping,
            kv_cache.shape[1],
            fp8_scale,
        )


class SparseAttnCompressC128Block8Kernel(
    VllmJitKernel["SparseAttnCompressC128Block8Kernel.CompileKey"]
):
    head_tile = 64
    rows_per_warp = 16
    elems_per_lane = 2
    lanes_per_row = head_tile // elems_per_lane
    num_warps = 8
    stats_lane_stride = lanes_per_row + 1
    final_reduce_steps = 3
    final_reduce_initial_offset = 4
    tb_size = num_warps * 32
    compress_ratio = 128
    state_block_size = 8
    rcp_ln2 = 1.4426950408889634

    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        state_width: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        head_tile = SparseAttnCompressC128Block8Kernel.head_tile
        rows_per_warp = SparseAttnCompressC128Block8Kernel.rows_per_warp
        elems_per_lane = SparseAttnCompressC128Block8Kernel.elems_per_lane
        lanes_per_row = SparseAttnCompressC128Block8Kernel.lanes_per_row
        num_warps = SparseAttnCompressC128Block8Kernel.num_warps
        stats_lane_stride = SparseAttnCompressC128Block8Kernel.stats_lane_stride
        final_reduce_steps = SparseAttnCompressC128Block8Kernel.final_reduce_steps
        final_reduce_initial_offset = (
            SparseAttnCompressC128Block8Kernel.final_reduce_initial_offset
        )
        tb_size = SparseAttnCompressC128Block8Kernel.tb_size
        compress_ratio = SparseAttnCompressC128Block8Kernel.compress_ratio
        state_block_size = SparseAttnCompressC128Block8Kernel.state_block_size
        rcp_ln2 = SparseAttnCompressC128Block8Kernel.rcp_ln2
        head_dim = compile_key.head_size
        num_splits = head_dim // head_tile
        state_width = compile_key.state_width

        @cute.kernel
        def device_kernel(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            compressed_kv: cute.Tensor,
        ):
            block_id, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            warp_id = cute.arch.make_warp_uniform(tid // 32)
            lane_id = tid % 32
            col_group = lane_id % lanes_per_row

            token_idx = block_id // num_splits
            split_idx = block_id - token_idx * num_splits
            col_base = split_idx * head_tile + col_group * elems_per_lane

            position = Int64(0)
            req_idx = Int32(0)
            slot_id = Int64(-1)
            has_position = token_idx < positions.shape[0]
            has_req_idx = token_idx < token_to_req_indices.shape[0]
            if lane_id == 0:
                slot_id = slot_mapping[token_idx]
            if lane_id == 0 and has_position:
                position = positions[token_idx]
            if lane_id == 0 and has_req_idx:
                req_idx = token_to_req_indices[token_idx]
            slot_id = cute.arch.shuffle_sync(slot_id, offset=0)
            position = cute.arch.shuffle_sync(position, offset=0)
            req_idx = cute.arch.shuffle_sync(req_idx, offset=0)
            boundary = has_position and (
                (position + Int64(1)) % Int64(compress_ratio) == Int64(0)
            )
            start = position - Int64(compress_ratio - 1)
            active = slot_id >= Int64(0) and has_req_idx and boundary

            if active:
                smem = cutlass.utils.SmemAllocator()
                s_max = smem.allocate_tensor(
                    Float32,
                    cute.make_layout(
                        (
                            num_warps,
                            lanes_per_row,
                            elems_per_lane,
                        ),
                        stride=(
                            stats_lane_stride * elems_per_lane,
                            elems_per_lane,
                            1,
                        ),
                    ),
                    byte_alignment=4,
                )
                s_sum = smem.allocate_tensor(
                    Float32,
                    cute.make_layout(
                        (
                            num_warps,
                            lanes_per_row,
                            elems_per_lane,
                        ),
                        stride=(
                            stats_lane_stride * elems_per_lane,
                            elems_per_lane,
                            1,
                        ),
                    ),
                    byte_alignment=4,
                )
                s_product = smem.allocate_tensor(
                    Float32,
                    cute.make_layout(
                        (
                            num_warps,
                            lanes_per_row,
                            elems_per_lane,
                        ),
                        stride=(
                            stats_lane_stride * elems_per_lane,
                            elems_per_lane,
                            1,
                        ),
                    ),
                    byte_alignment=4,
                )

                row_layout = cute.make_layout(
                    (rows_per_warp, elems_per_lane),
                    stride=(elems_per_lane, 1),
                )
                kv_vals = cute.make_rmem_tensor(row_layout, Float32)
                score_vals = cute.make_rmem_tensor(row_layout, Float32)
                local_max = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_sum = cute.make_rmem_tensor((elems_per_lane,), Float32)
                local_product = cute.make_rmem_tensor((elems_per_lane,), Float32)

                for e in cutlass.range_constexpr(elems_per_lane):
                    local_max[e] = -Float32.inf
                    local_sum[e] = Float32(0.0)
                    local_product[e] = Float32(0.0)

                first_block_index = start // Int64(state_block_size)
                warp_block_index = first_block_index + (warp_id * 2).to(Int64)
                block0_i32 = Int32(0)
                block1_i32 = Int32(0)
                if lane_id == 0:
                    block0_i32 = block_table[req_idx, warp_block_index]
                    block1_i32 = block_table[req_idx, warp_block_index + Int64(1)]
                block0_i32 = cute.arch.shuffle_sync(block0_i32, offset=0)
                block1_i32 = cute.arch.shuffle_sync(block1_i32, offset=0)

                cp_f32x2 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=64
                )
                final_mask_and_clamp = const_expr(
                    (cute.arch.WARP_SIZE - num_warps) << 8 | (cute.arch.WARP_SIZE - 1)
                )
                col_tile = col_base.to(Int64) // Int64(elems_per_lane)
                score_col_tile = col_tile + Int64(state_width // elems_per_lane)

                for i in cutlass.range_constexpr(rows_per_warp):
                    block_number_i32 = block0_i32
                    block_offset = Int64(i)
                    if const_expr(i >= state_block_size):
                        block_number_i32 = block1_i32
                        block_offset = Int64(i - state_block_size)
                    row_tensor = state_cache[
                        block_number_i32.to(Int64), block_offset, None
                    ]
                    kv_src = cute.local_tile(
                        row_tensor,
                        tiler=(elems_per_lane,),
                        coord=(col_tile,),
                    )
                    score_src = cute.local_tile(
                        row_tensor,
                        tiler=(elems_per_lane,),
                        coord=(score_col_tile,),
                    )
                    cute.copy(cp_f32x2, kv_src, kv_vals[i, None])
                    cute.copy(cp_f32x2, score_src, score_vals[i, None])

                    for e in cutlass.range_constexpr(elems_per_lane):
                        local_max[e] = cute.arch.fmax(local_max[e], score_vals[i, e])

                for e in cutlass.range_constexpr(elems_per_lane):
                    for i in cutlass.range_constexpr(rows_per_warp):
                        exp_score = cute.math.exp2(
                            (score_vals[i, e] - local_max[e]) * Float32(rcp_ln2),
                            fastmath=True,
                        )
                        local_sum[e] += exp_score
                        local_product[e] += kv_vals[i, e] * exp_score

                for e in cutlass.range_constexpr(elems_per_lane):
                    s_max[warp_id, col_group, e] = local_max[e]
                    s_sum[warp_id, col_group, e] = local_sum[e]
                    s_product[warp_id, col_group, e] = local_product[e]
                cute.arch.sync_threads()

                out_group = tid // num_warps
                final_lane = tid % num_warps
                final_groups_per_pass = const_expr(tb_size // num_warps)
                for pass_idx in cutlass.range_constexpr(
                    head_tile // final_groups_per_pass
                ):
                    out_idx = pass_idx * final_groups_per_pass + out_group
                    out_lane = out_idx // elems_per_lane
                    out_elem = out_idx % elems_per_lane

                    local_warp_max = s_max[final_lane, out_lane, out_elem]
                    global_max = local_warp_max
                    for step in cutlass.range_constexpr(final_reduce_steps):
                        offset = const_expr(final_reduce_initial_offset >> step)
                        global_max = cute.arch.fmax(
                            global_max,
                            cute.arch.shuffle_sync_bfly(
                                global_max,
                                offset=offset,
                                mask_and_clamp=final_mask_and_clamp,
                            ),
                        )

                    scale = cute.math.exp2(
                        (local_warp_max - global_max) * Float32(rcp_ln2),
                        fastmath=True,
                    )
                    global_sum = s_sum[final_lane, out_lane, out_elem] * scale
                    global_product = s_product[final_lane, out_lane, out_elem] * scale
                    for step in cutlass.range_constexpr(final_reduce_steps):
                        offset = const_expr(final_reduce_initial_offset >> step)
                        global_sum += cute.arch.shuffle_sync_bfly(
                            global_sum,
                            offset=offset,
                            mask_and_clamp=final_mask_and_clamp,
                        )
                        global_product += cute.arch.shuffle_sync_bfly(
                            global_product,
                            offset=offset,
                            mask_and_clamp=final_mask_and_clamp,
                        )

                    if final_lane == 0:
                        compressed_kv.iterator[
                            token_idx.to(Int64) * compressed_kv.stride[0]
                            + (split_idx * head_tile + out_idx).to(Int64)
                        ] = global_product / global_sum

        @cute.jit
        def host_entrypoint(
            state_cache: cute.Tensor,
            token_to_req_indices: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            block_table: cute.Tensor,
            compressed_kv: cute.Tensor,
            stream: CUstream,
        ):
            grid = (slot_mapping.shape[0] * num_splits, 1, 1)
            device_kernel(
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                compressed_kv,
            ).launch(grid=grid, block=(tb_size, 1, 1), stream=stream)

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self, *, head_size: int, state_width: int
    ) -> CompileKey:
        return self.CompileKey(head_size=head_size, state_width=state_width)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        return self._trace_dispatch(self.dispatch)(
            head_size=hf_config.head_dim,
            state_width=hf_config.head_dim,
            enabled=128 in hf_config.compress_ratios,
            _when=lambda *, enabled: enabled,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        head_size = compile_key.head_size
        state_width = compile_key.state_width
        if head_size % SparseAttnCompressC128Block8Kernel.head_tile != 0:
            raise ValueError("head_size must be divisible by the 64-wide head tile.")
        num_positions = cute.sym_int()
        num_slots = cute.sym_int()
        num_req_indices = cute.sym_int()
        num_blocks = cute.sym_int()
        state_cache_block_size = cute.sym_int()
        block_table_width = cute.sym_int()
        state_cache_width = state_width * 2
        state_cache = cute.runtime.make_fake_tensor(
            Float32,
            (num_blocks, state_cache_block_size, state_cache_width),
            stride=(
                cute.sym_int64(divisibility=16),
                cute.sym_int64(divisibility=16),
                1,
            ),
            assumed_align=16,
        )
        token_to_req_indices = make_fake_tensor(
            Int32,
            (num_req_indices,),
            divisibility=4,
        )
        positions = make_fake_tensor(Int64, (num_positions,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_slots,), divisibility=8)
        block_table = make_fake_tensor(
            Int32,
            (cute.sym_int(), block_table_width),
            divisibility=1,
        )
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_slots, head_size),
            stride=(head_size, 1),
            assumed_align=4,
        )
        host_entrypoint = self.kernel(compile_key)

        compiled = compile_cutedsl(
            host_entrypoint,
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            compressed_kv,
        )
        self._compiled_cache[compile_key] = compiled

    def __call__(
        self,
        *,
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        head_dim: int,
        compressed_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        compile_key = self.dispatch(head_size=head_dim, state_width=head_dim)
        runtime_context = {
            "state_cache_shape": tuple(state_cache.shape),
            "head_dim": head_dim,
        }
        kernel = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        if compressed_kv is None:
            compressed_kv = torch.empty(
                (num_actual, head_dim),
                dtype=torch.float32,
                device=state_cache.device,
            )
        kernel(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            compressed_kv,
        )
        return compressed_kv


class SparseAttnNormRopeStoreKernel(
    VllmJitKernel["SparseAttnNormRopeStoreKernel.CompileKey"]
):
    min_scale = 1.0e-4

    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        rope_head_dim: int
        fp8_max: float
        quant_block: int
        token_stride: int
        scale_dim: int
        kv_block_stride: int
        compress_ratio: int
        norm_weight_dtype: type[cutlass.Numeric]
        kv_cache_block_size: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        min_scale = SparseAttnNormRopeStoreKernel.min_scale
        head_dim = compile_key.head_size
        rope_dim = compile_key.rope_head_dim
        nope_dim = head_dim - rope_dim
        fp8_max = compile_key.fp8_max
        quant_block = compile_key.quant_block
        token_stride = compile_key.token_stride
        scale_dim = compile_key.scale_dim
        num_warps = head_dim // quant_block
        nope_blocks = nope_dim // quant_block
        tb_size = head_dim // 2
        compress_ratio = compile_key.compress_ratio
        static_kv_cache_block_size = compile_key.kv_cache_block_size

        @cute.kernel
        def device_kernel(
            compressed_kv: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
        ):
            token_idx, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            warp_id = cute.arch.make_warp_uniform(tid // 32)
            lane_id = tid % 32
            elem0 = tid * 2

            position = Int64(0)
            kv_slot_idx = Int64(-1)
            has_position = token_idx < positions.shape[0]
            slot_id = Int64(-1)
            if lane_id == 0:
                slot_id = slot_mapping[token_idx]
            if lane_id == 0 and has_position:
                position = positions[token_idx]
            has_kv_slot_idx = token_idx < kv_slot_mapping.shape[0]
            if lane_id == 0 and has_kv_slot_idx:
                kv_slot_idx = kv_slot_mapping[token_idx]
            slot_id = cute.arch.shuffle_sync(slot_id, offset=0)
            position = cute.arch.shuffle_sync(position, offset=0)
            kv_slot_idx = cute.arch.shuffle_sync(kv_slot_idx, offset=0)
            boundary = has_position and (
                (position + Int64(1)) % Int64(compress_ratio) == Int64(0)
            )
            active = slot_id >= Int64(0) and boundary and kv_slot_idx >= Int64(0)

            if active:
                k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
                k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
                static_block_size = Int64(static_kv_cache_block_size)
                page = kv_slot_idx // static_block_size
                kv_offset = kv_slot_idx - page * static_block_size
                scale_row_offset = static_block_size * Int64(token_stride)
                value_base = page * k_cache.stride[0] + kv_offset * Int64(token_stride)
                scale_base = (
                    page * k_cache.stride[0]
                    + scale_row_offset
                    + kv_offset * Int64(scale_dim)
                )
                weight0 = rms_norm_weight[elem0].to(Float32)
                weight1 = rms_norm_weight[elem0 + 1].to(Float32)

                base = token_idx.to(Int64) * compressed_kv.stride[0] + elem0.to(Int64)
                x0 = compressed_kv.iterator[base]
                x1 = compressed_kv.iterator[base + Int64(1)]

                local_sumsq = x0 * x0 + x1 * x1
                warp_sum = local_sumsq
                for step in cutlass.range_constexpr(5):
                    offset = const_expr(16 >> step)
                    warp_sum += cute.arch.shuffle_sync_bfly(warp_sum, offset)

                smem = cutlass.utils.SmemAllocator()
                partial_sums = smem.allocate_tensor(
                    Float32, cute.make_layout((num_warps,)), byte_alignment=4
                )

                if lane_id == 0:
                    partial_sums[warp_id] = warp_sum
                cute.arch.sync_threads()

                total = partial_sums[lane_id % num_warps]
                sum_mask_and_clamp = const_expr(
                    (cute.arch.WARP_SIZE - num_warps) << 8 | (cute.arch.WARP_SIZE - 1)
                )
                for step in cutlass.range_constexpr(3):
                    offset = const_expr(4 >> step)
                    total += cute.arch.shuffle_sync_bfly(
                        total,
                        offset,
                        mask_and_clamp=sum_mask_and_clamp,
                    )

                rrms = cute.math.rsqrt(
                    total / Float32(head_dim) + rms_norm_eps, fastmath=True
                )
                x0 = x0 * rrms * weight0
                x1 = x1 * rrms * weight1

                if warp_id == nope_blocks:
                    pair_idx = lane_id
                    compressed_pos = position - Int64(compress_ratio - 1)
                    cs_base = compressed_pos * cos_sin_cache.stride[0] + pair_idx.to(
                        Int64
                    )
                    cos_v = cos_sin_cache.iterator[cs_base]
                    sin_v = cos_sin_cache.iterator[cs_base + Int64(rope_dim // 2)]
                    real = x0 * cos_v - x1 * sin_v
                    imag = x0 * sin_v + x1 * cos_v
                    packed = _fp32x2_to_bf16x2(real, imag)
                    out_base = value_base + Int64(nope_dim) + (lane_id * 4).to(Int64)
                    k_cache_u32.iterator[out_base // Int64(4)] = packed
                else:
                    q_packed = _fp32x2_to_bf16x2(x0, x1)
                    q0, q1 = _bf16x2_to_fp32(q_packed)
                    abs0 = cute.math.absf(q0)
                    abs1 = cute.math.absf(q1)
                    local_absmax = cute.arch.fmax(abs0, abs1)
                    absmax = local_absmax
                    for step in cutlass.range_constexpr(5):
                        offset = const_expr(16 >> step)
                        absmax = cute.arch.fmax(
                            absmax, cute.arch.shuffle_sync_bfly(absmax, offset)
                        )
                    scale_raw = cute.arch.fmax(
                        Float32(min_scale),
                        absmax / Float32(fp8_max),
                    )
                    bits = _recast_val(scale_raw, Uint32)
                    ue8m0 = ((bits + Uint32(0x7FFFFF)) >> Uint32(23)) & Uint32(0xFF)
                    inv_scale = _recast_val(
                        (Uint32(254) - ue8m0) << Uint32(23), Float32
                    )
                    y0 = cutlass.min(
                        cute.arch.fmax(q0 * inv_scale, Float32(-fp8_max)),
                        Float32(fp8_max),
                    )
                    y1 = cutlass.min(
                        cute.arch.fmax(q1 * inv_scale, Float32(-fp8_max)),
                        Float32(fp8_max),
                    )
                    packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                    out_base = value_base + (warp_id * quant_block + lane_id * 2).to(
                        Int64
                    )
                    k_cache_u16.iterator[out_base // Int64(2)] = packed_fp8
                    if lane_id == 0:
                        k_cache.iterator[scale_base + warp_id.to(Int64)] = ue8m0.to(
                            Uint8
                        )
                        if warp_id == 0:
                            k_cache.iterator[scale_base + Int64(nope_blocks)] = Uint8(0)

        @cute.jit
        def host_entrypoint(
            compressed_kv: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            stream: CUstream,
        ):
            grid = (slot_mapping.shape[0], 1, 1)
            device_kernel(
                compressed_kv,
                positions,
                slot_mapping,
                rms_norm_weight,
                rms_norm_eps,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
            ).launch(grid=grid, block=(tb_size, 1, 1), stream=stream)

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        cache_block_size: int,
        cache_alignment: int,
        norm_weight_dtype: type[cutlass.Numeric],
        head_size: int,
        rope_head_dim: int,
    ) -> CompileKey:
        raw_kv_cache_block_size = cache_block_size // compress_ratio
        kv_cache_block_size = (
            raw_kv_cache_block_size if raw_kv_cache_block_size >= 1 else 1
        )
        token_stride = 576
        scale_dim = 8
        return self.CompileKey(
            head_size=head_size,
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=64,
            token_stride=token_stride,
            scale_dim=scale_dim,
            kv_block_stride=round_up(
                kv_cache_block_size * (token_stride + scale_dim),
                cache_alignment,
            ),
            compress_ratio=compress_ratio,
            norm_weight_dtype=norm_weight_dtype,
            kv_cache_block_size=kv_cache_block_size,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=128,
            cache_block_size=cache_config.block_size,
            cache_alignment=576,
            norm_weight_dtype=torch_to_cute_dtype(vllm_config.model_config.dtype),
            head_size=hf_config.head_dim,
            rope_head_dim=hf_config.qk_rope_head_dim,
            enabled=(
                cache_config.block_size > 0
                and 128 in hf_config.compress_ratios
                and cache_config.cache_dtype == "fp8_ds_mla"
            ),
            _when=lambda *, enabled: enabled,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        head_size = compile_key.head_size
        rope_head_dim = compile_key.rope_head_dim
        quant_block = compile_key.quant_block
        token_stride = compile_key.token_stride
        scale_dim = compile_key.scale_dim
        kv_block_stride = compile_key.kv_block_stride
        norm_weight_dtype = compile_key.norm_weight_dtype
        static_kv_cache_block_size = compile_key.kv_cache_block_size
        if quant_block != 64:
            raise ValueError(
                "CuTe DSL sparse-attn store currently requires quant_block=64.",
            )
        if rope_head_dim != 64:
            raise ValueError(
                "CuTe DSL sparse-attn store currently requires rope_head_dim=64.",
            )
        if head_size % quant_block != 0:
            raise ValueError("head_size must be divisible by quant_block.")
        if token_stride < head_size + rope_head_dim:
            raise ValueError("token_stride is too small for the packed FP8/BF16 row.")
        expected_scale_dim = (head_size - rope_head_dim) // quant_block + 1
        if scale_dim < expected_scale_dim:
            raise ValueError("scale_dim is too small for the UE8M0 scale row.")
        if static_kv_cache_block_size <= 0:
            raise ValueError(
                "CuTe DSL sparse-attn store requires a positive static "
                "kv_cache_block_size.",
            )
        num_positions = cute.sym_int()
        num_slots = cute.sym_int()
        num_kv_slots = cute.sym_int()
        max_pos = cute.sym_int()
        num_blocks = cute.sym_int()
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_slots, head_size),
            stride=(head_size, 1),
            assumed_align=4,
        )
        positions = make_fake_tensor(Int64, (num_positions,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_slots,), divisibility=8)
        rms_norm_weight = make_fake_tensor(
            norm_weight_dtype,
            (head_size,),
            divisibility=4,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            Float32,
            (max_pos, rope_head_dim),
            stride=(rope_head_dim, 1),
            assumed_align=4,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (num_blocks, cute.sym_int(), cute.sym_int()),
            stride=(kv_block_stride, cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        kv_slot_mapping = make_fake_tensor(Int64, (num_kv_slots,), divisibility=8)
        host_entrypoint = self.kernel(compile_key)

        compiled = compile_cutedsl(
            host_entrypoint,
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            Float32(0.0),
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
        )
        self._compiled_cache[compile_key] = compiled

    def __call__(
        self,
        *,
        compressed_kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
    ) -> Any:
        compile_key = self.dispatch(
            compress_ratio=compress_ratio,
            cache_block_size=kv_cache.shape[1] * compress_ratio,
            cache_alignment=576,
            norm_weight_dtype=torch_to_cute_dtype(rms_norm_weight.dtype),
            head_size=head_dim,
            rope_head_dim=rope_head_dim,
        )
        runtime_context = {
            "kv_cache_shape": tuple(kv_cache.shape),
            "kv_cache_stride": tuple(kv_cache.stride()),
            "compress_ratio": compress_ratio,
        }
        kernel = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        return kernel(
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache,
            kv_slot_mapping,
        )


class SparseAttnNormRopeStoreFullKernel(
    VllmJitKernel["SparseAttnNormRopeStoreFullKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        head_size: int
        rope_head_dim: int
        fp8_max: float
        quant_block: int
        compress_ratio: int
        store_full_fp8: bool
        norm_weight_dtype: type[cutlass.Numeric]

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        head_dim = compile_key.head_size
        rope_dim = compile_key.rope_head_dim
        nope_dim = head_dim - rope_dim
        fp8_max = compile_key.fp8_max
        quant_block = compile_key.quant_block
        num_warps = head_dim // quant_block
        nope_blocks = nope_dim // quant_block
        tb_size = head_dim // 2
        compress_ratio = compile_key.compress_ratio
        store_full_fp8 = compile_key.store_full_fp8

        @cute.kernel
        def device_kernel(
            compressed_kv: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
            fp8_scale: cute.Tensor,
        ):
            token_idx, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            warp_id = cute.arch.make_warp_uniform(tid // 32)
            lane_id = tid % 32
            elem0 = tid * 2

            slot_id = slot_mapping[token_idx]
            has_position = token_idx < positions.shape[0]
            position = Int64(0)
            if has_position:
                position = positions[token_idx]
            boundary = has_position and (
                (position + Int64(1)) % Int64(compress_ratio) == Int64(0)
            )
            has_kv_slot_idx = token_idx < kv_slot_mapping.shape[0]
            kv_slot_idx = Int64(-1)
            if has_kv_slot_idx:
                kv_slot_idx = kv_slot_mapping[token_idx]
            active = slot_id >= Int64(0) and boundary and kv_slot_idx >= Int64(0)

            if active:
                base = token_idx.to(Int64) * compressed_kv.stride[0] + elem0.to(Int64)
                x0 = compressed_kv.iterator[base]
                x1 = compressed_kv.iterator[base + Int64(1)]

                local_sumsq = x0 * x0 + x1 * x1
                warp_sum = local_sumsq
                for step in cutlass.range_constexpr(5):
                    offset = const_expr(16 >> step)
                    warp_sum += cute.arch.shuffle_sync_bfly(warp_sum, offset)

                smem = cutlass.utils.SmemAllocator()
                partial_sums = smem.allocate_tensor(
                    Float32, cute.make_layout((num_warps,)), byte_alignment=4
                )
                rrms_shared = smem.allocate_tensor(
                    Float32, cute.make_layout((1,)), byte_alignment=4
                )

                if lane_id == 0:
                    partial_sums[warp_id] = warp_sum
                cute.arch.sync_threads()
                if tid == 0:
                    total = Float32(0.0)
                    for i in cutlass.range_constexpr(num_warps):
                        total += partial_sums[i]
                    rrms_shared[0] = cute.math.rsqrt(
                        total / Float32(head_dim) + rms_norm_eps, fastmath=True
                    )
                cute.arch.sync_threads()

                rrms = rrms_shared[0]
                x0 = x0 * rrms * rms_norm_weight[elem0].to(Float32)
                x1 = x1 * rrms * rms_norm_weight[elem0 + 1].to(Float32)

                page = kv_slot_idx // kv_cache_block_size
                kv_offset = kv_slot_idx - page * kv_cache_block_size
                value_base = page * k_cache.stride[0] + kv_offset * k_cache.stride[1]

                if const_expr(store_full_fp8):
                    k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
                    inv_fp8 = Float32(1.0) / fp8_scale[0]
                    fp8_v0 = x0
                    fp8_v1 = x1
                    if warp_id == nope_blocks:
                        compressed_pos = (position // Int64(compress_ratio)) * Int64(
                            compress_ratio
                        )
                        pair_idx = lane_id
                        cs_base = compressed_pos * cos_sin_cache.stride[
                            0
                        ] + pair_idx.to(Int64)
                        cos_v = cos_sin_cache.iterator[cs_base]
                        sin_v = cos_sin_cache.iterator[cs_base + Int64(rope_dim // 2)]
                        fp8_v0 = x0 * cos_v - x1 * sin_v
                        fp8_v1 = x0 * sin_v + x1 * cos_v
                    fp8_packed_bf16 = _fp32x2_to_bf16x2(fp8_v0, fp8_v1)
                    b0, b1 = _bf16x2_to_fp32(fp8_packed_bf16)
                    y0 = cutlass.min(
                        cutlass.max(b0 * inv_fp8, Float32(-fp8_max)),
                        Float32(fp8_max),
                    )
                    y1 = cutlass.min(
                        cutlass.max(b1 * inv_fp8, Float32(-fp8_max)),
                        Float32(fp8_max),
                    )
                    packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                    out_base = value_base + elem0.to(Int64)
                    k_cache_u16.iterator[out_base // Int64(2)] = packed_fp8
                else:
                    k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
                    bf16_v0 = x0
                    bf16_v1 = x1
                    if warp_id == nope_blocks:
                        compressed_pos = (position // Int64(compress_ratio)) * Int64(
                            compress_ratio
                        )
                        pair_idx = lane_id
                        cs_base = compressed_pos * cos_sin_cache.stride[
                            0
                        ] + pair_idx.to(Int64)
                        cos_v = cos_sin_cache.iterator[cs_base]
                        sin_v = cos_sin_cache.iterator[cs_base + Int64(rope_dim // 2)]
                        bf16_v0 = x0 * cos_v - x1 * sin_v
                        bf16_v1 = x0 * sin_v + x1 * cos_v
                    bf16_packed = _fp32x2_to_bf16x2(bf16_v0, bf16_v1)
                    out_base = value_base + (elem0 * 2).to(Int64)
                    k_cache_u32.iterator[out_base // Int64(4)] = bf16_packed

        @cute.jit
        def host_entrypoint(
            compressed_kv: cute.Tensor,
            positions: cute.Tensor,
            slot_mapping: cute.Tensor,
            rms_norm_weight: cute.Tensor,
            rms_norm_eps: Float32,
            cos_sin_cache: cute.Tensor,
            k_cache: cute.Tensor,
            kv_slot_mapping: cute.Tensor,
            kv_cache_block_size: Int64,
            fp8_scale: cute.Tensor,
            stream: CUstream,
        ):
            grid = (slot_mapping.shape[0], 1, 1)
            device_kernel(
                compressed_kv,
                positions,
                slot_mapping,
                rms_norm_weight,
                rms_norm_eps,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
                kv_cache_block_size,
                fp8_scale,
            ).launch(grid=grid, block=(tb_size, 1, 1), stream=stream)

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        store_full_fp8: bool,
        norm_weight_dtype: type[cutlass.Numeric],
        head_size: int,
        rope_head_dim: int,
    ) -> CompileKey:
        return self.CompileKey(
            head_size=head_size,
            rope_head_dim=rope_head_dim,
            fp8_max=448.0,
            quant_block=64,
            compress_ratio=compress_ratio,
            store_full_fp8=store_full_fp8,
            norm_weight_dtype=norm_weight_dtype,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        return self._trace_dispatch(self.dispatch)(
            compress_ratio=128,
            store_full_fp8=cache_config.cache_dtype in ("fp8", "fp8_e4m3"),
            norm_weight_dtype=torch_to_cute_dtype(vllm_config.model_config.dtype),
            head_size=hf_config.head_dim,
            rope_head_dim=hf_config.qk_rope_head_dim,
            enabled=(
                cache_config.block_size > 0
                and 128 in hf_config.compress_ratios
                and cache_config.cache_dtype != "fp8_ds_mla"
            ),
            _when=lambda *, enabled: enabled,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if self._compiled_cache_contains(compile_key):
            return
        head_size = compile_key.head_size
        rope_head_dim = compile_key.rope_head_dim
        quant_block = compile_key.quant_block
        norm_weight_dtype = compile_key.norm_weight_dtype
        if quant_block != 64:
            raise ValueError(
                "CuTe DSL sparse-attn store currently requires quant_block=64.",
            )
        if rope_head_dim != 64:
            raise ValueError(
                "CuTe DSL sparse-attn store currently requires rope_head_dim=64.",
            )
        if head_size % quant_block != 0:
            raise ValueError("head_size must be divisible by quant_block.")
        num_positions = cute.sym_int()
        num_slots = cute.sym_int()
        num_kv_slots = cute.sym_int()
        max_pos = cute.sym_int()
        num_blocks = cute.sym_int()
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_slots, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=4,
        )
        positions = make_fake_tensor(Int64, (num_positions,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_slots,), divisibility=8)
        rms_norm_weight = make_fake_tensor(
            norm_weight_dtype,
            (head_size,),
            divisibility=4,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            Float32,
            (max_pos, rope_head_dim),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=4,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (num_blocks, cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(divisibility=16), cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        kv_slot_mapping = make_fake_tensor(Int64, (num_kv_slots,), divisibility=8)
        fp8_scale = make_fake_tensor(Float32, (1,), divisibility=1)
        host_entrypoint = self.kernel(compile_key)

        compiled = compile_cutedsl(
            host_entrypoint,
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            Float32(0.0),
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            Int64(0),
            fp8_scale,
        )
        self._compiled_cache[compile_key] = compiled

    def __call__(
        self,
        *,
        compressed_kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        compress_ratio: int,
        store_full_fp8: bool,
        fp8_scale: torch.Tensor | None,
        head_dim: int,
        rope_head_dim: int,
    ) -> Any:
        compile_key = self.dispatch(
            compress_ratio=compress_ratio,
            store_full_fp8=store_full_fp8,
            norm_weight_dtype=torch_to_cute_dtype(rms_norm_weight.dtype),
            head_size=head_dim,
            rope_head_dim=rope_head_dim,
        )
        runtime_context = {
            "kv_cache_shape": tuple(kv_cache.shape),
            "kv_cache_stride": tuple(kv_cache.stride()),
            "compress_ratio": compress_ratio,
            "store_full_fp8": store_full_fp8,
        }
        kernel = self._get_or_compile(
            compile_key,
            runtime_context=runtime_context,
        )
        if fp8_scale is None:
            fp8_scale = torch.ones(1, dtype=torch.float32, device=kv_cache.device)
        return kernel(
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            kv_cache.view(torch.uint8),
            kv_slot_mapping,
            kv_cache.shape[1],
            fp8_scale,
        )


def split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compressed_kv: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 512,
    state_width: int = 512,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
    compress_ratio: int = 128,
    overlap: bool = False,
    store_full_kv: bool = False,
    store_full_fp8: bool = False,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    del fp8_max, quant_block, token_stride, scale_dim, kv_block_stride
    if not (
        head_size == 512
        and state_width == head_size
        and compress_ratio == 128
        and not overlap
        and block_size == 8
    ):
        raise ValueError(
            "CuTe DSL split sparse-attn wrapper only supports the real "
            "DeepSeek V4 C128 layout: head_size=512, state_width=512, "
            "compress_ratio=128, overlap=False, block_size=8."
        )
    if k_cache.ndim != 3:
        raise ValueError(
            "CuTe DSL sparse-attn store expects the real DeepSeek V4 "
            f"3D k_cache layout [num_blocks, block_size, 584], got ndim={k_cache.ndim}."
        )
    if not store_full_kv and kv_cache_block_size != k_cache.shape[1]:
        raise ValueError(
            "CuTe DSL split sparse-attn wrapper expected kv_cache_block_size "
            f"to match k_cache.shape[1], got {kv_cache_block_size} and "
            f"{k_cache.shape[1]}."
        )
    if positions.numel() == 0:
        return
    if store_full_fp8 and not store_full_kv:
        raise ValueError("store_full_fp8 requires store_full_kv.")

    _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL(
        state_cache=state_cache,
        num_actual=compressed_kv.shape[0],
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        head_dim=head_size,
        compressed_kv=compressed_kv,
    )
    if store_full_kv:
        _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL(
            compressed_kv=compressed_kv,
            positions=positions,
            slot_mapping=slot_mapping,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache=k_cache,
            kv_slot_mapping=kv_slot_mapping,
            compress_ratio=compress_ratio,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
            head_dim=head_size,
            rope_head_dim=rope_head_dim,
        )
        return
    _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL(
        compressed_kv=compressed_kv,
        positions=positions,
        slot_mapping=slot_mapping,
        rms_norm_weight=rms_norm_weight,
        rms_norm_eps=rms_norm_eps,
        cos_sin_cache=cos_sin_cache,
        kv_cache=k_cache,
        kv_slot_mapping=kv_slot_mapping,
        compress_ratio=compress_ratio,
        head_dim=head_size,
        rope_head_dim=rope_head_dim,
    )


class SparseAttnCompressorCuteDSL:
    def __init__(
        self,
        *,
        c4_store: SparseAttnCompressNormRopeStoreC4Kernel,
        c4_store_full: SparseAttnCompressNormRopeStoreFullC4Kernel,
        c128_compress: SparseAttnCompressC128Block8Kernel,
        c128_store: SparseAttnNormRopeStoreKernel,
        c128_store_full: SparseAttnNormRopeStoreFullKernel,
    ) -> None:
        self.c4_store = c4_store
        self.c4_store_full = c4_store_full
        self.c128_compress = c128_compress
        self.c128_store = c128_store
        self.c128_store_full = c128_store_full

    def __call__(
        self,
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        state_width: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        pdl_kwargs: dict,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        overlap: bool,
        use_fp4_cache: bool,
        rms_norm_weight: torch.Tensor,
        rms_norm_eps: float,
        quant_block: int,
        token_stride: int,
        scale_dim: int,
        store_full_kv: bool = False,
        store_full_fp8: bool = False,
        fp8_scale: torch.Tensor | None = None,
    ) -> None:
        del pdl_kwargs, state_width, overlap, use_fp4_cache
        del quant_block, token_stride, scale_dim
        if positions.numel() == 0:
            return
        if kv_cache.ndim != 3:
            raise ValueError(
                "CuTe DSL sparse-attn compressor expects the real DeepSeek V4 "
                f"3D kv_cache layout, got ndim={kv_cache.ndim}.",
            )
        if store_full_fp8 and (not store_full_kv):
            raise ValueError("store_full_fp8 requires store_full_kv.")
        if compress_ratio == 4:
            if store_full_kv:
                self.c4_store_full(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req_indices,
                    positions=positions,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    block_size=block_size,
                    rms_norm_weight=rms_norm_weight,
                    rms_norm_eps=rms_norm_eps,
                    cos_sin_cache=cos_sin_cache,
                    kv_cache=kv_cache,
                    kv_slot_mapping=k_cache_metadata.slot_mapping,
                    compress_ratio=compress_ratio,
                    store_full_fp8=store_full_fp8,
                    fp8_scale=fp8_scale,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                )
            else:
                self.c4_store(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req_indices,
                    positions=positions,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    block_size=block_size,
                    rms_norm_weight=rms_norm_weight,
                    rms_norm_eps=rms_norm_eps,
                    cos_sin_cache=cos_sin_cache,
                    kv_cache=kv_cache,
                    kv_slot_mapping=k_cache_metadata.slot_mapping,
                    compress_ratio=compress_ratio,
                    head_dim=head_dim,
                    rope_head_dim=rope_head_dim,
                )
            return
        if compress_ratio != 128:
            raise ValueError(
                "CuTe DSL sparse-attn compressor supports compress_ratio 4 or "
                f"128, got {compress_ratio}.",
            )
        compressed_kv = self.c128_compress(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            head_dim=head_dim,
        )
        if store_full_kv:
            self.c128_store_full(
                compressed_kv=compressed_kv,
                positions=positions,
                slot_mapping=slot_mapping,
                rms_norm_weight=rms_norm_weight,
                rms_norm_eps=rms_norm_eps,
                cos_sin_cache=cos_sin_cache,
                kv_cache=kv_cache,
                kv_slot_mapping=k_cache_metadata.slot_mapping,
                compress_ratio=compress_ratio,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
            )
            return
        self.c128_store(
            compressed_kv=compressed_kv,
            positions=positions,
            slot_mapping=slot_mapping,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
        )


_SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL = (
    SparseAttnCompressNormRopeStoreC4Kernel()
)
_SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL = (
    SparseAttnCompressNormRopeStoreFullC4Kernel()
)
_SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL = SparseAttnCompressC128Block8Kernel()
_SPARSE_ATTN_NORM_ROPE_STORE_KERNEL = SparseAttnNormRopeStoreKernel()
_SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL = SparseAttnNormRopeStoreFullKernel()
_SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL = SparseAttnCompressorCuteDSL(
    c4_store=_SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL,
    c4_store_full=_SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL,
    c128_compress=_SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL,
    c128_store=_SPARSE_ATTN_NORM_ROPE_STORE_KERNEL,
    c128_store_full=_SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL,
)
