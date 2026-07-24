# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float8E4M3FN, Float32, Int64, Uint8, Uint16, Uint32

from vllm.cute_utils import TORCH_TO_CUTE_DTYPE, cvt


def _make_fake_tensor(dtype, shape, divisibility):
    stride = tuple(
        1 if i == len(shape) - 1 else cute.sym_int64(divisibility=divisibility)
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride,
        assumed_align=divisibility * dtype.width // 8,
    )


def is_fused_q_cutedsl_supported(
    q_pe: torch.Tensor,
    index_q: torch.Tensor | None,
    ql_nope: torch.Tensor,
    *,
    has_indexer: bool,
    quantize_mqa: bool,
) -> bool:
    if not (
        quantize_mqa
        and q_pe.dtype == ql_nope.dtype == torch.bfloat16
        and q_pe.shape[-1] == 64
        and ql_nope.shape[-1] == 512
    ):
        return False
    return not has_indexer or (
        index_q is not None
        and index_q.dtype == torch.bfloat16
        and index_q.shape[1] % 16 == 0
        and index_q.shape[-1] == 128
    )


def fused_q_cutedsl(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    rope_cache: torch.Tensor,
    ql_nope: torch.Tensor,
    q_scale: torch.Tensor,
    mqa_output: torch.Tensor,
    idx_q: torch.Tensor,
    idx_rope_cache: torch.Tensor,
    idx_weights: torch.Tensor,
    idx_weights_softmax_scale: float,
    idx_weights_head_scale: float,
    idx_q_fp8: torch.Tensor,
    idx_weights_out: torch.Tensor,
    has_indexer: bool = True,
    index_rope_interleave: bool = True,
) -> None:
    _, num_heads, rope_dim = q_pe.shape
    _, _, nope_dim = ql_nope.shape
    _, num_idx_heads, idx_dim = idx_q.shape

    if has_indexer:
        idx_rope_type = TORCH_TO_CUTE_DTYPE[idx_rope_cache.dtype]
        idx_weights_type = TORCH_TO_CUTE_DTYPE[idx_weights.dtype]
    else:
        idx_dim = num_idx_heads = 0
        idx_q = idx_rope_cache = idx_q_fp8 = None
        idx_weights = idx_weights_out = None
        idx_rope_type = idx_weights_type = None

    rope_type = TORCH_TO_CUTE_DTYPE[rope_cache.dtype]
    compiled = FusedQKernel.compile(
        rope_dim,
        nope_dim,
        num_heads,
        rope_type,
        idx_dim,
        num_idx_heads,
        idx_rope_type,
        idx_weights_type,
        index_rope_interleave,
    )
    compiled(
        positions,
        q_pe,
        rope_cache,
        ql_nope,
        q_scale.view(1),
        mqa_output,
        idx_q,
        idx_rope_cache,
        idx_weights,
        idx_q_fp8,
        idx_weights_out,
        float(idx_weights_softmax_scale * idx_weights_head_scale),
    )


class FusedQKernel:
    def __init__(
        self,
        rope_dim: int,
        nope_dim: int,
        num_heads: int,
        idx_dim: int,
        num_idx_heads: int,
        index_rope_interleave: bool,
    ) -> None:
        assert rope_dim == 64
        assert nope_dim == 512
        assert idx_dim in (128, 0)

        self.rope_dim = rope_dim
        self.nope_dim = nope_dim
        self.num_heads = num_heads
        self.idx_dim = idx_dim
        self.num_idx_heads = num_idx_heads
        self.index_rope_interleave = index_rope_interleave

        # mqa:     rope_dim=64, nope_dim=512, num_heads=64/TP
        # indexer: rope_dim=64, nope_dim=64,  num_heads=32

        self.num_warps = 4
        assert num_heads % self.num_warps == 0
        assert num_idx_heads % (4 * self.num_warps) == 0
        self.num_ctas_per_tok = num_heads // self.num_warps
        self.num_ctas_per_idx_tok = num_idx_heads // (4 * self.num_warps)

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        idx_q: cute.Tensor,
        idx_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
        stream: CUstream,
    ):
        num_tokens = positions.shape[0]
        if cutlass.const_expr(self.idx_dim == 0):
            grid = (num_tokens, self.num_ctas_per_tok, 1)
        else:
            num_mqa_ctas = num_tokens * self.num_ctas_per_tok
            num_idx_ctas = num_tokens * self.num_ctas_per_idx_tok
            grid = (num_mqa_ctas + num_idx_ctas, 1, 1)

        self.kernel(
            positions,
            q_pe,
            rope_cache,
            ql_nope,
            q_scale,
            mqa_output,
            idx_q,
            idx_rope_cache,
            idx_weights,
            idx_q_fp8,
            idx_weights_out,
            weight_scale,
        ).launch(
            grid=grid,
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        q_pe_rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        idx_q: cute.Tensor,
        idx_q_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
    ):
        if cutlass.const_expr(self.idx_dim == 0):
            token_id, group_id, _ = cute.arch.block_idx()
            self.mqa(
                positions,
                q_pe,
                q_pe_rope_cache,
                ql_nope,
                q_scale,
                mqa_output,
                token_id,
                group_id,
            )
        else:
            # CTA-specialization
            bid, _, _ = cute.arch.block_idx()
            num_mqa_ctas = positions.shape[0] * self.num_ctas_per_tok
            if bid < num_mqa_ctas:
                self.mqa(
                    positions,
                    q_pe,
                    q_pe_rope_cache,
                    ql_nope,
                    q_scale,
                    mqa_output,
                    bid // self.num_ctas_per_tok,
                    bid % self.num_ctas_per_tok,
                )
            else:
                bid -= num_mqa_ctas
                self.indexer(
                    positions,
                    idx_q,
                    idx_q_rope_cache,
                    idx_weights,
                    idx_q_fp8,
                    idx_weights_out,
                    weight_scale,
                    bid // self.num_ctas_per_idx_tok,
                    bid % self.num_ctas_per_idx_tok,
                )

    @cute.jit
    def mqa(
        self,
        positions: cute.Tensor,
        q_pe: cute.Tensor,
        q_pe_rope_cache: cute.Tensor,
        ql_nope: cute.Tensor,
        q_scale: cute.Tensor,
        mqa_output: cute.Tensor,
        token_id,
        group_id,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        head_id = group_id * self.num_warps + warp_id

        cute.arch.griddepcontrol_wait()

        pos = positions[token_id]
        inv_scale = 1.0 / q_scale[0]

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_32B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=256)
        cp_16B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)
        cp_4B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=32)
        cp_2B = cute.make_copy_atom(cp_op, Uint8, num_bits_per_copy=16)

        ##### issue all loads asap #####
        rQ_nope_bf16 = cute.make_rmem_tensor(16, BFloat16)
        rQ_rope_bf16 = cute.make_rmem_tensor(2, BFloat16)

        src_ql_nope = cute.local_tile(
            ql_nope[token_id, head_id, None], (16,), (lane_id,)
        )
        src_q_rope = cute.local_tile(q_pe[token_id, head_id, None], (2,), (lane_id,))
        cute.copy(cp_32B, src_ql_nope, rQ_nope_bf16)
        cute.copy(cp_4B, src_q_rope, rQ_rope_bf16)

        rCos_raw = q_pe_rope_cache[pos, 0 + lane_id]
        rSin_raw = q_pe_rope_cache[pos, 32 + lane_id]

        ##### process NoPE #####
        rQ_nope_f32 = cvt.bf16x2_to_fp32x2(rQ_nope_bf16).load() * inv_scale
        rQ_nope_f8 = cute.make_rmem_tensor(16, Float8E4M3FN)
        rQ_nope_f8.store(rQ_nope_f32.to(Float8E4M3FN))
        dst_Q_nope = cute.local_tile(
            mqa_output[token_id, head_id, None], (16,), (lane_id,)
        )
        cute.copy(cp_16B, rQ_nope_f8, dst_Q_nope)

        ##### process RoPE ######
        rQ_rope_f32 = cvt.bf16x2_to_fp32x2(rQ_rope_bf16)
        rCos = rCos_raw.to(Float32)
        rSin = rSin_raw.to(Float32)
        r0 = (rQ_rope_f32[0] * rCos - rQ_rope_f32[1] * rSin) * inv_scale
        r1 = (rQ_rope_f32[1] * rCos + rQ_rope_f32[0] * rSin) * inv_scale

        cute.arch.griddepcontrol_launch_dependents()

        # TensorSSA fp32->fp8 cvt has a bug. rely on direct PTX
        rQ_rope_f8 = cute.make_rmem_tensor(2, Float8E4M3FN)
        cute.recast_tensor(rQ_rope_f8, Uint16)[0] = cvt.fp32x2_to_fp8x2(r0, r1)
        dst_Q_rope = cute.local_tile(
            mqa_output[token_id, head_id, None], (2,), (256 + lane_id,)
        )
        cute.copy(cp_2B, rQ_rope_f8, dst_Q_rope)

    @cute.jit
    def indexer(
        self,
        positions: cute.Tensor,
        idx_q: cute.Tensor,
        idx_q_rope_cache: cute.Tensor,
        idx_weights: cute.Tensor,
        idx_q_fp8: cute.Tensor,
        idx_weights_out: cute.Tensor,
        weight_scale: Float32,
        token_id,
        group_id,
    ):
        tid, _, _ = cute.arch.thread_idx()
        subwarp_id = tid // 8
        sublane_id = tid % 8
        head_id = group_id * (4 * self.num_warps) + subwarp_id
        cute.arch.griddepcontrol_wait()

        pos = positions[token_id]

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_16B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)
        cp_8B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=64)
        cp_4B = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=32)

        ##### issue all loads first #####
        rQ_rope_bf16 = cute.make_rmem_tensor(8, BFloat16)
        if cutlass.const_expr(self.index_rope_interleave):
            src_idx_q_rope = cute.local_tile(
                idx_q[token_id, head_id, None], (8,), (sublane_id,)
            )
            cute.copy(cp_16B, src_idx_q_rope, rQ_rope_bf16)
        else:
            src_idx_q_rope = cute.zipped_divide(
                idx_q[token_id, head_id, None], (4,)
            )  # (4,32)
            cute.copy(
                cp_8B,
                src_idx_q_rope[None, 0 + sublane_id],
                cute.local_tile(rQ_rope_bf16, (4,), (0,)),
            )
            cute.copy(
                cp_8B,
                src_idx_q_rope[None, 8 + sublane_id],
                cute.local_tile(rQ_rope_bf16, (4,), (1,)),
            )

        rQ_nope_bf16 = cute.make_rmem_tensor(8, BFloat16)
        src_idx_q_nope = cute.local_tile(
            idx_q[token_id, head_id, None], (8,), (8 + sublane_id,)
        )
        cute.copy(cp_16B, src_idx_q_nope, rQ_nope_bf16)

        rCos_raw = cute.make_rmem_tensor(4, idx_q_rope_cache.element_type)
        rSin_raw = cute.make_rmem_tensor(4, idx_q_rope_cache.element_type)
        rope_cache_view = cute.zipped_divide(idx_q_rope_cache[pos, None], (4,))

        if cutlass.const_expr(idx_q_rope_cache.element_type == Float32):
            cute.copy(cp_16B, rope_cache_view[None, sublane_id], rCos_raw)
            cute.copy(cp_16B, rope_cache_view[None, 8 + sublane_id], rSin_raw)
        elif cutlass.const_expr(idx_q_rope_cache.element_type == BFloat16):
            cute.copy(cp_8B, rope_cache_view[None, sublane_id], rCos_raw)
            cute.copy(cp_8B, rope_cache_view[None, 8 + sublane_id], rSin_raw)

        # unpack to FP32
        rQ_rope_f32 = cvt.bf16x2_to_fp32x2(rQ_rope_bf16)
        rQ_nope_f32 = cvt.bf16x2_to_fp32x2(rQ_nope_bf16)
        if cutlass.const_expr(idx_q_rope_cache.element_type == Float32):
            rCos = rCos_raw
            rSin = rSin_raw
        elif cutlass.const_expr(idx_q_rope_cache.element_type == BFloat16):
            rCos = cvt.bf16x2_to_fp32x2(rCos_raw)
            rSin = cvt.bf16x2_to_fp32x2(rSin_raw)

        # apply rope
        for i in cutlass.range_constexpr(4):
            if cutlass.const_expr(self.index_rope_interleave):
                r0 = rQ_rope_f32[i * 2 + 0] * rCos[i] - rQ_rope_f32[i * 2 + 1] * rSin[i]
                r1 = rQ_rope_f32[i * 2 + 1] * rCos[i] + rQ_rope_f32[i * 2 + 0] * rSin[i]
                rQ_rope_f32[i * 2 + 0] = r0
                rQ_rope_f32[i * 2 + 1] = r1
            else:
                r0 = rQ_rope_f32[0 + i] * rCos[i] - rQ_rope_f32[4 + i] * rSin[i]
                r1 = rQ_rope_f32[4 + i] * rCos[i] + rQ_rope_f32[0 + i] * rSin[i]
                rQ_rope_f32[0 + i] = r0
                rQ_rope_f32[4 + i] = r1

        # amax
        amax = Float32(1e-4)
        for i in cutlass.range_constexpr(8):
            amax = cute.arch.fmax(amax, cute.math.absf(rQ_rope_f32[i]))
            amax = cute.arch.fmax(amax, cute.math.absf(rQ_nope_f32[i]))

        # warp reduction among 8 lanes
        for i in cutlass.range_constexpr(3):
            other = cute.arch.shuffle_sync_bfly(amax, 1 << i)
            amax = cute.arch.fmax(amax, other)

        # compute scale from amax
        # exp2(ceil(log2(scale))) via bit manipulation
        scale = amax * (1.0 / 448.0)
        bits = scale.bitcast(Uint32)
        exp_bits = (bits + Uint32(0x007FFFFF)) & Uint32(0x7F800000)
        scale = exp_bits.bitcast(Float32)
        inv_scale = (Uint32(0x7F000000) - exp_bits).bitcast(Float32)

        for i in cutlass.range_constexpr(8):
            rQ_nope_f32[i] *= inv_scale
            rQ_rope_f32[i] *= inv_scale

        cute.arch.griddepcontrol_launch_dependents()

        # quantize and store
        rQ_nope_f8 = cute.make_rmem_tensor(8, Float8E4M3FN)
        rQ_nope_f8.store(rQ_nope_f32.load().to(Float8E4M3FN))
        dst_idx_q_nope = cute.local_tile(
            idx_q_fp8[token_id, head_id, None], (8,), (8 + sublane_id,)
        )
        cute.copy(cp_8B, rQ_nope_f8, dst_idx_q_nope)

        rQ_rope_f8 = cute.make_rmem_tensor(8, Float8E4M3FN)
        rQ_rope_f8.store(rQ_rope_f32.load().to(Float8E4M3FN))
        if cutlass.const_expr(self.index_rope_interleave):
            dst_idx_q_rope = cute.local_tile(
                idx_q_fp8[token_id, head_id, None], (8,), (sublane_id,)
            )
            cute.copy(cp_8B, rQ_rope_f8, dst_idx_q_rope)
        else:
            dst_idx_q_rope = cute.zipped_divide(
                idx_q_fp8[token_id, head_id, None], (4,)
            )  # (4,32)
            cute.copy(
                cp_4B,
                cute.local_tile(rQ_rope_f8, (4,), (0,)),
                dst_idx_q_rope[None, 0 + sublane_id],
            )
            cute.copy(
                cp_4B,
                cute.local_tile(rQ_rope_f8, (4,), (1,)),
                dst_idx_q_rope[None, 8 + sublane_id],
            )

        # scale indexer weights
        if sublane_id == 0:
            w = idx_weights[token_id, head_id].to(Float32)
            idx_weights_out[token_id, head_id] = w * scale * weight_scale

    @cache
    @staticmethod
    def compile(
        rope_dim: int,
        nope_dim: int,
        num_heads: int,
        rope_type: type[cutlass.Numeric],
        idx_dim: int,
        num_idx_heads: int,
        idx_rope_type: type[cutlass.Numeric] | None,
        idx_weights_type: type[cutlass.Numeric] | None,
        index_rope_interleave: bool,
    ):
        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()

        positions = _make_fake_tensor(Int64, (num_tokens,), divisibility=1)
        q_pe = _make_fake_tensor(
            BFloat16, (num_tokens, num_heads, rope_dim), divisibility=16
        )
        rope_cache = _make_fake_tensor(rope_type, (max_pos, rope_dim), divisibility=8)
        ql_nope = _make_fake_tensor(
            BFloat16, (num_tokens, num_heads, nope_dim), divisibility=16
        )
        q_scale = _make_fake_tensor(Float32, (1,), divisibility=4)
        mqa_output = _make_fake_tensor(
            Float8E4M3FN,
            (num_tokens, num_heads, nope_dim + rope_dim),
            divisibility=16,
        )

        if idx_rope_type is not None:
            index_q = _make_fake_tensor(
                BFloat16, (num_tokens, num_idx_heads, idx_dim), divisibility=16
            )
            index_rope_cache = _make_fake_tensor(
                idx_rope_type, (max_pos, rope_dim), divisibility=8
            )
            index_weights = _make_fake_tensor(
                idx_weights_type, (num_tokens, num_idx_heads), divisibility=8
            )
            index_q_fp8 = _make_fake_tensor(
                Float8E4M3FN, (num_tokens, num_idx_heads, idx_dim), divisibility=16
            )
            index_weights_out = _make_fake_tensor(
                Float32, (num_tokens, num_idx_heads), divisibility=4
            )
        else:
            index_q = index_rope_cache = index_q_fp8 = None
            index_weights = index_weights_out = None

        kernel = FusedQKernel(
            rope_dim,
            nope_dim,
            num_heads,
            idx_dim,
            num_idx_heads,
            index_rope_interleave,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            positions,
            q_pe,
            rope_cache,
            ql_nope,
            q_scale,
            mqa_output,
            index_q,
            index_rope_cache,
            index_weights,
            index_q_fp8,
            index_weights_out,
            Float32(0.0),
            stream,
            options="--enable-tvm-ffi",
        )
