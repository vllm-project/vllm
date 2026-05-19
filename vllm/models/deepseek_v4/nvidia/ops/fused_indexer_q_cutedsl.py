# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int64, Uint8, Uint32, const_expr
from quack.compile_utils import make_fake_tensor

from vllm.models.deepseek_v4.nvidia.ops.cutedsl_utils import (
    _bf16x2_abs,
    _bf16x2_max,
    _bf16x2_to_fp32,
    _fp32x2_to_bf16x2,
    _fp32x4_to_fp8x4,
    _fp32x8_to_fp4x8,
    _recast_val,
)
from vllm.vllm_flash_attn.cute import utils as cute_utils

# MXFP4: 32 elements per block, packed 2 nibbles per byte, ue8m0 block scale.
MXFP4_BLOCK_SIZE = 32

_TORCH_TO_CUTE = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
}


def fused_indexer_q_rope_quant_mxfp4_cutedsl(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    index_q_packed: torch.Tensor,
    index_q_scale: torch.Tensor,
    index_weights_out: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = index_q.shape
    rope_dim = index_q_cos_sin_cache.shape[-1]
    rope_type = _TORCH_TO_CUTE[index_q_cos_sin_cache.dtype]

    # compile all variants at first invocation
    for coarsen in (1, 4):
        IndexerQMxFp4Kernel.compile(head_dim, rope_dim, num_heads, rope_type, coarsen)

    # heuristic
    coarsen = 1 if num_tokens < 512 else 4
    compiled = IndexerQMxFp4Kernel.compile(
        head_dim, rope_dim, num_heads, rope_type, coarsen
    )
    scale = float(index_weights_softmax_scale * index_weights_head_scale)
    compiled(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_q_packed,
        index_q_scale,
        index_weights_out,
        scale,
    )


def fused_indexer_q_rope_quant_fp8_cutedsl(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    index_q_fp8: torch.Tensor,
    index_weights_out: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = index_q.shape
    rope_dim = index_q_cos_sin_cache.shape[-1]
    rope_type = _TORCH_TO_CUTE[index_q_cos_sin_cache.dtype]

    for coarsen in (1, 4):
        IndexerQFp8Kernel.compile(head_dim, rope_dim, num_heads, rope_type, coarsen)

    coarsen = 1 if num_tokens < 512 else 4
    compiled = IndexerQFp8Kernel.compile(
        head_dim, rope_dim, num_heads, rope_type, coarsen
    )
    scale = float(index_weights_softmax_scale * index_weights_head_scale)
    # The cute kernel treats the FP8 buffer as raw bytes (Uint8).
    compiled(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_weights,
        index_q_fp8.view(torch.uint8),
        index_weights_out,
        scale,
    )


class IndexerQRopeQuantKernel:
    """Shared infrastructure for indexer-Q RoPE+quant fused kernels.

    Subclasses implement ``kernel`` for a particular Q quantization scheme
    (MXFP4, FP8 e4m3, …). The base class owns the launch geometry and the
    common preamble: thread/token addressing, the BF16 Q load, and the
    interleaved-RoPE pass over the trailing ``rope_dim`` lanes.
    """

    def __init__(
        self,
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
        coarsen: int = 4,
    ):
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.num_heads = num_heads
        self.cos_sin_dtype = cos_sin_dtype

        # process multiple heads at the same time to armotize RoPE load costs
        assert num_heads % coarsen == 0
        self.coarsen = coarsen

        # later we will use 32B load = 16 BF16 elems
        # thus, head_dim=128 requires 8 threads to handle.
        # let's call subwarp = 8 threads.
        self.subwarp_size = head_dim // 16
        self.tb_size = 128
        self.threads_per_token = (self.num_heads // self.coarsen) * self.subwarp_size

    @cute.jit
    def _load_q_and_rope(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
    ):
        """Compute thread indices, load Q (BF16), and apply interleaved RoPE.

        Returns a tuple
            (q_bf16x2, tid, global_tid, sublane, token_id, head_tile_id,
             head_start, in_bounds, num_token_heads)
        where ``q_bf16x2`` is a (coarsen, 8) rmem tile of Uint32 packed
        bf16x2 pairs covering the 16 BF16 lanes owned by this thread for
        each of ``coarsen`` heads. RoPE is applied in place to the
        trailing ``rope_dim`` lanes; the leading nope lanes pass through.
        """
        block_id, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        num_tokens = q.shape[0]
        num_token_heads = num_tokens * self.num_heads
        global_tid = block_id * self.tb_size + tid

        global_subwarp_id = global_tid // self.subwarp_size
        sublane = tid % self.subwarp_size

        token_id = global_subwarp_id // (self.num_heads // self.coarsen)
        head_tile_id = global_subwarp_id % (self.num_heads // self.coarsen)
        head_start = head_tile_id * self.coarsen

        # NOTE: token_id may exceed bounds, hence we need to add load/store guards
        # we can't do early exit because CuteDSL doesn't support it. and we also need
        # all threads in a warp to be active since we utilize warp shuffle later.
        # must_in_bounds is constexpr, True when 1 threadblock fit within 1 token
        # position. the compiler will remove bounds check when that happens.
        must_in_bounds = cutlass.const_expr(self.tb_size % self.threads_per_token == 0)
        in_bounds = must_in_bounds or (token_id < num_tokens)

        cp_op = cute.nvgpu.CopyUniversalOp()

        _layout = cute.make_layout((self.coarsen, 8), stride=(8, 1))
        q_bf16x2 = cute.make_rmem_tensor(_layout, Uint32)

        if in_bounds:
            # we can't do cute.copy() on the whole 2D tile directly because
            # cute.copy() wants the 1st mode to be covered by the copy atom,
            # and other modes as for loop. there is no fast way to
            # "transpose" the tensor view.
            q_tile = cute.local_tile(
                q[token_id, None, None],
                tiler=(self.coarsen, 16),
                coord=(head_tile_id, sublane),
            )
            cp_u32x8 = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=256)
            for i in cutlass.range_constexpr(self.coarsen):
                src = cute.recast_tensor(q_tile[i, None], Uint32)
                cute.copy(cp_u32x8, src, q_bf16x2[i, None])

        # RoPE applies only to the trailing rope_dim values. We keep the rounded
        # BF16 result in q_bits so the later amax and quantization see BF16.
        # cos_sin_cache layout: [max_pos, rope_dim]
        if in_bounds and sublane * 16 >= self.nope_dim:
            cos_vals = cute.make_rmem_tensor((8,), Float32)
            sin_vals = cute.make_rmem_tensor((8,), Float32)

            pos = positions[token_id]

            # select 8 elems from cos and sin
            cos_id = sublane - self.nope_dim // 16
            sin_id = cos_id + self.rope_dim // 16
            cos_src = cute.local_tile(
                cos_sin_cache[pos, None], tiler=(8,), coord=(cos_id,)
            )
            sin_src = cute.local_tile(
                cos_sin_cache[pos, None], tiler=(8,), coord=(sin_id,)
            )

            cp_f32x8 = cute.make_copy_atom(cp_op, Float32, num_bits_per_copy=256)
            cp_u32x4 = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)

            if const_expr(self.cos_sin_dtype is Float32):
                cute.copy(cp_f32x8, cos_src, cos_vals)
                cute.copy(cp_f32x8, sin_src, sin_vals)
            else:
                cos_bf16x2 = cute.make_rmem_tensor((4,), Uint32)
                sin_bf16x2 = cute.make_rmem_tensor((4,), Uint32)
                cute.copy(cp_u32x4, cute.recast_tensor(cos_src, Uint32), cos_bf16x2)
                cute.copy(cp_u32x4, cute.recast_tensor(sin_src, Uint32), sin_bf16x2)

                for i in cutlass.range_constexpr(4):
                    cos0, cos1 = _bf16x2_to_fp32(cos_bf16x2[i])
                    sin0, sin1 = _bf16x2_to_fp32(sin_bf16x2[i])
                    cos_vals[i * 2] = cos0
                    cos_vals[i * 2 + 1] = cos1
                    sin_vals[i * 2] = sin0
                    sin_vals[i * 2 + 1] = sin1

            for i in cutlass.range_constexpr(self.coarsen):
                for j in cutlass.range_constexpr(8):
                    q0, q1 = _bf16x2_to_fp32(q_bf16x2[i, j])
                    rot0 = q0 * cos_vals[j] - q1 * sin_vals[j]
                    rot1 = q0 * sin_vals[j] + q1 * cos_vals[j]
                    # convert back to BF16 to match numerics
                    q_bf16x2[i, j] = _fp32x2_to_bf16x2(rot0, rot1)

        return (
            q_bf16x2,
            tid,
            global_tid,
            sublane,
            token_id,
            head_tile_id,
            head_start,
            in_bounds,
            num_token_heads,
        )


class IndexerQMxFp4Kernel(IndexerQRopeQuantKernel):
    """Eight-thread subwarps process one ``(token, head)`` row."""

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_quant: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
        stream: CUstream,
    ):
        total_threads = q.shape[0] * self.threads_per_token
        grid = (cute.ceil_div(total_threads, self.tb_size), 1, 1)
        self.kernel(
            positions,
            q,
            cos_sin_cache,
            weights,
            q_quant,
            q_scale,
            weights_out,
            scale,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_quant: cute.Tensor,
        q_scale: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
    ):
        (
            q_bf16x2,
            tid,
            global_tid,
            sublane,
            token_id,
            head_tile_id,
            head_start,
            in_bounds,
            num_token_heads,
        ) = self._load_q_and_rope(positions, q, cos_sin_cache)

        cp_op = cute.nvgpu.CopyUniversalOp()

        # layout: [coarsen, 8]
        q_fp4_tile = cute.local_tile(
            q_quant[token_id, None, None],
            tiler=(self.coarsen, 8),
            coord=(head_tile_id, sublane),
        )

        for i in cutlass.range_constexpr(self.coarsen):
            # compute amax in packed bf16x2 to save instructions
            # Each thread holds 16 elems. Two adjacent threads form one 32-elem
            # MXFP4 block, so a width-2 shuffle gives the block amax.
            amax_bf16x2 = _bf16x2_abs(q_bf16x2[i, 0])
            for j in cutlass.range_constexpr(1, 8):
                amax_bf16x2 = _bf16x2_max(amax_bf16x2, _bf16x2_abs(q_bf16x2[i, j]))
            amax_bf16x2 = cute_utils.warp_reduce(
                amax_bf16x2,
                _bf16x2_max,
                width=MXFP4_BLOCK_SIZE // 16,
            )
            amax_pair = _bf16x2_to_fp32(amax_bf16x2)
            amax = cute_utils.fmax(amax_pair[0], amax_pair[1])

            if in_bounds:
                # compute block scale with bit manipulation
                # UE8M0 stores ceil(log2(fp4_scale)) + 127. Adding the mantissa mask
                # increments the exponent whenever fp4_scale is not exactly a power of 2
                eps = cutlass.const_expr(float.fromhex("0x6p-126"))
                fp4_scale = cute_utils.fmax(amax, eps) * Float32(1.0 / 6.0)
                bits = _recast_val(fp4_scale, Uint32)
                ue8m0 = cute_utils.shr_u32(
                    bits + Uint32(0x7FFFFF), Uint32(23)
                ) & Uint32(0xFF)

                # Only one of the two threads in an MXFP4 block writes the shared scale.
                if tid % 2 == 0:
                    mx_block = sublane // 2
                    q_scale[token_id, head_start + i, mx_block] = Uint8(ue8m0)

                # If scale = 2^A and ue8m0 = A + 127, then inverse scale has exponent
                # -A + 127 = 254 - ue8m0.
                inv_scale_bits = (Uint32(254) - ue8m0) << Uint32(23)
                inv_fp4_scale = _recast_val(inv_scale_bits, Float32)

                vals = cute.make_rmem_tensor(16, Float32)
                for j in cutlass.range_constexpr(8):
                    q0, q1 = _bf16x2_to_fp32(q_bf16x2[i, j])
                    vals[j * 2] = q0 * inv_fp4_scale
                    vals[j * 2 + 1] = q1 * inv_fp4_scale

                # pack to FP4
                packed = cute.make_rmem_tensor((2,), Uint32)
                packed[0] = _fp32x8_to_fp4x8(vals, 0)
                packed[1] = _fp32x8_to_fp4x8(vals, 8)

                dst = q_fp4_tile[i, None]
                cp_u32x2 = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=64)
                cute.copy(cp_u32x2, packed, cute.recast_tensor(dst, Uint32))

        # Weight scaling is independent of the Q subwarp work. The first
        # num_tokens * num_heads logical threads cover one weight each.
        if global_tid < num_token_heads:
            weight_token_id = global_tid // self.num_heads
            weight_head_id = global_tid % self.num_heads
            weights_out[weight_token_id, weight_head_id] = (
                weights[weight_token_id, weight_head_id].to(Float32) * scale
            )

    @cache
    @staticmethod
    def compile(
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
        coarsen: int = 4,
    ):
        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()

        q = make_fake_tensor(
            BFloat16, (num_tokens, num_heads, head_dim), divisibility=16
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=1)
        cos_sin_cache = make_fake_tensor(
            cos_sin_dtype,
            (max_pos, rope_dim),
            divisibility=8,
        )
        weights = make_fake_tensor(BFloat16, (num_tokens, num_heads), divisibility=8)
        q_fp4 = make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // 2),
            divisibility=16,
        )
        q_scale = make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE),
            divisibility=4,
        )
        weights_out = make_fake_tensor(Float32, (num_tokens, num_heads), divisibility=4)

        kernel = IndexerQMxFp4Kernel(
            head_dim, rope_dim, num_heads, cos_sin_dtype, coarsen
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp4,
            q_scale,
            weights_out,
            Float32(0.0),
            stream,
            options="--enable-tvm-ffi",
        )


class IndexerQFp8Kernel(IndexerQRopeQuantKernel):
    """Eight-thread subwarps process one ``(token, head)`` row and emit
    float8 e4m3fn with a single per-(token, head) scalar scale folded
    into the per-token weight (mirrors ``_fused_indexer_q_rope_quant_kernel``).
    """

    def __init__(
        self,
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
        coarsen: int = 4,
    ):
        super().__init__(head_dim, rope_dim, num_heads, cos_sin_dtype, coarsen)
        # Each subwarp owns `coarsen` heads; we use the first `coarsen`
        # threads of the subwarp to write the per-head weights using the
        # fp8 scale computed in the matching loop iteration.
        assert self.coarsen <= self.subwarp_size, (
            f"FP8 kernel requires coarsen ({self.coarsen}) <= "
            f"subwarp_size ({self.subwarp_size}) for the weight-fold step"
        )

    @cute.jit
    def __call__(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_fp8: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
        stream: CUstream,
    ):
        total_threads = q.shape[0] * self.threads_per_token
        grid = (cute.ceil_div(total_threads, self.tb_size), 1, 1)
        self.kernel(
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp8,
            weights_out,
            scale,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        positions: cute.Tensor,
        q: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        weights: cute.Tensor,
        q_fp8: cute.Tensor,
        weights_out: cute.Tensor,
        scale: Float32,
    ):
        (
            q_bf16x2,
            _tid,
            _global_tid,
            sublane,
            token_id,
            head_tile_id,
            head_start,
            in_bounds,
            _num_token_heads,
        ) = self._load_q_and_rope(positions, q, cos_sin_cache)

        cp_op = cute.nvgpu.CopyUniversalOp()

        # layout: [coarsen, 16] bytes (one e4m3fn per element).
        q_fp8_tile = cute.local_tile(
            q_fp8[token_id, None, None],
            tiler=(self.coarsen, 16),
            coord=(head_tile_id, sublane),
        )

        for i in cutlass.range_constexpr(self.coarsen):
            # Reduce amax across the full head_dim: each thread already holds
            # the max over its 16 lanes; a width=subwarp_size warp shuffle
            # spreads the head-wide max to every lane in the subwarp.
            amax_bf16x2 = _bf16x2_abs(q_bf16x2[i, 0])
            for j in cutlass.range_constexpr(1, 8):
                amax_bf16x2 = _bf16x2_max(amax_bf16x2, _bf16x2_abs(q_bf16x2[i, j]))
            amax_bf16x2 = cute_utils.warp_reduce(
                amax_bf16x2,
                _bf16x2_max,
                width=self.subwarp_size,
            )
            amax_pair = _bf16x2_to_fp32(amax_bf16x2)
            amax = cute_utils.fmax(amax_pair[0], amax_pair[1])

            # scale = max(amax, eps) / fp8_max, then rounded UP to the next
            # power of two. Adding the mantissa mask before shifting out the
            # mantissa bumps the exponent whenever s isn't a pure pow2.
            fp32_scale = cute_utils.fmax(amax, Float32(1e-4)) * Float32(1.0 / 448.0)
            bits = _recast_val(fp32_scale, Uint32)
            scale_exp = cute_utils.shr_u32(
                bits + Uint32(0x7FFFFF), Uint32(23)
            ) & Uint32(0xFF)

            # rounded scale = 2^(scale_exp - 127); bit pattern is scale_exp << 23
            fp8_scale_bits = scale_exp << Uint32(23)
            fp8_scale = _recast_val(fp8_scale_bits, Float32)
            # inverse = 2^-(scale_exp - 127); bit pattern is (254 - scale_exp) << 23
            inv_scale_bits = (Uint32(254) - scale_exp) << Uint32(23)
            inv_fp8_scale = _recast_val(inv_scale_bits, Float32)

            # Weight fold: weights_out = weights * q_scale * scale_combined.
            # All threads in the subwarp share the same fp8_scale after the
            # warp_reduce above, so we let thread `sublane == i` write the
            # weight for head `head_start + i`.
            if in_bounds and sublane == i:
                head_id = head_start + i
                weights_out[token_id, head_id] = (
                    weights[token_id, head_id].to(Float32) * scale * fp8_scale
                )

            if in_bounds:
                # 16 BF16 → 16 e4m3 bytes per thread, packed into 4 b32s
                # (one cp.async-shaped 128-bit store per row).
                packed = cute.make_rmem_tensor((4,), Uint32)
                for j in cutlass.range_constexpr(4):
                    q0, q1 = _bf16x2_to_fp32(q_bf16x2[i, j * 2])
                    q2, q3 = _bf16x2_to_fp32(q_bf16x2[i, j * 2 + 1])
                    packed[j] = _fp32x4_to_fp8x4(
                        q0 * inv_fp8_scale,
                        q1 * inv_fp8_scale,
                        q2 * inv_fp8_scale,
                        q3 * inv_fp8_scale,
                    )

                dst = q_fp8_tile[i, None]
                cp_u32x4 = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)
                cute.copy(cp_u32x4, packed, cute.recast_tensor(dst, Uint32))

    @cache
    @staticmethod
    def compile(
        head_dim: int = 128,
        rope_dim: int = 64,
        num_heads: int = 64,
        cos_sin_dtype: type[cutlass.Numeric] = Float32,
        coarsen: int = 4,
    ):
        num_tokens = cute.sym_int()
        max_pos = cute.sym_int()

        q = make_fake_tensor(
            BFloat16, (num_tokens, num_heads, head_dim), divisibility=16
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=1)
        cos_sin_cache = make_fake_tensor(
            cos_sin_dtype,
            (max_pos, rope_dim),
            divisibility=8,
        )
        weights = make_fake_tensor(BFloat16, (num_tokens, num_heads), divisibility=8)
        q_fp8 = make_fake_tensor(
            Uint8,
            (num_tokens, num_heads, head_dim),
            divisibility=16,
        )
        weights_out = make_fake_tensor(Float32, (num_tokens, num_heads), divisibility=4)

        kernel = IndexerQFp8Kernel(
            head_dim, rope_dim, num_heads, cos_sin_dtype, coarsen
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            positions,
            q,
            cos_sin_cache,
            weights,
            q_fp8,
            weights_out,
            Float32(0.0),
            stream,
            options="--enable-tvm-ffi",
        )
