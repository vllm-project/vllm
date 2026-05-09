# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# once we have more CuteDSL kernels in vLLM, we can refactor small helper functions
# to a separate file
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Uint8, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor


def dequant_and_gather_k_cutedsl(
    out: torch.Tensor,  # [num_reqs, max_num_tokens, head_size]
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_bytes]
    seq_lens: torch.Tensor,  # [num_reqs]
    gather_lens: torch.Tensor | None,  # [num_reqs]
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq]
    block_size: int,
    offset: int,
) -> None:
    _, block_size, _ = k_cache.shape
    DequantGatherKCacheKernel.compile(block_size=block_size)(
        out, k_cache, seq_lens, gather_lens, block_table, offset
    )


@dsl_user_op
def _fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    # there is only fp8->fp16, no fp8->bf16,
    # so we have this monster here
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 x0, x1;\n\t"
        ".reg .b16 t00, t01, t10, t11;\n\t"
        "mov.b32 {x0, x1}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, x0;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, x1;\n\t"
        "mov.b32 {t00, t01}, $0;\n\t"
        "mov.b32 {t10, t11}, $1;\n\t"
        "cvt.rn.bf16.f16 t00, t00;\n\t"
        "cvt.rn.bf16.f16 t01, t01;\n\t"
        "cvt.rn.bf16.f16 t10, t10;\n\t"
        "cvt.rn.bf16.f16 t11, t11;\n\t"
        "mov.b32 $0, {t00, t01};\n\t"
        "mov.b32 $1, {t10, t11};\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def _bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


class DequantGatherKCacheKernel:
    # hard-coded for DSv4
    head_dim = 512
    group_size = 64  # 1 scale per 64 elems

    def __init__(self, fp8_dim: int = 448, block_size: int = 64):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.block_size = block_size

        self.num_warps = 4
        self.tb_size = self.num_warps * 32

        self.cp_bytes = 16
        self.threads_per_tok = self.head_dim // self.cp_bytes

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,  # [num_reqs, max_num_tokens, head_size]
        k_cache: cute.Tensor,  # [num_blocks, block_size, head_bytes]
        seq_lens: cute.Tensor,  # [num_reqs]
        gather_lens: cute.Tensor | None,  # [num_reqs]
        block_table: cute.Tensor,  # [num_reqs, max_blocks_per_req]
        offset: Int32,
        stream: CUstream,
    ):
        # split k_cache into k_data and k_scale
        # each [block_size, head_bytes] block is actually a concat of
        # [block_size, fp8_dim + bf16_dim * 2] and [block_size, 8]
        data_dim = cutlass.const_expr(self.fp8_dim + self.bf16_dim * 2)
        k_ptr = cute.make_ptr(Uint8, k_cache.iterator.toint(), assumed_align=32)
        cache_stride = cute.assume(k_cache.stride[0], 32)
        k_fp8 = cute.make_tensor(
            k_ptr,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.fp8_dim),
                stride=(cache_stride, data_dim, 1),
            ),
        )
        k_bf16 = cute.make_tensor(
            cute.recast_ptr(k_ptr + self.fp8_dim, dtype=BFloat16),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.bf16_dim),
                stride=(cache_stride // 2, data_dim // 2, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_ptr + (self.block_size * data_dim),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, 8),
                stride=(cache_stride, 8, 1),
            ),
        )

        grid = (out.shape[0], 1024, 1)
        self.kernel(
            out,
            k_fp8,
            k_bf16,
            k_scale,
            seq_lens,
            gather_lens,
            block_table,
            offset,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,  # [num_reqs, max_num_tokens, head_size]
        k_fp8: cute.Tensor,  # [num_blocks, block_size, fp8_dim + bf16_dim * 2]
        k_bf16: cute.Tensor,  # [num_blocks, block_size, bf16_dim]
        k_scale: cute.Tensor,  # [num_blocks, block_size, 8]
        seq_lens: cute.Tensor,  # [num_reqs]
        gather_lens: cute.Tensor | None,  # [num_reqs]
        block_table: cute.Tensor,  # [num_reqs, max_blocks_per_req]
        offset: Int32,
    ):
        req_id, worker_id, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        _, num_workers, _ = cute.arch.grid_dim()

        subwarp_id = tid // self.threads_per_tok
        sublane_id = tid % self.threads_per_tok

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]
        start_pos = seq_len - gather_len

        cp_bytes = self.cp_bytes

        # at each position, we have 448 FP8 values + 64 BF16 values.
        # to make our lives easier, we will use 16B loads for FP8,
        # then 32B stores for the dequantized BF16.
        # hence 1 warp (32 threads) can cover 1 position/token exactly.
        #
        # the first 28 threads do 16B loads = 448 FP8 values.
        # the last 4 threads do 32B loads = 64 BF16 values.
        # then the whole warp do 32B stores = 512 BF16 values.
        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_fp8_atom = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=cp_bytes * 8)
        cp_bf16_atom = cute.make_copy_atom(
            cp_op, Uint32, num_bits_per_copy=cp_bytes * 8 * 2
        )

        for i in range(
            worker_id * self.num_warps + subwarp_id,
            gather_len,
            num_workers * self.num_warps,
        ):
            pos = start_pos + i
            page_id = block_table[req_id, pos // self.block_size]

            # we don't do bounds check here to avoid warp divergence.
            # the last 4 threads will load and dequantize to garbage.
            # due to our custom K cache layout and how we reconstruct
            # the view, cute.autovec_copy() doesn't work without adding
            # extensive alignment/divisibility hints. to keep the code
            # compact, we will just issue ld PTX directly.
            k_block_offset = pos % self.block_size
            data = cute.make_rmem_tensor((cp_bytes // 4,), Uint32)
            src = cute.local_tile(
                k_fp8[page_id, k_block_offset, None],
                tiler=(cp_bytes,),
                coord=(sublane_id,),
            )
            cute.copy(cp_fp8_atom, cute.recast_tensor(src, Uint32), data)
            scale = k_scale[
                page_id, k_block_offset, sublane_id * cp_bytes // self.group_size
            ]

            # convert to bf16x2 via bit manipulation
            scale_u32 = Uint32(scale)
            scale_bf16x2 = (scale_u32 << Uint32(23)) | (scale_u32 << Uint32(7))

            # cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 requires PTX 9.2 (CUDA 13.2)
            dequant = cute.make_rmem_tensor(cp_bytes // 4 * 2, Uint32)
            for j in cutlass.range_constexpr(cp_bytes // 4):
                tmp = _fp8x4_to_bf16x4(data[j])

                # bf16 multiply is safe
                dequant[j * 2] = _bf16x2_mul(tmp[0], scale_bf16x2)
                dequant[j * 2 + 1] = _bf16x2_mul(tmp[1], scale_bf16x2)

            # the last 4 threads load BF16 data
            fp8_threads = cutlass.const_expr(self.fp8_dim // cp_bytes)
            if sublane_id >= fp8_threads:
                src_ = cute.local_tile(
                    k_bf16[page_id, k_block_offset, None],
                    tiler=(cp_bytes,),
                    coord=(sublane_id - fp8_threads,),
                )
                cute.copy(cp_bf16_atom, cute.recast_tensor(src_, Uint32), dequant)

            dst = cute.local_tile(
                out[req_id, offset + i, None],
                tiler=(cp_bytes,),
                coord=(sublane_id,),
            )
            cute.copy(cp_bf16_atom, dequant, cute.recast_tensor(dst, Uint32))

    @cache
    @staticmethod
    def compile(fp8_dim: int = 448, block_size: int = 64):
        num_reqs = cute.sym_int()
        head_dim = DequantGatherKCacheKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8

        out = make_fake_tensor(BFloat16, (num_reqs, cute.sym_int(), head_dim), 16)
        k_cache = make_fake_tensor(Uint8, (cute.sym_int(), block_size, head_bytes))
        seq_lens = make_fake_tensor(Int32, (num_reqs,))
        gather_lens = make_fake_tensor(Int32, (num_reqs,))
        block_table = make_fake_tensor(Int32, (num_reqs, cute.sym_int()))

        kernel = DequantGatherKCacheKernel(fp8_dim, block_size)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            Int32(0),
            stream,
            options="--enable-tvm-ffi",
        )
