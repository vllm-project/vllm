# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Uint8, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
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
    DequantGatherKBaseKernel.compile(block_size=block_size)(
        out, k_cache, seq_lens, gather_lens, block_table, offset
    )


def dequant_and_gather_k_cpasync_cutedsl(
    out: torch.Tensor,  # [num_reqs, max_num_tokens, head_size]
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_bytes]
    seq_lens: torch.Tensor,  # [num_reqs]
    gather_lens: torch.Tensor | None,  # [num_reqs]
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_seq]
    block_size: int,
    offset: int,
) -> None:
    _, block_size, _ = k_cache.shape
    DequantGatherKCpasyncKernel.compile(block_size=block_size)(
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


class DequantGatherKBaseKernel:
    # hard-coded for DSv4
    head_dim = 512
    group_size = 64  # 1 scale per 64 elems

    def __init__(self, fp8_dim: int = 448, block_size: int = 64):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.data_dim = fp8_dim + self.bf16_dim * 2
        self.block_size = block_size

        self.num_warps = 4
        self.tb_size = self.num_warps * 32

        # using 8B copy is faster for small shapes
        self.cp_bytes = 16
        self.threads_per_tok = self.head_dim // self.cp_bytes
        self.toks_per_cta = self.tb_size // self.threads_per_tok

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
        k_fp8 = cute.make_tensor(
            k_cache.iterator,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.fp8_dim),
                stride=(k_cache.stride[0], self.data_dim, 1),
            ),
        )
        k_bf16 = cute.make_tensor(
            cute.recast_ptr(k_cache.iterator + self.fp8_dim, dtype=BFloat16),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.bf16_dim),
                stride=(k_cache.stride[0] // 2, self.data_dim // 2, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_cache.iterator + (self.block_size * self.data_dim),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, 8),
                stride=(k_cache.stride[0], 8, 1),
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
        k_fp8: cute.Tensor,  # [num_blocks, block_size, fp8_dim]
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

        cp_bytes = self.cp_bytes
        u32_vec_size = cutlass.const_expr(cp_bytes // 4)

        k_fp8_slice = cute.logical_divide(k_fp8, (None, None, cp_bytes))
        k_bf16_slice = cute.logical_divide(k_bf16, (None, None, cp_bytes))
        out_slice = cute.logical_divide(out, (None, None, cp_bytes))

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]
        start_pos = seq_len - gather_len

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp_fp8_atom = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=cp_bytes * 8)
        cp_bf16_atom = cute.make_copy_atom(
            cp_op, Uint32, num_bits_per_copy=cp_bytes * 8 * 2
        )

        for i in range(
            worker_id * self.toks_per_cta + subwarp_id,
            gather_len,
            num_workers * self.toks_per_cta,
        ):
            pos = start_pos + i
            page_id = block_table[req_id, pos // self.block_size]

            block_offset = pos % self.block_size
            data = cute.make_rmem_tensor((u32_vec_size,), Uint32)

            # k_fp8_slice: (num_blocks, block_size, (cp_bytes, fp8_dim/cp_bytes))
            src_fp8 = k_fp8_slice[page_id, block_offset, (None, sublane_id)]
            cute.copy(cp_fp8_atom, cute.recast_tensor(src_fp8, Uint32), data)

            scale = k_scale[
                page_id, block_offset, sublane_id * cp_bytes // self.group_size
            ]

            scale_u32 = Uint32(scale)
            scale_bf16x2 = (scale_u32 << Uint32(23)) | (scale_u32 << Uint32(7))

            dequant = cute.make_rmem_tensor(u32_vec_size * 2, Uint32)
            for j in cutlass.range_constexpr(u32_vec_size):
                tmp = _fp8x4_to_bf16x4(data[j])

                dequant[j * 2] = _bf16x2_mul(tmp[0], scale_bf16x2)
                dequant[j * 2 + 1] = _bf16x2_mul(tmp[1], scale_bf16x2)

            fp8_threads = cutlass.const_expr(self.fp8_dim // cp_bytes)
            if sublane_id >= fp8_threads:
                # k_bf16_slice: (num_blocks, block_size, (cp_bytes, bf16_dim/cp_bytes))
                src_bf16 = k_bf16_slice[
                    page_id, block_offset, (None, sublane_id - fp8_threads)
                ]
                cute.copy(cp_bf16_atom, cute.recast_tensor(src_bf16, Uint32), dequant)

            # out_slice: (num_reqs, max_num_toks, head_dim)
            dst = out_slice[req_id, offset + i, (None, sublane_id)]
            cute.copy(cp_bf16_atom, dequant, cute.recast_tensor(dst, Uint32))

    @cache
    @staticmethod
    def compile(fp8_dim: int = 448, block_size: int = 64):
        num_reqs = cute.sym_int()
        head_dim = DequantGatherKBaseKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8

        out = make_fake_tensor(BFloat16, (num_reqs, cute.sym_int(), head_dim), 16)
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (cute.sym_int(), block_size, head_bytes),
            stride=(cute.sym_int64(divisibility=32), head_bytes, 1),
            assumed_align=32,
        )
        seq_lens = make_fake_tensor(Int32, (num_reqs,))
        gather_lens = make_fake_tensor(Int32, (num_reqs,))
        block_table = make_fake_tensor(Int32, (num_reqs, cute.sym_int()))

        kernel = DequantGatherKBaseKernel(fp8_dim, block_size)
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


class DequantGatherKCpasyncKernel:
    # hard-coded for DSv4
    head_dim = 512
    group_size = 64  # 1 scale per 64 elems

    def __init__(self, fp8_dim: int = 448, block_size: int = 64):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.data_dim = fp8_dim + self.bf16_dim * 2
        self.block_size = block_size

        self.num_warps = 4
        self.tb_size = self.num_warps * 32
        self.num_stages = 2

        # using 8B copy is faster for small shapes
        self.cp_bytes = 16
        self.threads_per_tok = self.head_dim // self.cp_bytes
        self.toks_per_cta = self.tb_size // self.threads_per_tok

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
        k_fp8 = cute.make_tensor(
            k_cache.iterator,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.fp8_dim),
                stride=(k_cache.stride[0], self.data_dim, 1),
            ),
        )
        k_bf16 = cute.make_tensor(
            cute.recast_ptr(k_cache.iterator + self.fp8_dim, dtype=BFloat16),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.bf16_dim),
                stride=(k_cache.stride[0] // 2, self.data_dim // 2, 1),
            ),
        )
        k_scale = cute.make_tensor(
            k_cache.iterator + (self.block_size * self.data_dim),
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, 8),
                stride=(k_cache.stride[0], 8, 1),
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

    @cute.jit
    def load_g2s(
        self,
        k_fp8_slice: cute.Tensor,
        block_table: cute.Tensor,  # i32[num_reqs, max_blocks_per_req]
        buf_slice: cute.Tensor,  # u32[cp_bytes / 4, num_stages]
        req_id,  # i32
        pos,  # i32
        sublane_id,  # i32
        stage_id,  # i32
    ):
        if cutlass.const_expr(self.cp_bytes == 16):
            cache_mode = cute.nvgpu.LoadCacheMode.GLOBAL  # .cg
        else:
            cache_mode = cute.nvgpu.LoadCacheMode.ALWAYS  # .ca
        cp_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode),
            Uint32,
            num_bits_per_copy=self.cp_bytes * 8,
        )
        page_id = block_table[req_id, pos // self.block_size]
        block_offset = pos % self.block_size

        # k_fp8_slice: [num_blocks, block_size, (cp_bytes, fp8_dim/cp_bytes)]
        src = k_fp8_slice[page_id, block_offset, (None, sublane_id)]
        cute.copy(cp_atom, cute.recast_tensor(src, Uint32), buf_slice[None, stage_id])

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,  # [num_reqs, max_num_tokens, head_size]
        k_fp8: cute.Tensor,  # [num_blocks, block_size, fp8_dim]
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

        cp_bytes = self.cp_bytes
        u32_vec_size = cutlass.const_expr(cp_bytes // 4)

        k_fp8_slice = cute.logical_divide(k_fp8, (None, None, cp_bytes))
        k_bf16_slice = cute.logical_divide(k_bf16, (None, None, cp_bytes))
        out_slice = cute.logical_divide(out, (None, None, cp_bytes))

        smem = cutlass.utils.SmemAllocator()
        smem_buf = smem.allocate_tensor(
            Uint32,
            cute.make_layout((self.head_dim // 4, self.toks_per_cta, self.num_stages)),
            byte_alignment=16,
        )
        buf_slice = cute.logical_divide(smem_buf, (u32_vec_size, None, None))[
            (None, sublane_id), subwarp_id, None
        ]

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]
        start_pos = seq_len - gather_len

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

        for i in cutlass.range_constexpr(self.num_stages - 1):
            next_pos = (
                start_pos
                + worker_id * self.toks_per_cta
                + subwarp_id
                + i * num_workers * self.toks_per_cta
            )
            if next_pos < seq_len:
                self.load_g2s(
                    k_fp8_slice, block_table, buf_slice, req_id, next_pos, sublane_id, i
                )
            cute.arch.cp_async_commit_group()
        prefetch_stage = self.num_stages - 1

        compute_stage = 0

        for i in range(
            worker_id * self.toks_per_cta + subwarp_id,
            gather_len,
            num_workers * self.toks_per_cta,
        ):
            pos = start_pos + i

            next_pos = pos + num_workers * self.toks_per_cta * (self.num_stages - 1)
            if next_pos < seq_len:
                self.load_g2s(
                    k_fp8_slice,
                    block_table,
                    buf_slice,
                    req_id,
                    next_pos,
                    sublane_id,
                    prefetch_stage,
                )
                prefetch_stage = (prefetch_stage + 1) % self.num_stages
            cute.arch.cp_async_commit_group()

            page_id = block_table[req_id, pos // self.block_size]
            block_offset = pos % self.block_size
            data = cute.make_rmem_tensor((u32_vec_size,), Uint32)

            scale = k_scale[
                page_id, block_offset, sublane_id * cp_bytes // self.group_size
            ]

            # don't need syncthreads/syncwarp because
            # the same thread loads the same data
            cute.arch.cp_async_wait_group(self.num_stages - 1)
            cute.copy(cp_fp8_atom, buf_slice[None, compute_stage], data)
            compute_stage = (compute_stage + 1) % self.num_stages

            # convert to bf16x2 via bit manipulation
            scale_u32 = Uint32(scale)
            scale_bf16x2 = (scale_u32 << Uint32(23)) | (scale_u32 << Uint32(7))

            # cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 requires PTX 9.2 (CUDA 13.2)
            dequant = cute.make_rmem_tensor(u32_vec_size * 2, Uint32)
            for j in cutlass.range_constexpr(u32_vec_size):
                tmp = _fp8x4_to_bf16x4(data[j])

                # bf16 multiply is safe
                dequant[j * 2] = _bf16x2_mul(tmp[0], scale_bf16x2)
                dequant[j * 2 + 1] = _bf16x2_mul(tmp[1], scale_bf16x2)

            # the last 4 threads load BF16 data
            fp8_threads = cutlass.const_expr(self.fp8_dim // cp_bytes)
            if sublane_id >= fp8_threads:
                # k_bf16_slice: (num_blocks, block_size, (cp_bytes, bf16_dim/cp_bytes))
                src_bf16 = k_bf16_slice[
                    page_id, block_offset, (None, sublane_id - fp8_threads)
                ]
                cute.copy(cp_bf16_atom, cute.recast_tensor(src_bf16, Uint32), dequant)

            # out_slice: (num_reqs, max_num_toks, head_dim)
            dst = out_slice[req_id, offset + i, (None, sublane_id)]
            cute.copy(cp_bf16_atom, dequant, cute.recast_tensor(dst, Uint32))

    @cache
    @staticmethod
    def compile(fp8_dim: int = 448, block_size: int = 64):
        num_reqs = cute.sym_int()
        head_dim = DequantGatherKCpasyncKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8

        out = make_fake_tensor(BFloat16, (num_reqs, cute.sym_int(), head_dim), 16)
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (cute.sym_int(), block_size, head_bytes),
            stride=(cute.sym_int64(divisibility=32), head_bytes, 1),
            assumed_align=32,
        )
        seq_lens = make_fake_tensor(Int32, (num_reqs,))
        gather_lens = make_fake_tensor(Int32, (num_reqs,))
        block_table = make_fake_tensor(Int32, (num_reqs, cute.sym_int()))

        kernel = DequantGatherKCpasyncKernel(fp8_dim, block_size)
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
