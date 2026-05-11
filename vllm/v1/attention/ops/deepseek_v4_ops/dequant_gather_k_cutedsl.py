# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Int32, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from vllm.v1.attention.ops.deepseek_v4_ops.cutedsl_utils import (
    _bf16x2_mul,
    _fp8x4_to_bf16x4,
)


def dequantize_and_gather_k_cache_cutedsl(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    DequantGatherKCacheKernel.compile(
        block_size=block_size,
        has_gather_lens=gather_lens is not None,
    )(out, k_cache, seq_lens, gather_lens, block_table, offset)


class DequantGatherKCacheKernel:
    # Hard-coded for DSv4.
    head_dim = 512
    group_size = 64  # 1 scale per 64 elems

    def __init__(self, fp8_dim: int = 448, block_size: int = 64):
        self.fp8_dim = fp8_dim
        self.bf16_dim = self.head_dim - fp8_dim
        self.data_dim = fp8_dim + self.bf16_dim * 2
        self.block_size = block_size

        self.num_warps = 4
        self.tb_size = self.num_warps * 32
        self.num_stages = 4

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,
        k_cache: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor | None,
        block_table: cute.Tensor,
        offset: Int32,
        stream: CUstream,
    ):
        # Split k_cache into k_data and k_scale. Each [block_size, head_bytes]
        # block is actually a concat of
        # [block_size, fp8_dim + bf16_dim * 2] and [block_size, 8].
        k_data = cute.make_tensor(
            k_cache.iterator,
            layout=cute.make_layout(
                (k_cache.shape[0], self.block_size, self.data_dim),
                stride=(k_cache.stride[0], self.data_dim, 1),
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
            k_data,
            k_scale,
            seq_lens,
            gather_lens,
            block_table,
            offset,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.jit
    def load_g2s(
        self,
        k_data_slice: cute.Tensor,
        k_scale: cute.Tensor,
        block_table: cute.Tensor,
        s_kdata_slice: cute.Tensor,
        s_kscale: cute.Tensor,
        req_id,
        pos,
        lane_id,
        stage_id,
    ):
        # k_data_slice: [num_blocks, block_size, (16, data_dim/16)]
        # s_kdata_slice: [(4, data_dim/16), num_stages]

        op = cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL)
        cp16_atom = cute.make_copy_atom(op, Uint32, num_bits_per_copy=128)
        cp8_atom = cute.make_copy_atom(cpasync.CopyG2SOp(), Uint8, num_bits_per_copy=64)
        page_id = block_table[req_id, pos // self.block_size]
        block_offset = pos % self.block_size

        # Load the first 512 bytes (32x16B).
        idx = lane_id
        src = k_data_slice[page_id, block_offset, (None, idx)]
        cute.copy(
            cp16_atom,
            cute.recast_tensor(src, Uint32),
            s_kdata_slice[(None, idx), stage_id],
        )

        # Load the tail 64 bytes.
        idx += 32
        if idx < cutlass.const_expr(self.data_dim // 16):
            src = k_data_slice[page_id, block_offset, (None, idx)]
            cute.copy(
                cp16_atom,
                cute.recast_tensor(src, Uint32),
                s_kdata_slice[(None, idx), stage_id],
            )
        elif idx == cutlass.const_expr(self.data_dim // 16):
            cute.copy(
                cp8_atom,
                k_scale[page_id, block_offset, None],
                s_kscale[None, stage_id],
            )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        k_data: cute.Tensor,
        k_scale: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor | None,
        block_table: cute.Tensor,
        offset: Int32,
    ):
        req_id, worker_id, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        _, num_workers, _ = cute.arch.grid_dim()

        # Prepare smem.
        smem = cutlass.utils.SmemAllocator()
        s_kdata = smem.allocate_tensor(
            Uint32,
            cute.make_layout((self.data_dim // 4, self.num_warps, self.num_stages)),
            byte_alignment=16,
        )[None, warp_id, None]
        s_kscale = smem.allocate_tensor(
            Uint8,
            cute.make_layout((8, self.num_warps, self.num_stages)),
            byte_alignment=8,
        )[None, warp_id, None]

        # Prepare for 16B cp.async, also for BF16 smem loads later.
        k_data_slice = cute.logical_divide(k_data, (None, None, 16))
        s_kdata_16B_slice = cute.logical_divide(s_kdata, (4, None))

        # Load FP8 elems in 8B units, so once dequantized, they are 16B units.
        s_kdata_8B_slice = cute.logical_divide(s_kdata, (2, None))

        # 16B st.global.
        out_slice = cute.logical_divide(out, (None, None, 8))

        cp_op = cute.nvgpu.CopyUniversalOp()
        cp8_atom = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=64)
        cp16_atom = cute.make_copy_atom(cp_op, Uint32, num_bits_per_copy=128)

        seq_len = seq_lens[req_id]
        gather_len = seq_len
        if cutlass.const_expr(gather_lens is not None):
            gather_len = gather_lens[req_id]  # type: ignore[index]
        start_pos = seq_len - gather_len

        # Start prefetch.
        for i in cutlass.range_constexpr(self.num_stages - 1):
            next_pos = (
                start_pos
                + worker_id * self.num_warps
                + warp_id
                + i * num_workers * self.num_warps
            )
            if next_pos < seq_len:
                self.load_g2s(
                    k_data_slice,
                    k_scale,
                    block_table,
                    s_kdata_16B_slice,
                    s_kscale,
                    req_id,
                    next_pos,
                    lane_id,
                    i,
                )
            cute.arch.cp_async_commit_group()
        prefetch_stage = self.num_stages - 1
        compute_stage = 0

        # Main loop.
        for i in range(
            worker_id * self.num_warps + warp_id,
            gather_len,
            num_workers * self.num_warps,
        ):
            pos = start_pos + i

            # Prefetch next stage.
            next_pos = pos + num_workers * self.num_warps * (self.num_stages - 1)
            if next_pos < seq_len:
                self.load_g2s(
                    k_data_slice,
                    k_scale,
                    block_table,
                    s_kdata_16B_slice,
                    s_kscale,
                    req_id,
                    next_pos,
                    lane_id,
                    prefetch_stage,
                )
                prefetch_stage = (prefetch_stage + 1) % self.num_stages
            cute.arch.cp_async_commit_group()

            # Wait for gmem->smem to finish.
            cute.arch.cp_async_wait_group(self.num_stages - 1)
            cute.arch.sync_warp()

            # There are 512 elems per token. As a warp, data0 holds the first
            # 256 elems and data1 holds the second 256 elems, i.e. each thread
            # holds 8 FP8 elems. This keeps the dequantized 8 BF16 elems as
            # contiguous 16B global stores. On Blackwell, this might not be
            # necessary as we have 32B global stores, but doing it this way
            # does not seem to be slower.
            data0 = cute.make_rmem_tensor((2,), Uint32)
            data1 = cute.make_rmem_tensor((2,), Uint32)
            cute.copy(cp8_atom, s_kdata_8B_slice[(None, lane_id), compute_stage], data0)
            cute.copy(
                cp8_atom,
                s_kdata_8B_slice[(None, lane_id + 32), compute_stage],
                data1,
            )

            # Convert to bf16x2 via bit manipulation. FP8 scales are per 64
            # elements. An 8-element chunk advances the scale index by
            # chunk_id * 8 // group_size.
            scale0_u32 = Uint32(s_kscale[lane_id * 8 // self.group_size, compute_stage])
            scale0_bf16x2 = (scale0_u32 << Uint32(23)) | (scale0_u32 << Uint32(7))
            scale1_u32 = Uint32(
                s_kscale[(lane_id + 32) * 8 // self.group_size, compute_stage]
            )
            scale1_bf16x2 = (scale1_u32 << Uint32(23)) | (scale1_u32 << Uint32(7))

            # cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 requires PTX 9.2
            # (CUDA 13.2).
            dequant0 = cute.make_rmem_tensor(4, Uint32)
            dequant1 = cute.make_rmem_tensor(4, Uint32)
            for j in cutlass.range_constexpr(2):
                tmp0 = _fp8x4_to_bf16x4(data0[j])
                tmp1 = _fp8x4_to_bf16x4(data1[j])

                # BF16 multiply is safe because the scales are exact powers of 2.
                dequant0[j * 2] = _bf16x2_mul(tmp0[0], scale0_bf16x2)
                dequant1[j * 2] = _bf16x2_mul(tmp1[0], scale1_bf16x2)
                dequant0[j * 2 + 1] = _bf16x2_mul(tmp0[1], scale0_bf16x2)
                dequant1[j * 2 + 1] = _bf16x2_mul(tmp1[1], scale1_bf16x2)

            # Last 64 elems are BF16 tail, corresponds to dequant1 of last
            # 8 threads. We have 448 FP8 + 64 BF16 -> 28x 16B for FP8 +
            # 8x 16B for BF16.
            if lane_id + 32 >= self.fp8_dim // 8:
                idx = self.fp8_dim // 16 + (lane_id + 32) - self.fp8_dim // 8
                cute.copy(
                    cp16_atom,
                    s_kdata_16B_slice[(None, idx), compute_stage],
                    dequant1,
                )

            # Store two 16B BF16 chunks per lane: first half, then second half.
            dst = out_slice[req_id, offset + i, (None, lane_id)]
            cute.copy(cp16_atom, dequant0, cute.recast_tensor(dst, Uint32))

            dst = out_slice[req_id, offset + i, (None, lane_id + 32)]
            cute.copy(cp16_atom, dequant1, cute.recast_tensor(dst, Uint32))

            compute_stage = (compute_stage + 1) % self.num_stages

    @cache
    @staticmethod
    def compile(
        fp8_dim: int = 448,
        block_size: int = 64,
        has_gather_lens: bool = True,
    ):
        num_reqs = cute.sym_int()
        head_dim = DequantGatherKCacheKernel.head_dim
        head_bytes = fp8_dim + (head_dim - fp8_dim) * 2 + 8

        out = make_fake_tensor(BFloat16, (num_reqs, cute.sym_int(), head_dim), 16)
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (cute.sym_int(), block_size, head_bytes),
            stride=(cute.sym_int64(divisibility=32), head_bytes, 1),
            assumed_align=32,
        )
        seq_lens = make_fake_tensor(Int32, (num_reqs,))
        gather_lens = make_fake_tensor(Int32, (num_reqs,)) if has_gather_lens else None
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
