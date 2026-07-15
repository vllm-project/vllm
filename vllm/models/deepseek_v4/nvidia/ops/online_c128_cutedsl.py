# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels for DeepSeek V4 online C128 state updates."""

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from quack.compile_utils import make_fake_tensor

RCP_LN2 = 1.4426950408889634


class OnlineC128MergeKernel:
    """Merge planned token segments into per-request running state."""

    elems_per_lane = 2

    def __init__(self, head_size: int, compress_ratio: int):
        self.head_dim = head_size
        self.compress_ratio = compress_ratio
        self.tb_size = head_size // self.elems_per_lane

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        score: cute.Tensor,
        ape: cute.Tensor,
        positions: cute.Tensor,
        run_state: cute.Tensor,
        segments: cute.Tensor,
        compressed_kv: cute.Tensor,
        stream: CUstream,
    ):
        grid = (segments.shape[0], 1, 1)
        self.kernel(
            kv,
            score,
            ape,
            positions,
            run_state,
            segments,
            compressed_kv,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        kv: cute.Tensor,
        score: cute.Tensor,
        ape: cute.Tensor,
        positions: cute.Tensor,
        run_state: cute.Tensor,
        segments: cute.Tensor,
        compressed_kv: cute.Tensor,
    ):
        seg_id, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        col0 = tid * self.elems_per_lane

        row_base = segments[seg_id, 0]
        num_rows = segments[seg_id, 1]
        read_row = segments[seg_id, 2]
        emit_token = segments[seg_id, 3]
        write_row = segments[seg_id, 4]

        max_off = Int64(0)
        sum_off = Int64(self.head_dim)
        wsum_off = Int64(2 * self.head_dim)

        run_state_w = run_state.stride[0]
        kv_w = kv.stride[0]
        score_w = score.stride[0]
        ape_w = ape.stride[0]
        compressed_w = compressed_kv.stride[0]

        local_max = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
        local_sum = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
        local_product = cute.make_rmem_tensor((self.elems_per_lane,), Float32)

        for e in cutlass.range_constexpr(self.elems_per_lane):
            local_max[e] = -Float32.inf
            local_sum[e] = Float32(0.0)
            local_product[e] = Float32(0.0)

        if read_row >= Int32(0):
            base = read_row.to(Int64) * run_state_w + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                local_max[e] = run_state.iterator[base + max_off + Int64(e)]
                local_sum[e] = run_state.iterator[base + sum_off + Int64(e)]
                local_product[e] = run_state.iterator[base + wsum_off + Int64(e)]

        for r in cutlass.range(num_rows, unroll=1):
            token = row_base + r
            tok64 = token.to(Int64)
            position = positions[tok64]
            ape_row = (position % Int64(self.compress_ratio)) * ape_w
            kv_base = tok64 * kv_w + col0.to(Int64)
            score_base = tok64 * score_w + col0.to(Int64)
            ape_base = ape_row + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                kv_e = kv.iterator[kv_base + Int64(e)].to(Float32)
                score_e = score.iterator[score_base + Int64(e)].to(Float32)
                ape_e = ape.iterator[ape_base + Int64(e)]
                score_e = score_e + ape_e
                new_max = cute.arch.fmax(local_max[e], score_e)
                old_scale = cute.math.exp2(
                    (local_max[e] - new_max) * Float32(RCP_LN2), fastmath=True
                )
                new_scale = cute.math.exp2(
                    (score_e - new_max) * Float32(RCP_LN2), fastmath=True
                )
                local_sum[e] = local_sum[e] * old_scale + new_scale
                local_product[e] = local_product[e] * old_scale + kv_e * new_scale
                local_max[e] = new_max

        if emit_token >= Int32(0):
            ebase = emit_token.to(Int64) * compressed_w + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                compressed_kv.iterator[ebase + Int64(e)] = (
                    local_product[e] / local_sum[e]
                )

        if write_row >= Int32(0):
            wbase = write_row.to(Int64) * run_state_w + col0.to(Int64)
            for e in cutlass.range_constexpr(self.elems_per_lane):
                run_state.iterator[wbase + max_off + Int64(e)] = local_max[e]
                run_state.iterator[wbase + sum_off + Int64(e)] = local_sum[e]
                run_state.iterator[wbase + wsum_off + Int64(e)] = local_product[e]

    @cache
    @staticmethod
    def compile(head_size: int = 512, compress_ratio: int = 128):
        if head_size % OnlineC128MergeKernel.elems_per_lane != 0:
            raise ValueError("head_size must be even.")
        num_tokens = cute.sym_int()
        num_rows = cute.sym_int()
        num_segments = cute.sym_int()
        num_output_tokens = cute.sym_int()

        kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        score = cute.runtime.make_fake_tensor(
            Float32,
            (num_tokens, head_size),
            stride=(cute.sym_int64(divisibility=4), 1),
            assumed_align=16,
        )
        ape = cute.runtime.make_fake_tensor(
            Float32,
            (compress_ratio, head_size),
            stride=(head_size, 1),
            assumed_align=16,
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=8)
        run_state = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, 3 * head_size),
            stride=(cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        segments = make_fake_tensor(Int32, (num_segments, 5), divisibility=1)
        compressed_kv = cute.runtime.make_fake_tensor(
            Float32,
            (num_output_tokens, head_size),
            stride=(head_size, 1),
            assumed_align=4,
        )
        kernel = OnlineC128MergeKernel(head_size, compress_ratio)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            kv,
            score,
            ape,
            positions,
            run_state,
            segments,
            compressed_kv,
            stream,
            options="--enable-tvm-ffi",
        )


def online_c128_merge(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    run_state: torch.Tensor,
    segments: torch.Tensor,
    compressed_kv: torch.Tensor,
    compress_ratio: int = 128,
) -> None:
    if segments.numel() == 0:
        return
    head_size = compressed_kv.shape[-1]
    if kv.dtype != torch.float32 or score.dtype != torch.float32:
        raise ValueError(
            "online_c128_merge expects fp32 kv/score, got "
            f"{kv.dtype} / {score.dtype}."
        )
    compiled = OnlineC128MergeKernel.compile(
        head_size=head_size,
        compress_ratio=compress_ratio,
    )
    compiled(kv, score, ape, positions, run_state, segments, compressed_kv)
