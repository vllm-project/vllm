# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream

_HC = 4
_K = 320
_BLOCK_SIZE = 32
_OUTPUTS_PER_BLOCK = 2
_VECTOR_WIDTH = 2


class HCSiluUpGateMixKernel:
    """Fused Qwen4Exp decode HC up-projection and gated stream reduction."""

    def __init__(self) -> None:
        self.element_type = cutlass.BFloat16

    @cute.jit
    def __call__(
        self,
        g_lora: cute.Tensor,
        g_weight: cute.Tensor,
        g_x: cute.Tensor,
        g_out: cute.Tensor,
        stream: CUstream,
    ) -> None:
        hidden_size = cute.size(g_out, mode=[1])
        copy_lora = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=_VECTOR_WIDTH * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_weight = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=_VECTOR_WIDTH * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(
            g_lora,
            g_weight,
            g_x,
            g_out,
            hidden_size,
            copy_lora,
            copy_weight,
        ).launch(
            grid=[cute.ceil_div(hidden_size, _OUTPUTS_PER_BLOCK), 1, 1],
            block=[_BLOCK_SIZE, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_lora: cute.Tensor,
        g_weight: cute.Tensor,
        g_x: cute.Tensor,
        g_out: cute.Tensor,
        hidden_size: cutlass.Int32,
        copy_lora: cute.CopyAtom,
        copy_weight: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()

        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (_OUTPUTS_PER_BLOCK, _HC),
                stride=(_HC, 1),
            ),
            cutlass.Float32,
        )
        acc.fill(0.0)

        lora_vec = cute.logical_divide(g_lora, (None, _VECTOR_WIDTH))
        weight_vec = cute.logical_divide(g_weight, (None, _VECTOR_WIDTH))
        lora_tiles = cute.logical_divide(lora_vec, (None, (None, _BLOCK_SIZE)))
        weight_tiles = cute.logical_divide(weight_vec, (None, (None, _BLOCK_SIZE)))
        thread_lora = lora_tiles[None, (None, (tidx, None))]
        lora_regs = cute.make_rmem_tensor(
            cute.make_layout((_VECTOR_WIDTH,), stride=(1,)),
            self.element_type,
        )
        weight_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (_OUTPUTS_PER_BLOCK, _HC, _VECTOR_WIDTH),
                stride=(_HC * _VECTOR_WIDTH, _VECTOR_WIDTH, 1),
            ),
            self.element_type,
        )

        hidden_base = block_idx * _OUTPUTS_PER_BLOCK
        for k_tile in cutlass.range_constexpr(_K // (_BLOCK_SIZE * _VECTOR_WIDTH)):
            cute.copy(copy_lora, thread_lora[0, None, k_tile], lora_regs)
            for output in cutlass.range_constexpr(_OUTPUTS_PER_BLOCK):
                for hc_stream in cutlass.range_constexpr(_HC):
                    weight_row = hc_stream * hidden_size + hidden_base + output
                    thread_weight = weight_tiles[weight_row, (None, (tidx, None))]
                    cute.copy(
                        copy_weight,
                        thread_weight[None, k_tile],
                        weight_regs[output, hc_stream, None],
                    )

            raw_lora = lora_regs.load().to(cutlass.Float32) / _HC
            activated_lora = (raw_lora / (1.0 + cute.exp(-raw_lora, fastmath=True))).to(
                self.element_type
            )
            activated_lora = activated_lora.to(cutlass.Float32)
            weights = weight_regs.load().to(cutlass.Float32)
            for vector_lane in cutlass.range_constexpr(_VECTOR_WIDTH):
                for output in cutlass.range_constexpr(_OUTPUTS_PER_BLOCK):
                    for hc_stream in cutlass.range_constexpr(_HC):
                        acc[output, hc_stream] += (
                            activated_lora[vector_lane]
                            * weights[output, hc_stream, vector_lane]
                        )

        for output in cutlass.range_constexpr(_OUTPUTS_PER_BLOCK):
            for hc_stream in cutlass.range_constexpr(_HC):
                acc[output, hc_stream] = cute.arch.warp_reduction_sum(
                    acc[output, hc_stream]
                )

        if tidx == 0:
            for output in cutlass.range_constexpr(_OUTPUTS_PER_BLOCK):
                hidden_idx = hidden_base + output
                if hidden_idx < hidden_size:
                    mixed = cutlass.Float32(0.0)
                    for hc_stream in cutlass.range_constexpr(_HC):
                        x = g_x[0, hc_stream * hidden_size + hidden_idx].to(
                            cutlass.Float32
                        )
                        mixed += x / (
                            1.0 + cute.exp(-acc[output, hc_stream], fastmath=True)
                        )
                    g_out[0, hidden_idx] = (mixed / _HC).to(self.element_type)
