# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# RoutedExpertFinalize is adapted from finalizeKernel in FlashInfer
# (csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu,
# flashinfer-python 0.7.0.post1), which carries the notice below. Modified by
# the vLLM contributors: written in CuTe DSL, one output value per thread for a
# fixed hidden size, accumulated with fused multiply-adds, with a PDL
# grid-dependency wait.
#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FlashInfer's NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA /
# Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
"""Top-k finalize of the DeepSeek-V4.1 low-batch routed-expert GEMMs in CuTe DSL.

GEMM2 (``routed_expert_gemm_cute.py``) leaves one unweighted BF16 row per
routed slot. Layers whose MoE output is not handed to the fused all-reduce +
mHC (the DSpark draft's, which finalize in the layer) reduce those rows here::

    out[t, h] = BF16( sum_k  w[t, k] * rows[permuted[t, k], h] )

with the FP32 sum accumulated in slot order ``k = 0..top_k-1`` by fused
multiply-adds and slots whose permuted row is -1 skipped. This is the
arithmetic of FlashInfer's TRT-LLM-gen finalize kernel for small batches (one
thread per output value, grid ``hidden / 256 x tokens``), so the output matches
its ``do_finalize=True`` path given the same GEMM2 rows.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

HIDDEN = 5120
TILE = 8  # GEMM2 rows per routed slot, at most
MAX_TOP_K = 8  # routed slots per token, at most
FINALIZE_THREADS = 256  # one thread per output value


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    """``a * b + c`` rounded once (``fma.rn.f32``)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class RoutedExpertFinalize:
    """The finalize kernel.

    PDL: waits for GEMM2's grid before reading; its dependents launch only on
    its completion (the next kernel may not wait).
    """

    @cute.jit
    def __call__(
        self,
        rows_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        weights_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tokens: Int32,
        top_k: Int32,
        stream: cuda.CUstream,
    ):
        """``rows``: BF16 GEMM2 rows [tokens * top_k * 8, 5120];
        ``permuted``: int32 and ``weights``: FP32 [tokens, top_k];
        ``out``: BF16 [tokens, 5120]."""
        slots = cute.make_layout(tokens * top_k)
        self.kernel(
            cute.make_tensor(
                rows_ptr,
                cute.make_layout((tokens * top_k * TILE, HIDDEN), stride=(HIDDEN, 1)),
            ),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(weights_ptr, slots),
            cute.make_tensor(
                out_ptr, cute.make_layout((tokens, HIDDEN), stride=(HIDDEN, 1))
            ),
            top_k,
        ).launch(
            grid=(HIDDEN // FINALIZE_THREADS, tokens, 1),
            block=[FINALIZE_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        rows: cute.Tensor,
        permuted: cute.Tensor,
        weights: cute.Tensor,
        out: cute.Tensor,
        top_k: Int32,
    ):
        """One output value per thread; CTA (column block, token)."""
        cute.arch.griddepcontrol_wait()
        thread, _, _ = cute.arch.thread_idx()
        column, token, _ = cute.arch.block_idx()
        hidden = column * FINALIZE_THREADS + thread
        first = token * top_k
        # Every slot's row and weight, then every row value, so that the loads
        # overlap; the sum then runs in slot order.
        slot_rows = cute.make_rmem_tensor((MAX_TOP_K,), Int32)
        scales = cute.make_rmem_tensor((MAX_TOP_K,), Float32)
        values = cute.make_rmem_tensor((MAX_TOP_K,), Float32)
        for k in cutlass.range_constexpr(MAX_TOP_K):
            slot_rows[k] = Int32(-1)
            scales[k] = Float32(0.0)
            if k < top_k:
                slot_rows[k] = permuted[first + k]
                scales[k] = weights[first + k]
        for k in cutlass.range_constexpr(MAX_TOP_K):
            values[k] = Float32(0.0)
            if slot_rows[k] != -1:
                values[k] = rows[slot_rows[k], hidden].to(Float32)
        data = Float32(0.0)
        for k in cutlass.range_constexpr(MAX_TOP_K):
            if slot_rows[k] != -1:
                data = fma_rn(scales[k], values[k], data)
        out[token, hidden] = data.to(BFloat16)
