# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Fuse K-axis mixing, 1/sqrt(K) scaling, and inverse signs after HadaCore."""

import math

from vllm.tilelang_utils import T, tilelang


@tilelang.jit
def make_composite_hadamard_epilogue(k: int, p: int, inverse: bool, dtype: str):
    hidden_size = k * p
    num_tokens = T.dynamic("num_tokens")
    threads = 256
    scale = 1.0 / math.sqrt(k)

    @T.prim_func
    def composite_hadamard_epilogue(
        x: T.Tensor[(num_tokens, hidden_size), dtype],
        sign: T.Tensor[(hidden_size,), T.int8],
        hk: T.Tensor[(k, k), T.float32],
        output: T.Tensor[(num_tokens, hidden_size), dtype],
    ):
        with T.Kernel(num_tokens, T.ceildiv(p, threads), threads=threads) as (
            token,
            tile,
        ):
            thread = T.get_thread_binding()
            position = tile * threads + thread
            if position < p:
                for output_block in T.unroll(k):
                    accumulator = T.alloc_var(T.float32)
                    accumulator = 0.0
                    for input_block in T.unroll(k):
                        if inverse:
                            coefficient = hk[output_block, input_block]
                        else:
                            coefficient = hk[input_block, output_block]
                        accumulator += (
                            T.cast(x[token, input_block * p + position], T.float32)
                            * coefficient
                        )
                    accumulator *= scale
                    if inverse:
                        accumulator *= T.cast(
                            sign[output_block * p + position], T.float32
                        )
                    output[token, output_block * p + position] = T.cast(
                        accumulator, dtype
                    )

    return composite_hadamard_epilogue
