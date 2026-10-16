# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Fused TileLang Hadamard implementation for NVIDIA warp32 GPUs."""

import math
from collections.abc import Callable

import torch

from vllm.tilelang_utils import T, tilelang

from .autotune import (
    get_composite_hadamard_configs,
    supply_composite_hadamard_inputs,
)
from .utils import (
    GEMM_HADAMARD_ORDER,
    is_pow2,
    resolve_hadamard_dim,
)


@tilelang.autotune(
    configs=get_composite_hadamard_configs,
    supply_prog=supply_composite_hadamard_inputs,
    skip_check=True,
)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def make_composite_hadamard_kernel(
    hidden_size: int,
    base_order: int,
    inverse: bool,
    dtype: str = "bfloat16",
    num_threads: int = 256,
    mma_rows: int = 32,
    gemm_policy: str = "square",
):
    """Build one CTA/token: dense H_B -> remaining H_P -> H_K."""
    if num_threads <= 0 or num_threads % 32:
        raise ValueError("num_threads must be a positive multiple of 32")
    if hidden_size % base_order:
        raise ValueError("hidden_size must be divisible by base_order")
    width = hidden_size // base_order
    if not is_pow2(width):
        raise ValueError("P must be a power of two")

    use_gemm = width >= 32
    gemm_order = GEMM_HADAMARD_ORDER

    log_width = int(math.log2(width))
    first_stage = int(math.log2(gemm_order)) if use_gemm else 0
    load_iterations = (hidden_size + num_threads - 1) // num_threads
    pair_count = hidden_size // 2
    pair_iterations = (pair_count + num_threads - 1) // num_threads
    pairs_per_block = width // 2
    scale = 1.0 / math.sqrt(hidden_size)
    if gemm_policy == "square":
        tilelang_gemm_policy = T.GemmWarpPolicy.Square
    elif gemm_policy == "full_col":
        tilelang_gemm_policy = T.GemmWarpPolicy.FullCol
    else:
        raise ValueError(f"unsupported GEMM policy: {gemm_policy!r}")

    if use_gemm:
        rows_per_token = hidden_size // gemm_order
        if mma_rows <= 0 or mma_rows % 16:
            raise ValueError("mma_rows must be a positive multiple of 16")
        gemm_rows = mma_rows
        gemm_batches = (rows_per_token + gemm_rows - 1) // gemm_rows
    else:
        rows_per_token = 0
        gemm_rows = 0
        gemm_batches = 0

    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def composite_hadamard_rotation(
        x: T.Tensor[(num_tokens, hidden_size), dtype],
        sign: T.Tensor[(hidden_size,), T.int8],
        hk: T.Tensor[(base_order, base_order), T.float32],
        h_gemm: T.Tensor[(gemm_order, gemm_order), dtype],
        out: T.Tensor[(num_tokens, hidden_size), dtype],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as token:
            thread = T.get_thread_binding()
            shared = T.alloc_shared((hidden_size,), T.float32)

            # 1. Dense H32 primitive.
            if use_gemm:
                h_gemm_shared = T.alloc_shared((gemm_order, gemm_order), dtype)
                gemm_input = T.alloc_shared((gemm_rows, gemm_order), dtype)
                gemm_output = T.alloc_fragment((gemm_rows, gemm_order), T.float32)
                for row, column in T.Parallel(gemm_order, gemm_order):
                    h_gemm_shared[row, column] = h_gemm[row, column]

                for batch in T.serial(gemm_batches):
                    for row, column in T.Parallel(gemm_rows, gemm_order):
                        global_row = batch * gemm_rows + row
                        if global_row < rows_per_token:
                            index = global_row * gemm_order + column
                            value = T.alloc_var(T.float32)
                            value = T.cast(x[token, index], T.float32)
                            if not inverse:
                                value *= T.cast(sign[index], T.float32)
                            gemm_input[row, column] = T.cast(value, dtype)
                        else:
                            gemm_input[row, column] = T.cast(0.0, dtype)
                    T.sync_threads()

                    T.clear(gemm_output)
                    T.gemm(
                        gemm_input,
                        h_gemm_shared,
                        gemm_output,
                        transpose_B=True,
                        policy=tilelang_gemm_policy,
                    )
                    for row, column in T.Parallel(gemm_rows, gemm_order):
                        global_row = batch * gemm_rows + row
                        if global_row < rows_per_token:
                            index = global_row * gemm_order + column
                            shared[index] = gemm_output[row, column]
                    T.sync_threads()
            else:
                for iteration in T.serial(load_iterations):
                    index = thread + iteration * num_threads
                    if index < hidden_size:
                        value = T.alloc_var(T.float32)
                        value = T.cast(x[token, index], T.float32)
                        if not inverse:
                            value *= T.cast(sign[index], T.float32)
                        shared[index] = value
                T.sync_threads()

            # 2. Remaining H_P butterfly stages.
            for stage_offset in T.unroll(log_width - first_stage):
                stage = first_stage + stage_offset
                stride = 1 << stage
                for iteration in T.serial(pair_iterations):
                    pair_id = thread + iteration * num_threads
                    if pair_id < pair_count:
                        block = pair_id // pairs_per_block
                        local_pair = pair_id - block * pairs_per_block
                        group = local_pair // stride
                        offset = local_pair - group * stride
                        left_index = block * width + group * (2 * stride) + offset
                        right_index = left_index + stride
                        left = shared[left_index]
                        right = shared[right_index]
                        shared[left_index] = left + right
                        shared[right_index] = left - right
                T.sync_threads()

            # 3/4. H_K mixing, normalization, and direct global store.
            for iteration in T.serial(load_iterations):
                index = thread + iteration * num_threads
                if index < hidden_size:
                    result = T.alloc_var(T.float32)
                    if base_order == 1:
                        result = shared[index] * scale
                    else:
                        output_block = index // width
                        position = index - output_block * width
                        accumulator = T.alloc_var(T.float32)
                        accumulator = 0.0
                        for input_block in T.serial(base_order):
                            if inverse:
                                coefficient = hk[output_block, input_block]
                            else:
                                coefficient = hk[input_block, output_block]
                            accumulator += (
                                shared[input_block * width + position] * coefficient
                            )
                        result = accumulator * scale
                    if inverse:
                        result *= T.cast(sign[index], T.float32)
                    out[token, index] = T.cast(result, dtype)

    return composite_hadamard_rotation


_KERNEL_CACHE: dict[tuple[int, int, bool, str], Callable[..., object]] = {}


def composite_hadamard(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
    *,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Run the fused GPU transform and preserve ``x[..., hidden_size]``."""
    if x.ndim < 1 or not x.is_cuda:
        raise ValueError("x must be a GPU tensor with a hidden last dimension")

    hidden_size = x.shape[-1]
    base_order, width = resolve_hadamard_dim(hidden_size, k=k, p=p)
    if sign.shape != (hidden_size,) or sign.dtype != torch.int8:
        raise ValueError("sign must be a [hidden_size] int8 tensor")
    if x.dtype == torch.bfloat16:
        tilelang_dtype = "bfloat16"
    elif x.dtype == torch.float16:
        tilelang_dtype = "float16"
    elif x.dtype == torch.float32:
        tilelang_dtype = "float32"
    else:
        raise ValueError("Hadamard supports bfloat16, float16, and float32")

    if hk.shape != (base_order, base_order):
        raise ValueError(f"hk must have shape ({base_order}, {base_order})")
    if h_gemm.shape != (GEMM_HADAMARD_ORDER, GEMM_HADAMARD_ORDER):
        raise ValueError(
            f"h_gemm must have shape ({GEMM_HADAMARD_ORDER}, {GEMM_HADAMARD_ORDER})"
        )

    key = (
        hidden_size,
        base_order,
        inverse,
        tilelang_dtype,
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = make_composite_hadamard_kernel.compile(
            hidden_size,
            base_order,
            inverse,
            tilelang_dtype,
        )

    input_2d = x.reshape(-1, hidden_size).contiguous()
    output_2d = torch.empty_like(input_2d)
    _KERNEL_CACHE[key](input_2d, sign, hk, h_gemm, output_2d)
    return output_2d.reshape(x.shape)
