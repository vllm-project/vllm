# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang activation kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if TYPE_CHECKING or current_platform.is_cuda_alike():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for TileLang activation kernels but is not "
            "installed. "
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


if tilelang is not None:

    @tilelang.jit
    def silu_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 SiLU-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col].astype(T.float32)
                        silu = (T.sigmoid(gate) * gate).astype(T.bfloat16)
                        y[by, col] = silu * x[by, col + HIDDEN]

        return kernel

    @tilelang.jit
    def mul_and_silu_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 mul-and-SiLU."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col + HIDDEN].astype(T.float32)
                        silu = (T.sigmoid(gate) * gate).astype(T.bfloat16)
                        y[by, col] = x[by, col] * silu

        return kernel

    @tilelang.jit
    def fatrelu_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 FATReLU-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
            threshold: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col]
                        if gate > threshold:
                            y[by, col] = gate * x[by, col + HIDDEN]
                        else:
                            y[by, col] = T.bfloat16(0.0)

        return kernel

    @tilelang.jit
    def silu_and_mul_with_clamp_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 clamped SiLU-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
            limit: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col].astype(T.float32)
                        up = x[by, col + HIDDEN].astype(T.float32)
                        gate_clamped = T.min(gate, limit)
                        up_clamped = T.min(T.max(up, -limit), limit)
                        silu = (T.sigmoid(gate_clamped) * gate_clamped).astype(
                            T.bfloat16
                        )
                        y[by, col] = silu * up_clamped.astype(T.bfloat16)

        return kernel

    @tilelang.jit
    def gelu_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 GELU-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col].astype(T.float32)
                        gelu = (
                            0.5 * gate * (1.0 + T.erf(gate * 0.7071067811865476))
                        ).astype(T.bfloat16)
                        y[by, col] = gelu * x[by, col + HIDDEN]

        return kernel

    @tilelang.jit
    def gelu_tanh_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 tanh-approx GELU-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col].astype(T.float32)
                        inner = gate + 0.044715 * gate * gate * gate
                        gelu = (
                            0.5 * gate * (1.0 + T.tanh(0.7978845608028654 * inner))
                        ).astype(T.bfloat16)
                        y[by, col] = gelu * x[by, col + HIDDEN]

        return kernel

    @tilelang.jit
    def swigluoai_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 GPT-OSS SwiGLU OAI."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
            alpha: T.float32,
            limit: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col * 2].astype(T.float32)
                        up = x[by, col * 2 + 1].astype(T.float32)
                        gate_clamped = T.min(gate, limit)
                        up_clamped = T.min(T.max(up, -limit), limit)
                        glu = gate_clamped * T.sigmoid(gate_clamped * alpha)
                        y[by, col] = ((up_clamped + 1.0) * glu).astype(T.bfloat16)

        return kernel

    @tilelang.jit
    def swiglustep_and_mul_tilelang_kernel(
        BLOCK_SIZE: int = 1024,
        threads: int = 128,
    ):
        """Build a TileLang kernel for bf16 SwiGLU-step-and-mul."""
        NUM_TOKENS = T.dynamic("NUM_TOKENS")
        HIDDEN = T.dynamic("HIDDEN")

        @T.prim_func
        def kernel(
            x: T.Tensor[[NUM_TOKENS, HIDDEN * 2], T.bfloat16],
            y: T.Tensor[[NUM_TOKENS, HIDDEN], T.bfloat16],
            limit: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(HIDDEN, BLOCK_SIZE), NUM_TOKENS, threads=threads
            ) as (bx, by):
                col_start = bx * BLOCK_SIZE

                for i in T.Parallel(BLOCK_SIZE):
                    col = col_start + i
                    if col < HIDDEN:
                        gate = x[by, col].astype(T.float32)
                        up = x[by, col + HIDDEN].astype(T.float32)
                        gate_silu = T.sigmoid(gate) * gate
                        gate_clamped = T.min(gate_silu, limit)
                        up_clamped = T.min(T.max(up, -limit), limit)
                        y[by, col] = (gate_clamped * up_clamped).astype(T.bfloat16)

        return kernel

else:

    def _unavailable_tilelang_kernel(*args: object, **kwargs: object):
        raise RuntimeError(
            "TileLang activation kernels are only available on CUDA-like "
            "platforms with tilelang installed."
        )

    silu_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
    mul_and_silu_tilelang_kernel = _unavailable_tilelang_kernel
    fatrelu_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
    silu_and_mul_with_clamp_tilelang_kernel = _unavailable_tilelang_kernel
    gelu_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
    gelu_tanh_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
    swigluoai_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
    swiglustep_and_mul_tilelang_kernel = _unavailable_tilelang_kernel
