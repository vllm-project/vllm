# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured skinny GEMM plans for unquantized BF16 linear layers.

Plans are keyed by device capability, local `(N, K)` weight shape and token
count. A layer whose shape is listed runs the planned config for that token
count; other token counts use `F.linear`. Entries are kept only where the
skinny GEMM beats cuBLAS by more than 5% under CUDA graph replay.

CuTe DSL configs cover the dot-product regime (a few tokens). Triton configs
cover tens to hundreds of tokens, where the cuBLAS heuristic on some GPUs
picks kernels that reach only 60-85% of memory bandwidth.
"""

import functools

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.kernels.linear.triton_skinny_gemm import (
    TritonSkinnyGemmConfig,
)
from vllm.platforms import current_platform

SkinnyGemmPlan = dict[int, SkinnyGemmConfig | TritonSkinnyGemmConfig]

# DGX Spark (GB10), measured on GLM-5.3-Flash local shapes at TP=2.
SM121_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # Shared-expert fused gate/up and MLA fused q_a/kv_a projections.
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=4096),
    },
    # Shared-expert down projection.
    (4096, 1024): {
        1: SkinnyGemmConfig(1, 64, 1, vector_width=2, static_k=1024),
        2: SkinnyGemmConfig(2, 64, 1, vector_width=2, static_k=1024),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=1024),
    },
    # KDA output projection.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=4, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=2, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 1, vector_width=2, static_k=4096),
    },
    # MLA output projection.
    (4096, 8192): {
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, vector_width=2, static_k=8192),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, static_k=8192),
    },
    # MLA q_b projection.
    (8192, 1536): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=2, vector_width=2, static_k=1536),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=2, vector_width=2, static_k=1536),
    },
    # KDA fused QKV/beta/f_a/g_a input projection.
    (12576, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=2, vector_width=2),
        2: SkinnyGemmConfig(2, 256, 1),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=4096),
    },
    # LM head.
    (77440, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=2),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2),
    },
}

_C = TritonSkinnyGemmConfig

# Triton configs are (block_m, block_n, block_k, num_warps, num_stages), listed
# only at the CUDA graph sizes they were measured at. GPUs that share a compute
# capability share a table, so an entry is kept only if it beats cuBLAS by more
# than 5% on every measured GPU of that capability.

# A100-SXM4-40GB and A100 80GB PCIe.
SM80_TRITON_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # Llama-3.1-8B TP=1 o; Qwen3-8B TP=1 o.
    (4096, 4096): {
        16: _C(16, 64, 128, 4, 4),
        **dict.fromkeys(range(72, 97, 8), _C(32, 128, 128, 4, 4)),
        192: _C(64, 128, 128, 4, 3),
    },
    # Qwen3-8B TP=1 down.
    (4096, 12288): {
        8: _C(16, 64, 128, 4, 5),
        24: _C(16, 128, 128, 4, 4),
        32: _C(32, 64, 128, 8, 5),
        176: _C(64, 128, 128, 4, 3),
    },
    # Llama-3.1-8B TP=1 down.
    (4096, 14336): {
        **dict.fromkeys(range(8, 17, 8), _C(16, 64, 128, 4, 5)),
        24: _C(16, 128, 128, 4, 4),
        32: _C(32, 64, 128, 4, 5),
    },
    # Qwen3-14B TP=1 o.
    (5120, 5120): {
        24: _C(16, 128, 128, 4, 4),
        32: _C(32, 64, 128, 4, 4),
        112: _C(64, 128, 128, 4, 3),
        128: _C(64, 128, 128, 4, 3),
    },
    # Qwen3-14B TP=1 down.
    (5120, 17408): {
        8: _C(16, 64, 128, 4, 4),
        16: _C(16, 64, 128, 4, 5),
        24: _C(16, 128, 128, 8, 4),
        32: _C(32, 64, 128, 4, 5),
        **dict.fromkeys(range(40, 57, 8), _C(32, 128, 128, 4, 4)),
        64: _C(64, 64, 128, 4, 4),
    },
    # Llama-3.1-8B TP=1 qkv; Qwen3-8B TP=1 qkv.
    (6144, 4096): {
        56: _C(32, 128, 128, 4, 4),
        **dict.fromkeys(range(72, 129, 8), _C(64, 128, 128, 4, 3)),
    },
    # Qwen3-14B TP=1 qkv.
    (7168, 5120): {
        **dict.fromkeys(range(8, 17, 8), _C(16, 128, 128, 4, 3)),
        24: _C(16, 64, 128, 4, 3),
        32: _C(32, 128, 128, 4, 4),
        **dict.fromkeys(range(72, 97, 8), _C(32, 128, 128, 4, 3)),
    },
    # Qwen3-8B TP=1 gate_up.
    (24576, 4096): {
        8: _C(16, 128, 128, 4, 3),
        32: _C(32, 128, 128, 4, 3),
    },
    # Llama-3.1-8B TP=1 gate_up.
    (28672, 4096): {
        **dict.fromkeys(range(8, 17, 8), _C(16, 64, 128, 4, 2)),
        24: _C(16, 64, 128, 4, 3),
        32: _C(32, 64, 128, 4, 2),
        48: _C(32, 64, 128, 4, 3),
        56: _C(32, 64, 128, 8, 3),
        **dict.fromkeys(range(64, 81, 8), _C(64, 128, 64, 4, 3)),
        88: _C(64, 128, 64, 4, 4),
        96: _C(64, 128, 128, 4, 3),
        104: _C(64, 128, 64, 4, 4),
        **dict.fromkeys(range(112, 129, 8), _C(64, 128, 128, 4, 3)),
        **dict.fromkeys(range(136, 193, 8), _C(64, 128, 64, 4, 3)),
        224: _C(128, 64, 64, 4, 3),
    },
    # Qwen3-14B TP=1 gate_up.
    (34816, 5120): {
        8: _C(16, 64, 128, 4, 4),
        24: _C(16, 128, 128, 4, 3),
        32: _C(32, 128, 128, 4, 4),
        **dict.fromkeys(range(40, 57, 8), _C(32, 128, 128, 4, 3)),
        64: _C(64, 128, 128, 4, 3),
        **dict.fromkeys(range(72, 121, 8), _C(64, 128, 64, 4, 4)),
        128: _C(128, 128, 64, 8, 5),
        **dict.fromkeys(range(136, 169, 8), _C(64, 128, 64, 4, 4)),
        176: _C(64, 256, 64, 8, 4),
        **dict.fromkeys(range(184, 193, 8), _C(64, 256, 64, 8, 3)),
    },
    # Llama-3.1-8B TP=1 lm_head.
    (128256, 4096): {
        8: _C(16, 128, 128, 4, 4),
        **dict.fromkeys(range(136, 145, 8), _C(64, 128, 64, 4, 3)),
    },
    # Qwen3-8B TP=1 lm_head.
    (151936, 4096): {
        8: _C(16, 128, 128, 4, 4),
        32: _C(32, 256, 128, 4, 2),
    },
    # Qwen3-14B TP=1 lm_head.
    (151936, 5120): {
        **dict.fromkeys(range(8, 17, 8), _C(16, 128, 128, 4, 4)),
        32: _C(32, 256, 128, 4, 2),
    },
}

# L40S and L4.
SM89_TRITON_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # Llama-3.1-8B TP=1 o; Qwen3-8B TP=1 o.
    (4096, 4096): {
        48: _C(32, 32, 128, 4, 5),
        64: _C(32, 32, 128, 4, 5),
        72: _C(32, 64, 128, 4, 3),
        80: _C(32, 32, 128, 4, 5),
        88: _C(32, 128, 128, 4, 3),
        96: _C(32, 64, 128, 4, 3),
        **dict.fromkeys(range(104, 129, 8), _C(64, 64, 128, 4, 2)),
        144: _C(64, 128, 128, 8, 2),
        192: _C(64, 128, 128, 8, 2),
    },
    # Qwen3-8B TP=1 down.
    (4096, 12288): {
        **dict.fromkeys(range(8, 17, 8), _C(16, 16, 128, 4, 5)),
        **dict.fromkeys(range(40, 49, 8), _C(32, 16, 128, 4, 5)),
        **dict.fromkeys(range(56, 73, 8), _C(32, 32, 128, 4, 5)),
        80: _C(64, 32, 128, 4, 4),
        **dict.fromkeys(range(88, 97, 8), _C(32, 32, 128, 4, 5)),
        104: _C(64, 64, 128, 8, 4),
        120: _C(64, 64, 128, 4, 4),
        128: _C(32, 64, 128, 4, 5),
        **dict.fromkeys(range(160, 169, 8), _C(64, 128, 128, 4, 3)),
        176: _C(64, 128, 128, 4, 2),
    },
    # Llama-3.1-8B TP=1 down.
    (4096, 14336): {
        **dict.fromkeys(range(40, 65, 8), _C(32, 32, 128, 4, 5)),
        **dict.fromkeys(range(80, 89, 8), _C(32, 128, 128, 8, 2)),
        **dict.fromkeys(range(104, 113, 8), _C(64, 64, 128, 4, 2)),
        128: _C(64, 64, 128, 4, 2),
        144: _C(64, 128, 128, 8, 3),
        152: _C(64, 128, 128, 4, 2),
        **dict.fromkeys(range(168, 177, 8), _C(64, 128, 128, 4, 2)),
        192: _C(64, 128, 128, 4, 2),
    },
    # Qwen3-32B TP=2 o.
    (5120, 4096): {
        **dict.fromkeys(range(40, 49, 8), _C(16, 64, 128, 4, 4)),
        72: _C(16, 64, 128, 4, 5),
        80: _C(64, 64, 128, 4, 2),
        88: _C(64, 128, 128, 8, 3),
        **dict.fromkeys(range(96, 121, 8), _C(64, 128, 128, 8, 2)),
        128: _C(128, 64, 128, 4, 2),
    },
    # Qwen3-14B TP=1 o; Qwen3-32B TP=2 qkv.
    (5120, 5120): {
        48: _C(32, 32, 128, 4, 5),
        **dict.fromkeys(range(72, 97, 8), _C(64, 128, 128, 8, 2)),
        104: _C(64, 64, 128, 4, 2),
        112: _C(64, 128, 128, 8, 2),
        120: _C(64, 64, 128, 4, 2),
        128: _C(64, 128, 128, 8, 2),
        144: _C(32, 64, 64, 4, 3),
        152: _C(64, 64, 64, 4, 5),
        160: _C(64, 64, 128, 8, 4),
        168: _C(64, 64, 64, 4, 5),
        176: _C(64, 64, 128, 4, 3),
        192: _C(64, 128, 128, 8, 3),
        208: _C(64, 128, 64, 4, 3),
    },
    # Qwen3-32B TP=2 down.
    (5120, 12800): {
        24: _C(16, 16, 128, 4, 5),
        **dict.fromkeys(range(40, 65, 8), _C(32, 32, 128, 4, 5)),
        72: _C(16, 64, 128, 4, 5),
        **dict.fromkeys(range(80, 113, 8), _C(64, 128, 128, 8, 2)),
        120: _C(64, 128, 128, 8, 3),
        128: _C(128, 64, 128, 4, 3),
        136: _C(32, 64, 128, 4, 4),
        144: _C(32, 64, 128, 8, 4),
        152: _C(32, 64, 128, 4, 5),
    },
    # Qwen3-14B TP=1 down.
    (5120, 17408): {
        24: _C(16, 16, 128, 4, 5),
        **dict.fromkeys(range(40, 65, 8), _C(32, 32, 128, 4, 5)),
        72: _C(64, 128, 128, 8, 2),
        80: _C(16, 64, 128, 4, 5),
        **dict.fromkeys(range(88, 97, 8), _C(64, 128, 128, 4, 2)),
        104: _C(64, 128, 128, 8, 2),
        112: _C(64, 128, 128, 4, 2),
        120: _C(64, 128, 128, 8, 2),
        128: _C(128, 64, 128, 4, 3),
        **dict.fromkeys(range(136, 145, 8), _C(64, 64, 128, 4, 4)),
        152: _C(32, 128, 128, 8, 2),
        **dict.fromkeys(range(168, 177, 8), _C(64, 64, 128, 4, 4)),
        184: _C(64, 64, 128, 4, 3),
    },
    # Llama-3.1-8B TP=1 qkv; Qwen3-8B TP=1 qkv.
    (6144, 4096): {
        **dict.fromkeys(range(40, 65, 8), _C(32, 32, 128, 4, 5)),
        72: _C(64, 64, 128, 8, 2),
        80: _C(64, 32, 128, 4, 4),
        88: _C(64, 64, 128, 8, 2),
        96: _C(64, 32, 128, 4, 4),
        104: _C(64, 64, 128, 8, 2),
        112: _C(32, 128, 128, 4, 2),
        **dict.fromkeys(range(120, 129, 8), _C(64, 64, 128, 8, 2)),
        168: _C(64, 64, 128, 4, 2),
        **dict.fromkeys(range(184, 193, 8), _C(64, 64, 128, 4, 2)),
        200: _C(128, 64, 128, 4, 2),
    },
    # Qwen3-14B TP=1 qkv.
    (7168, 5120): {
        24: _C(16, 64, 128, 4, 4),
        **dict.fromkeys(range(40, 49, 8), _C(16, 64, 128, 4, 4)),
        56: _C(32, 64, 128, 8, 4),
        64: _C(64, 32, 128, 4, 4),
        72: _C(64, 32, 128, 4, 3),
        80: _C(64, 64, 128, 8, 3),
        **dict.fromkeys(range(88, 105, 8), _C(64, 64, 128, 4, 3)),
        112: _C(64, 64, 128, 8, 3),
        120: _C(64, 64, 128, 4, 3),
        128: _C(64, 64, 128, 8, 3),
        136: _C(64, 64, 64, 4, 3),
        144: _C(32, 128, 64, 4, 5),
        160: _C(64, 64, 128, 4, 2),
        184: _C(64, 64, 128, 4, 2),
        192: _C(128, 64, 128, 4, 2),
    },
    # Qwen3-8B TP=1 gate_up.
    (24576, 4096): {
        **dict.fromkeys(range(24, 41, 8), _C(16, 64, 128, 4, 5)),
        **dict.fromkeys(range(48, 57, 8), _C(32, 16, 128, 4, 5)),
        64: _C(32, 32, 128, 4, 5),
        72: _C(16, 64, 128, 4, 5),
        80: _C(32, 32, 128, 4, 5),
        88: _C(32, 128, 128, 4, 3),
        96: _C(64, 128, 128, 8, 3),
        104: _C(64, 128, 128, 4, 3),
        112: _C(64, 64, 128, 8, 2),
        120: _C(64, 128, 128, 8, 3),
        128: _C(128, 32, 128, 4, 2),
        136: _C(64, 128, 128, 4, 3),
        144: _C(64, 64, 128, 4, 2),
        152: _C(64, 64, 128, 8, 3),
        **dict.fromkeys(range(160, 169, 8), _C(64, 128, 128, 4, 3)),
        176: _C(64, 64, 128, 4, 2),
        200: _C(64, 128, 128, 4, 2),
        208: _C(64, 128, 128, 8, 2),
        **dict.fromkeys(range(216, 225, 8), _C(128, 64, 128, 4, 2)),
        232: _C(64, 128, 128, 4, 3),
        240: _C(64, 128, 128, 8, 2),
        248: _C(128, 128, 64, 8, 3),
        256: _C(64, 128, 128, 4, 2),
    },
    # Qwen3-32B TP=2 gate_up.
    (25600, 5120): {
        24: _C(16, 16, 128, 4, 5),
        32: _C(32, 32, 128, 4, 5),
        **dict.fromkeys(range(40, 49, 8), _C(32, 16, 128, 4, 5)),
        **dict.fromkeys(range(56, 65, 8), _C(32, 32, 128, 4, 5)),
        **dict.fromkeys(range(72, 89, 8), _C(64, 64, 128, 8, 2)),
        96: _C(64, 64, 128, 4, 3),
        **dict.fromkeys(range(104, 121, 8), _C(64, 64, 128, 8, 2)),
        128: _C(128, 64, 128, 4, 3),
        **dict.fromkeys(range(136, 169, 8), _C(64, 64, 128, 4, 2)),
        176: _C(128, 64, 128, 4, 3),
        184: _C(64, 64, 128, 4, 2),
        192: _C(128, 64, 128, 4, 3),
        200: _C(128, 64, 128, 4, 2),
        208: _C(128, 64, 128, 8, 2),
        216: _C(64, 128, 128, 4, 3),
        **dict.fromkeys(range(224, 233, 8), _C(128, 64, 128, 4, 3)),
        240: _C(128, 128, 64, 8, 3),
        248: _C(64, 128, 128, 8, 2),
        256: _C(128, 64, 128, 4, 3),
    },
    # Llama-3.1-8B TP=1 gate_up.
    (28672, 4096): {
        **dict.fromkeys(range(24, 33, 8), _C(16, 16, 128, 4, 5)),
        **dict.fromkeys(range(40, 49, 8), _C(32, 32, 128, 4, 5)),
        56: _C(16, 32, 128, 4, 5),
        64: _C(32, 32, 128, 4, 5),
        72: _C(32, 64, 128, 4, 4),
        **dict.fromkeys(range(80, 89, 8), _C(32, 64, 128, 8, 4)),
        96: _C(32, 64, 128, 4, 4),
        104: _C(64, 64, 128, 4, 3),
        112: _C(64, 64, 128, 8, 3),
        120: _C(32, 64, 128, 4, 3),
        128: _C(128, 64, 128, 4, 2),
        136: _C(64, 128, 128, 4, 3),
        152: _C(64, 64, 128, 8, 3),
        **dict.fromkeys(range(160, 177, 8), _C(64, 128, 128, 4, 3)),
        184: _C(64, 128, 64, 8, 4),
        192: _C(64, 64, 64, 4, 4),
        208: _C(128, 64, 128, 4, 3),
    },
    # Qwen3-14B TP=1 gate_up.
    (34816, 5120): {
        8: _C(16, 16, 128, 4, 5),
        24: _C(16, 16, 128, 4, 5),
        **dict.fromkeys(range(32, 65, 8), _C(32, 32, 128, 4, 5)),
        72: _C(64, 64, 128, 4, 3),
        80: _C(32, 128, 128, 4, 3),
        **dict.fromkeys(range(88, 121, 8), _C(64, 64, 128, 8, 3)),
        128: _C(64, 64, 128, 4, 3),
        **dict.fromkeys(range(136, 145, 8), _C(64, 64, 128, 4, 2)),
        152: _C(64, 128, 128, 4, 3),
        160: _C(64, 128, 128, 8, 3),
        168: _C(64, 128, 128, 4, 3),
        176: _C(64, 128, 128, 8, 3),
        184: _C(64, 128, 128, 4, 3),
        192: _C(64, 128, 128, 8, 2),
        200: _C(64, 128, 128, 8, 3),
        208: _C(64, 128, 128, 4, 2),
        216: _C(128, 64, 128, 8, 2),
        **dict.fromkeys(range(224, 233, 8), _C(64, 128, 128, 8, 3)),
    },
    # Qwen3-32B TP=2 lm_head.
    (75968, 5120): {
        8: _C(16, 16, 128, 4, 5),
        24: _C(16, 64, 128, 4, 5),
        32: _C(32, 32, 128, 4, 5),
        40: _C(16, 32, 128, 4, 5),
        **dict.fromkeys(range(48, 65, 8), _C(32, 32, 128, 4, 5)),
        **dict.fromkeys(range(72, 81, 8), _C(64, 64, 128, 8, 3)),
        88: _C(64, 128, 128, 8, 3),
        96: _C(32, 32, 128, 4, 5),
        104: _C(64, 32, 128, 4, 4),
        **dict.fromkeys(range(112, 121, 8), _C(64, 128, 128, 8, 3)),
        128: _C(32, 64, 128, 4, 3),
        136: _C(128, 64, 128, 8, 3),
        144: _C(64, 64, 128, 4, 3),
        152: _C(128, 64, 128, 4, 3),
        **dict.fromkeys(range(160, 169, 8), _C(64, 128, 128, 4, 3)),
        176: _C(64, 128, 128, 8, 3),
        184: _C(64, 128, 128, 4, 3),
        192: _C(128, 64, 128, 4, 3),
        200: _C(128, 64, 128, 8, 3),
        **dict.fromkeys(range(208, 233, 8), _C(128, 64, 128, 4, 3)),
        240: _C(64, 128, 128, 4, 3),
        **dict.fromkeys(range(248, 257, 8), _C(128, 64, 128, 4, 3)),
    },
    # Llama-3.1-8B TP=1 lm_head.
    (128256, 4096): {
        8: _C(16, 16, 128, 4, 5),
        24: _C(16, 32, 128, 4, 5),
        **dict.fromkeys(range(32, 65, 8), _C(32, 32, 128, 4, 5)),
        72: _C(32, 64, 128, 8, 4),
        80: _C(32, 64, 128, 4, 4),
        88: _C(32, 128, 128, 8, 3),
        96: _C(32, 64, 128, 4, 4),
        **dict.fromkeys(range(104, 121, 8), _C(64, 128, 128, 8, 3)),
        **dict.fromkeys(range(128, 169, 8), _C(64, 128, 128, 4, 3)),
        176: _C(64, 128, 128, 8, 3),
        **dict.fromkeys(range(184, 193, 8), _C(64, 128, 128, 4, 3)),
        200: _C(128, 64, 128, 4, 2),
        208: _C(64, 128, 128, 4, 3),
        **dict.fromkeys(range(216, 225, 8), _C(128, 64, 128, 4, 2)),
        232: _C(128, 64, 64, 8, 4),
        240: _C(128, 64, 128, 8, 2),
        248: _C(128, 128, 64, 8, 3),
        256: _C(128, 128, 64, 8, 4),
    },
    # Qwen3-8B TP=1 lm_head.
    (151936, 4096): {
        24: _C(16, 64, 128, 4, 5),
        **dict.fromkeys(range(32, 41, 8), _C(32, 32, 128, 4, 5)),
        48: _C(16, 64, 128, 4, 5),
        56: _C(32, 64, 128, 4, 4),
        64: _C(64, 16, 128, 4, 4),
        72: _C(64, 128, 128, 8, 3),
        80: _C(32, 32, 128, 4, 4),
        88: _C(64, 128, 128, 8, 3),
        96: _C(32, 128, 128, 8, 3),
        104: _C(64, 64, 128, 4, 3),
        **dict.fromkeys(range(112, 121, 8), _C(64, 128, 128, 4, 3)),
        128: _C(64, 128, 128, 4, 2),
        **dict.fromkeys(range(136, 153, 8), _C(64, 128, 128, 8, 3)),
        160: _C(64, 256, 128, 8, 2),
        **dict.fromkeys(range(168, 177, 8), _C(64, 128, 128, 8, 3)),
        184: _C(64, 128, 128, 4, 3),
        192: _C(64, 128, 128, 8, 3),
        **dict.fromkeys(range(208, 217, 8), _C(128, 64, 128, 4, 2)),
        232: _C(128, 256, 64, 8, 3),
        240: _C(128, 64, 128, 4, 2),
        248: _C(128, 128, 64, 8, 4),
        256: _C(128, 256, 64, 8, 3),
    },
    # Qwen3-14B TP=1 lm_head.
    (151936, 5120): {
        8: _C(16, 16, 128, 4, 5),
        24: _C(16, 32, 128, 4, 5),
        **dict.fromkeys(range(32, 49, 8), _C(32, 32, 128, 4, 5)),
        56: _C(32, 256, 128, 4, 2),
        64: _C(64, 128, 128, 8, 2),
        72: _C(64, 128, 128, 8, 3),
        80: _C(32, 32, 128, 4, 4),
        88: _C(64, 128, 128, 4, 2),
        96: _C(64, 128, 128, 8, 2),
        104: _C(64, 64, 128, 4, 2),
        112: _C(64, 128, 128, 8, 3),
        120: _C(64, 128, 128, 4, 2),
        **dict.fromkeys(range(128, 145, 8), _C(64, 128, 128, 4, 3)),
        152: _C(64, 128, 128, 8, 2),
        **dict.fromkeys(range(160, 169, 8), _C(64, 128, 128, 8, 3)),
        **dict.fromkeys(range(176, 193, 8), _C(64, 128, 128, 4, 2)),
        200: _C(128, 128, 64, 8, 3),
        216: _C(128, 64, 128, 4, 2),
        224: _C(128, 128, 64, 8, 4),
        232: _C(128, 256, 64, 8, 3),
    },
}

SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
    (12, 1): SM121_SKINNY_GEMM_PLANS,
}

TRITON_SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
    (8, 0): SM80_TRITON_SKINNY_GEMM_PLANS,
    (8, 9): SM89_TRITON_SKINNY_GEMM_PLANS,
}


def get_skinny_gemm_plan(n: int, k: int) -> SkinnyGemmPlan | None:
    """Return the token-count plan for an `(N, K)` weight on this device."""
    return _get_skinny_gemm_plan(
        current_platform.get_device_capability(),
        shape_dynamic_skinny_gemm.is_available(),
        n,
        k,
    )


@functools.cache
def _get_skinny_gemm_plan(
    capability: tuple[int, int] | None, cute_available: bool, n: int, k: int
) -> SkinnyGemmPlan | None:
    # Cached because the op looks the plan up on every call; the merged dict
    # would otherwise be rebuilt per layer per step in eager mode.
    plan: SkinnyGemmPlan = {}
    if cute_available:
        plan.update(SKINNY_GEMM_PLANS_BY_CAPABILITY.get(capability, {}).get((n, k), {}))
    plan.update(
        TRITON_SKINNY_GEMM_PLANS_BY_CAPABILITY.get(capability, {}).get((n, k), {})
    )
    return plan or None
