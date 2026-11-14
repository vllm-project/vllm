# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured skinny GEMM plans for unquantized BF16 linear layers.

Plans are keyed by device capability, local `(N, K)` weight shape and token
count. A layer whose shape is listed runs the planned config for that token
count; other token counts use `F.linear`. Entries are kept only where the
skinny GEMM beats cuBLAS by more than 5% under CUDA graph replay.
"""

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.platforms import current_platform

SkinnyGemmPlan = dict[int, SkinnyGemmConfig]

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

# RTX PRO 6000 Blackwell family (SM120, 156 SMs, GDDR7), measured on
# GLM-5.3-Flash local shapes at TP=8. The SM120 SKUs differ in SM count, so a
# shape is only listed where the plan wins on the smallest of them; see
# `benchmarks/kernels/benchmark_skinny_gemm_sm120.py` for the sweep.
SM120_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # LM head.
    (19360, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=4096),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, static_k=4096),
    },
    # MLA fused q_a/kv_a projection.
    (3336, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, static_k=4096),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=4, static_k=4096),
    },
    # KDA output projection.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=4, static_k=4096),
    },
    # MLA q_b projection.
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=4096),
    },
    # KDA fused QKV/beta/f_a/g_a input projection.
    (2560, 4096): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=4, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=4096),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=4, static_k=4096),
    },
    # KDA g_b projection.
    (160, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=4, static_k=4096),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=4, vector_width=8, static_k=4096),
    },
    # Router / KDA f_b projection.
    (288, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=4, static_k=4096),
    },
    # MLA q_a / kv_a output projection.
    (512, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=4, vector_width=4, static_k=4096),
    },
    # KDA o_norm output projection.
    (4096, 2048): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4, static_k=2048),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4, static_k=2048),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, static_k=2048),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=4, static_k=2048),
    },
    # MLA kv_b projection.
    (4096, 1536): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4, static_k=1536),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, static_k=1536),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4, static_k=1536),
    },
    # MLA q_b output projection.
    (2048, 1536): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=4, static_k=1536),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, static_k=1536),
        8: SkinnyGemmConfig(8, 64, 2, k_unroll=4, static_k=1536),
    },
    # Indexer wq_b projection.
    (4096, 1280): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=4, static_k=1280),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=4, vector_width=4, static_k=1280),
    },
    # Indexer wk / weights_proj.
    (4096, 1024): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4, static_k=1024),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4, static_k=1024),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, static_k=1024),
    },
}

SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
    (12, 0): SM120_SKINNY_GEMM_PLANS,
    (12, 1): SM121_SKINNY_GEMM_PLANS,
}


def get_skinny_gemm_plan(n: int, k: int) -> SkinnyGemmPlan | None:
    """Return the token-count plan for an `(N, K)` weight on this device."""
    plans = SKINNY_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability()
    )
    if plans is None or not shape_dynamic_skinny_gemm.is_available():
        return None
    return plans.get((n, k))
