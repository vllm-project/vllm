# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured skinny GEMM plans for unquantized BF16 linear layers.

Plans are keyed by device capability, local ``(N, K)`` weight shape and token
count. A layer whose shape is listed runs the planned config for that token
count; other token counts use ``F.linear``. Entries are kept only where the
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

# SM90, keyed by capability (9, 0) so it also selects H100 / H800 / H20; every
# entry was measured on an H200 (132 SMs, 4.8 TB/s) only. TP=1 local shapes of
# dense Qwen3 / Llama-3.1 models, plus the two BF16 projections of the Qwen3.5 /
# 3.6 MoE models. Every 4-token entry was also checked inside
# a running model (nsys, 4 concurrent requests); 4-token cells that win in
# isolation but lose there are left out. static_k entries have no k_unroll:
# the static path is fully unrolled.
SM90_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # Qwen3.5 / 3.6 MoE shared-expert gate (one output).
    (1, 2048): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 256, 1, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 256, 1),
    },
    # Qwen3.5 / 3.6 MoE GDN fused b/a projection.
    (64, 2048): {
        1: SkinnyGemmConfig(1, 256, 1),
        2: SkinnyGemmConfig(2, 256, 1),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4),
    },
    # Qwen3-4B attention output projection.
    (2560, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 4),
    },
    # Qwen3-4B down projection.
    (2560, 9728): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=9728),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=9728),
    },
    # Qwen3-8B / Llama-3.1-8B attention output projection.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=2, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=4096),
    },
    # Qwen3-8B down projection.
    (4096, 12288): {
        1: SkinnyGemmConfig(1, 256, 4, vector_width=4, static_k=12288),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=12288),
        4: SkinnyGemmConfig(4, 128, 4, vector_width=4, static_k=12288),
    },
    # Llama-3.1-8B down projection. M=4 is left to cuBLAS: it wins in isolation
    # but loses inside the model (measured with nsys at 4 concurrent requests).
    (4096, 14336): {
        1: SkinnyGemmConfig(1, 224, 1, vector_width=4, static_k=14336),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=14336),
    },
    # Qwen3-14B attention output projection.
    (5120, 5120): {
        1: SkinnyGemmConfig(1, 64, 4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=5120),
    },
    # Qwen3-32B attention output projection.
    (5120, 8192): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=2, static_k=8192),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=8192),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, vector_width=4),
    },
    # Qwen3-14B down projection.
    (5120, 17408): {
        1: SkinnyGemmConfig(1, 64, 4, static_k=17408),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=17408),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, vector_width=4),
    },
    # Qwen3-32B down projection.
    (5120, 25600): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=25600),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=25600),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, vector_width=4),
    },
    # Qwen3-4B fused QKV projection.
    (6144, 2560): {
        1: SkinnyGemmConfig(1, 32, 4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 4, static_k=2560),
    },
    # Qwen3-8B / Llama-3.1-8B fused QKV projection.
    (6144, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=2, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 3, vector_width=2, static_k=4096),
    },
    # Qwen3-14B fused QKV projection.
    (7168, 5120): {
        1: SkinnyGemmConfig(1, 64, 2, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=5120),
    },
    # Qwen3-32B fused QKV projection.
    (10240, 5120): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=2, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=5120),
    },
    # Qwen3-4B fused gate/up projection.
    (19456, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 2, vector_width=4, static_k=2560),
    },
    # Qwen3-8B fused gate/up projection.
    (24576, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=2),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=2, static_k=4096),
        4: SkinnyGemmConfig(4, 32, 4, vector_width=4, static_k=4096),
    },
    # Llama-3.1-8B fused gate/up projection.
    (28672, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=2),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=2, static_k=4096),
    },
    # Qwen3-14B fused gate/up projection.
    (34816, 5120): {
        1: SkinnyGemmConfig(1, 64, 1, vector_width=2, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=5120),
    },
    # Qwen3-32B fused gate/up projection.
    (51200, 5120): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=5120),
    },
    # Llama-3.1-8B LM head.
    (128256, 4096): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 256, 4, vector_width=4, static_k=4096),
    },
    # Qwen3-4B LM head.
    (151936, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4),
    },
    # Qwen3-8B LM head.
    (151936, 4096): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 256, 4, vector_width=4, static_k=4096),
    },
    # Qwen3-14B / Qwen3-32B LM head.
    (151936, 5120): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=5120),
    },
}

SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
    (9, 0): SM90_SKINNY_GEMM_PLANS,
    (12, 1): SM121_SKINNY_GEMM_PLANS,
}


def get_skinny_gemm_plan(n: int, k: int) -> SkinnyGemmPlan | None:
    """Return the token-count plan for an ``(N, K)`` weight on this device."""
    plans = SKINNY_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability()
    )
    if plans is None or not shape_dynamic_skinny_gemm.is_available():
        return None
    return plans.get((n, k))
