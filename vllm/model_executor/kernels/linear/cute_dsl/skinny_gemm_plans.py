# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured skinny GEMM plans.

``SKINNY_GEMM_PLANS_BY_CAPABILITY`` covers unquantized BF16 linear layers and
``SKINNY_FP8_GEMM_PLANS_BY_CAPABILITY`` block-scaled FP8 linear layers (W8A16,
BF16 input). Plans are keyed by device capability, local ``(N, K)`` weight
shape and token count. A layer whose shape is listed runs the planned config
for that token count; other token counts use the layer's usual kernel. Entries
are kept only where the skinny GEMM beats that kernel by more than 5% under
CUDA graph replay.
"""

from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_gemm import (
    shape_dynamic_skinny_fp8_gemm,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.platforms import current_platform

SkinnyGemmPlan = dict[int, SkinnyGemmConfig]
SkinnyFp8GemmPlan = dict[int, SkinnyGemmConfig]

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

SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
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


# W8A16 block-FP8 skinny GEMM plans (128x128 FP8 weight blocks, BF16 input).
# Entries are kept where the kernel beats the FlashInfer block-FP8 path
# (activation quantization included) by more than 5% under CUDA graph replay.
# H100 / H200, measured on an H200 for TP=1 local shapes.
SM90_SKINNY_FP8_GEMM_PLANS: dict[tuple[int, int], SkinnyFp8GemmPlan] = {
    # Qwen3.5/3.6-35B-A3B shared-expert gate/up.
    (1024, 2048): {
        1: SkinnyGemmConfig(1, 256, 2),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=16),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=2),
    },
    # Qwen3.5/3.6-35B-A3B shared-expert down.
    (2048, 512): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
        4: SkinnyGemmConfig(4, 32, 4, k_unroll=2, vector_width=16),
    },
    # Qwen3.5/3.6-35B-A3B attention o_proj and GDN out_proj.
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=16),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, vector_width=16),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, vector_width=16),
    },
    # Qwen3-4B o_proj.
    (2560, 4096): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=2, vector_width=16),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4),
        4: SkinnyGemmConfig(4, 64, 2, vector_width=16),
    },
    # Qwen3-4B down.
    (2560, 9728): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=4),
    },
    # Qwen3-8B o_proj.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=16),
    },
    # Qwen3-8B down.
    (4096, 12288): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 2, k_unroll=4, vector_width=16),
    },
    # Qwen3-14B o_proj.
    (5120, 5120): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
    },
    # Qwen3-32B o_proj.
    (5120, 8192): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
    },
    # Qwen3-14B down.
    (5120, 17408): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=2, vector_width=16),
    },
    # Qwen3-32B down.
    (5120, 25600): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=2, vector_width=16),
    },
    # Qwen3-4B QKV.
    (6144, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 64, 4),
    },
    # Qwen3-8B QKV.
    (6144, 4096): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=4, vector_width=16),
    },
    # Qwen3-14B QKV.
    (7168, 5120): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
    },
    # Qwen3.5/3.6-35B-A3B attention QKV (with output gate).
    (9216, 2048): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2),
    },
    # Qwen3-32B QKV.
    (10240, 5120): {
        1: SkinnyGemmConfig(1, 32, 4, k_unroll=4),
    },
    # Qwen3.5/3.6-35B-A3B GDN in_proj_qkvz.
    (12288, 2048): {
        1: SkinnyGemmConfig(1, 32, 4, k_unroll=4),
    },
    # Qwen3-4B gate/up.
    (19456, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
    },
    # Qwen3-8B gate/up.
    (24576, 4096): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4),
    },
    # Qwen3-14B gate/up.
    (34816, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=4),
    },
    # Qwen3-32B gate/up.
    (51200, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=4),
    },
}

# Largest token count any FP8 plan lists; callers reject larger batches before
# looking a plan up.
SKINNY_FP8_MAX_TOKENS = 4

SKINNY_FP8_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyFp8GemmPlan]
] = {
    (9, 0): SM90_SKINNY_FP8_GEMM_PLANS,
}


# Grouped W8A16 kernel for the routed experts of a block-FP8 MoE layer
# (skinny_fp8_moe_gemm.py), keyed by the (N, K) of one expert's weight and the
# token count. One activation row per CTA; measured on H200 with the experts'
# weights cold (16 rotating expert sets), against the Triton fused MoE path at
# the same token counts.
SM90_SKINNY_FP8_MOE_GEMM_PLANS: dict[tuple[int, int], SkinnyFp8GemmPlan] = {
    # Qwen3.5/3.6-35B-A3B experts: gate_up (2 x 512, 2048), down (2048, 512).
    (1024, 2048): {
        1: SkinnyGemmConfig(1, 64, 8, vector_width=16),
        2: SkinnyGemmConfig(1, 32, 8, vector_width=16),
        4: SkinnyGemmConfig(1, 32, 8, vector_width=16),
    },
    (2048, 512): {
        1: SkinnyGemmConfig(1, 64, 8),
        2: SkinnyGemmConfig(1, 32, 32),
        4: SkinnyGemmConfig(1, 32, 16, vector_width=16),
    },
    # Qwen3-30B-A3B experts: gate_up (2 x 768, 2048), down (2048, 768). With
    # 128 experts, 4 tokens already share enough experts that the Triton path
    # (one weight read per expert) is faster in the model, so stop at 2.
    (1536, 2048): {
        1: SkinnyGemmConfig(1, 64, 8),
        2: SkinnyGemmConfig(1, 32, 16, vector_width=16),
    },
    (2048, 768): {
        1: SkinnyGemmConfig(1, 32, 8),
        2: SkinnyGemmConfig(1, 32, 16),
    },
}

SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int], dict[tuple[int, int], SkinnyFp8GemmPlan]
] = {
    (9, 0): SM90_SKINNY_FP8_MOE_GEMM_PLANS,
}


def get_skinny_fp8_moe_gemm_plan(n: int, k: int) -> SkinnyFp8GemmPlan | None:
    """Return the token-count plan for one expert's ``(N, K)`` block-FP8 weight."""
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_moe_gemm import (
        shape_dynamic_skinny_fp8_moe_gemm,
    )

    plans = SKINNY_FP8_MOE_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability()
    )
    if plans is None or not shape_dynamic_skinny_fp8_moe_gemm.is_available():
        return None
    return plans.get((n, k))


def get_skinny_fp8_gemm_plan(n: int, k: int) -> SkinnyFp8GemmPlan | None:
    """Return the token-count plan for an ``(N, K)`` block-FP8 weight."""
    plans = SKINNY_FP8_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability()
    )
    if plans is None or not shape_dynamic_skinny_fp8_gemm.is_available():
        return None
    return plans.get((n, k))
