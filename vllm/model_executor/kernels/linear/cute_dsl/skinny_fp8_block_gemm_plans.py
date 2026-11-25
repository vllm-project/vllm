# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured skinny block-FP8 GEMM plans.

Plans are keyed by device capability, local `(N, K)` weight shape and token
count. A block-FP8 linear (1x128 activation groups, 128x128 weight blocks)
whose shape is listed runs the skinny GEMM config planned for that token
count. Other token counts keep the small-M dispatch of the CUTLASS block-FP8
kernel (FlashInfer's TRT-LLM GEMM for small M, else CUTLASS).
"""

import functools

from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_block_gemm import (
    SkinnyFp8BlockGemmConfig as C,
)
from vllm.platforms import current_platform

SkinnyFp8BlockGemmPlan = dict[int, C]

# B300 (SM103), Qwen3.6-35B-A3B-FP8 local shapes at TP1 and TP2. Config:
# C(num_rows, block_size, outputs_per_block, static_k). Comment: minimum
# speedup over 4 seeds x 3 interleaved rounds (CUDA graph replay, cold L2)
# over the kernel the small-M dispatch would otherwise run (FlashInfer TRT-LLM
# or CUTLASS) / over CUTLASS. Entries are kept only at >= 1.10x over both.
# The shared-expert gate_up_proj shapes (1024, 2048) and (512, 2048) have no
# 8-token entry: they win in isolation (1.38x / 1.70x), but decode at batch 8
# is 2-3% slower end to end with them (TP1 / TP2).
SM103_SKINNY_FP8_BLOCK_GEMM_PLANS: dict[tuple[int, int], SkinnyFp8BlockGemmPlan] = {
    # Qwen3.6-35B-A3B TP1 GDN out_proj / attention o_proj.
    (2048, 4096): {
        1: C(1, 64, 1, 4096),  # 1.95x / 2.31x
        2: C(2, 128, 2, 4096),  # 2.01x / 2.37x
        4: C(4, 64, 2, 4096),  # 1.44x / 1.70x
    },
    # Qwen3.6-35B-A3B shared-expert gate_up_proj (TP1).
    (1024, 2048): {
        1: C(1, 128, 2, 2048),  # 2.17x / 2.90x
        2: C(2, 128, 2, 2048),  # 2.12x / 2.84x
        4: C(4, 128, 1, 2048),  # 1.71x / 2.28x
    },
    # Qwen3.6-35B-A3B shared-expert down_proj (TP1).
    (2048, 512): {
        1: C(1, 32, 2, 512),  # 1.29x / 1.98x
        2: C(2, 32, 2, 512),  # 1.20x / 1.87x
        4: C(4, 32, 2, 512),  # 1.13x / 1.76x
    },
    # Qwen3.6-35B-A3B TP2 GDN in_proj_qkvz.
    (6144, 2048): {
        1: C(1, 64, 2, 2048),  # 1.29x / 1.57x
        2: C(2, 128, 2, 2048),  # 1.24x / 1.51x
    },
    # Qwen3.6-35B-A3B TP2 attention qkv_proj.
    (4608, 2048): {
        1: C(1, 32, 2, 2048),  # 1.29x / 1.61x
        2: C(2, 128, 2, 2048),  # 1.36x / 1.71x
    },
    # Qwen3.6-35B-A3B TP2 GDN out_proj / attention o_proj.
    (2048, 2048): {
        1: C(1, 64, 2, 2048),  # 1.74x / 2.27x
        2: C(2, 128, 2, 2048),  # 1.69x / 2.20x
        4: C(4, 128, 2, 2048),  # 1.50x / 1.96x
    },
    # Qwen3.6-35B-A3B TP2 shared-expert gate_up_proj.
    (512, 2048): {
        1: C(1, 128, 1, 2048),  # 2.29x / 3.12x
        2: C(2, 128, 1, 2048),  # 2.22x / 3.04x
        4: C(4, 128, 1, 2048),  # 2.05x / 2.80x
        16: C(16, 128, 2, 2048),  # 1.26x / 1.71x
    },
}

SKINNY_FP8_BLOCK_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyFp8BlockGemmPlan]
] = {
    # Measured on B300 only; other devices keep CUTLASS until swept.
    (10, 3): SM103_SKINNY_FP8_BLOCK_GEMM_PLANS,
}


@functools.cache
def get_skinny_fp8_block_gemm_plan(n: int, k: int) -> SkinnyFp8BlockGemmPlan | None:
    """Return the token-count plan for an `(N, K)` block-FP8 weight here."""
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_block_gemm import (
        shape_dynamic_skinny_fp8_block_gemm,
    )

    plans = SKINNY_FP8_BLOCK_GEMM_PLANS_BY_CAPABILITY.get(
        current_platform.get_device_capability()
    )
    if plans is None or not shape_dynamic_skinny_fp8_block_gemm.is_available():
        return None
    return plans.get((n, k))
