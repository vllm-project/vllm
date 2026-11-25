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

# B200 and B300. Every entry beats cuBLAS under CUDA graph replay on both GPUs: by at
# least 10% on B200, and on B300 by at least 10% except six dense Qwen3 entries
# (5-10%). Measured on:
# - the BF16 projections that stay unquantized in the Qwen3.5 / Qwen3.6 MoE FP8
#   checkpoints (GDN in_proj_ba, shared-expert gate, router, LM head) at the local
#   shapes of Qwen3.6-35B-A3B TP=1/2/4 and Qwen3.5-397B-A17B TP=4/8;
# - the dense Qwen3 local shapes (Qwen3-8B TP=1/2, Qwen3-14B TP=1/2/4,
#   Qwen3-32B TP=1/2/4/8);
# - the Qwen3 MoE attention, router and LM head local shapes (Qwen3-30B-A3B
#   TP=1/2/4, Qwen3-235B-A22B TP=4/8).
SM100_SKINNY_GEMM_PLANS: dict[tuple[int, int], SkinnyGemmPlan] = {
    # Qwen3.6-35B-A3B TP1 GDN in_proj_ba.
    (64, 2048): {
        1: SkinnyGemmConfig(1, 128, 2, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 2, static_k=2048),
        4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2048),
    },
    # Qwen3.6-35B-A3B TP2 GDN in_proj_ba.
    (32, 2048): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, static_k=2048),
    },
    # Qwen3.6-35B-A3B TP4 GDN in_proj_ba.
    (16, 2048): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 1, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, static_k=2048),
    },
    # Qwen3.6-35B-A3B shared-expert gate (replicated).
    (1, 2048): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 64, 1, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=2, vector_width=4, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, static_k=2048),
    },
    # Qwen3.6-35B-A3B router (replicated).
    (256, 2048): {
        1: SkinnyGemmConfig(1, 64, 1, k_unroll=2, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 1, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 1, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2048),
    },
    # Qwen3.6-35B-A3B TP1 LM head.
    (248320, 2048): {
        1: SkinnyGemmConfig(1, 256, 4),
        2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
    },
    # Qwen3.6-35B-A3B TP2 LM head.
    (124160, 2048): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
    },
    # Qwen3.6-35B-A3B TP4 LM head.
    (62080, 2048): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 4),
        4: SkinnyGemmConfig(4, 32, 4),
    },
    # Qwen3.5-397B-A17B TP4 GDN in_proj_ba.
    (32, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=2, static_k=4096),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=2, static_k=4096),
        8: SkinnyGemmConfig(8, 256, 1, k_unroll=2, static_k=4096),
    },
    # Qwen3.5-397B-A17B TP8 GDN in_proj_ba.
    (16, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, static_k=4096),
        4: SkinnyGemmConfig(4, 256, 1, static_k=4096),
        8: SkinnyGemmConfig(8, 256, 1, k_unroll=2, static_k=4096),
    },
    # Qwen3.5-397B-A17B shared-expert gate (replicated).
    (1, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, static_k=4096),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=2, static_k=4096),
    },
    # Qwen3.5-397B-A17B router (replicated).
    (512, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 256, 1, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 1, static_k=4096),
        8: SkinnyGemmConfig(8, 128, 2, k_unroll=2, static_k=4096),
    },
    # Qwen3.5-397B-A17B TP4 LM head.
    (62080, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=2),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4),
    },
    # Qwen3.5-397B-A17B TP8 LM head.
    (31040, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
    },
    # Dense Qwen3 (Qwen3-8B TP=1/2, Qwen3-14B TP=1/2/4, Qwen3-32B TP=1/2/4/8).
    # 32B TP8 o.
    (5120, 1024): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=1024),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2, vector_width=4, static_k=1024),
    },
    # 14B TP4 o.
    (5120, 1280): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=1280),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=1280),
    },
    # 8B TP2 o.
    (4096, 2048): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 64, 4, static_k=2048),
        4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=2048),
    },
    # 32B TP4 o.
    (5120, 2048): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 32, 4, static_k=2048),
    },
    # 14B TP2 o.
    (5120, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, vector_width=4, static_k=2560),
    },
    # 32B TP8 down.
    (5120, 3200): {
        1: SkinnyGemmConfig(1, 160, 2, k_unroll=2, vector_width=4, static_k=3200),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2, vector_width=4, static_k=3200),
    },
    # 8B TP2 qkv.
    (3072, 4096): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=2, static_k=4096),
    },
    # 8B TP1 o.
    (4096, 4096): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 4, vector_width=4, static_k=4096),
    },
    # 32B TP2 o.
    (5120, 4096): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=4096),
    },
    # 8B TP1 qkv.
    (6144, 4096): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=2, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=4096),
    },
    # 8B TP2 gate_up.
    (12288, 4096): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4, static_k=4096),
    },
    # 8B TP1 gate_up.
    (24576, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=4, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4, static_k=4096),
    },
    # 8B TP2 lm_head.
    (75968, 4096): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=4096),
    },
    # 8B TP1 lm_head.
    (151936, 4096): {
        1: SkinnyGemmConfig(1, 32, 2, vector_width=4, static_k=4096),
    },
    # 14B TP4 down.
    (5120, 4352): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=4352),
        2: SkinnyGemmConfig(2, 32, 4, static_k=4352),
    },
    # 32B TP8 qkv.
    (1280, 5120): {
        1: SkinnyGemmConfig(1, 160, 2, k_unroll=2, static_k=5120),
        2: SkinnyGemmConfig(2, 160, 2, static_k=5120),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4, static_k=5120),
        8: SkinnyGemmConfig(8, 64, 2, static_k=5120),
    },
    # 14B TP4 qkv.
    (1792, 5120): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 4, static_k=5120),
        4: SkinnyGemmConfig(4, 128, 4, vector_width=4, static_k=5120),
    },
    # 32B TP4 qkv.
    (2560, 5120): {
        1: SkinnyGemmConfig(1, 160, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4, static_k=5120),
        4: SkinnyGemmConfig(4, 32, 2, static_k=5120),
    },
    # 14B TP2 qkv.
    (3584, 5120): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=5120),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=4, vector_width=4, static_k=5120),
    },
    # 14B TP1 o, 32B TP2 qkv.
    (5120, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=5120),
        4: SkinnyGemmConfig(4, 32, 4, static_k=5120),
    },
    # 32B TP8 gate_up.
    (6400, 5120): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 32, 4, static_k=5120),
    },
    # 14B TP1 qkv.
    (7168, 5120): {
        1: SkinnyGemmConfig(1, 64, 1, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=4, vector_width=4, static_k=5120),
    },
    # 14B TP4 gate_up.
    (8704, 5120): {
        1: SkinnyGemmConfig(1, 128, 2, static_k=5120),
    },
    # 32B TP1 qkv.
    (10240, 5120): {
        1: SkinnyGemmConfig(1, 160, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=5120),
    },
    # 32B TP4 gate_up.
    (12800, 5120): {
        1: SkinnyGemmConfig(1, 64, 2, k_unroll=4, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4, static_k=5120),
    },
    # 14B TP2 gate_up.
    (17408, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=2, vector_width=4, static_k=5120),
    },
    # 32B TP8 lm_head.
    (18992, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 32, 4, static_k=5120),
    },
    # 32B TP2 gate_up.
    (25600, 5120): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=5120),
    },
    # 14B TP1 gate_up.
    (34816, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=4, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4, static_k=5120),
    },
    # 14B TP4 lm_head, 32B TP4 lm_head.
    (37984, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 160, 2, k_unroll=2, vector_width=4, static_k=5120),
    },
    # 32B TP1 gate_up.
    (51200, 5120): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2, static_k=5120),
    },
    # 14B TP2 lm_head, 32B TP2 lm_head.
    (75968, 5120): {
        1: SkinnyGemmConfig(1, 32, 2, k_unroll=2, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2, vector_width=4, static_k=5120),
    },
    # 14B TP1 lm_head, 32B TP1 lm_head.
    (151936, 5120): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=5120),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4, static_k=5120),
    },
    # 8B TP2 down.
    (4096, 6144): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=6144),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4, static_k=6144),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, vector_width=4, static_k=6144),
    },
    # 32B TP4 down.
    (5120, 6400): {
        1: SkinnyGemmConfig(1, 160, 2, k_unroll=2, vector_width=4, static_k=6400),
        2: SkinnyGemmConfig(2, 32, 4, static_k=6400),
        4: SkinnyGemmConfig(4, 32, 4, k_unroll=4, static_k=6400),
    },
    # 32B TP1 o.
    (5120, 8192): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=8192),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 128, 4),
    },
    # 14B TP2 down.
    (5120, 8704): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=8704),
        2: SkinnyGemmConfig(2, 32, 4, static_k=8704),
    },
    # 8B TP1 down.
    (4096, 12288): {
        1: SkinnyGemmConfig(1, 256, 2, static_k=12288),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=2, vector_width=4, static_k=12288),
    },
    # 32B TP2 down.
    (5120, 12800): {
        1: SkinnyGemmConfig(1, 160, 2, vector_width=4, static_k=12800),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2, static_k=12800),
    },
    # 14B TP1 down.
    (5120, 17408): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4, static_k=17408),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=17408),
    },
    # 32B TP1 down.
    (5120, 25600): {
        1: SkinnyGemmConfig(1, 160, 1, k_unroll=4, vector_width=4, static_k=25600),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4, static_k=25600),
    },
    # Qwen3 MoE (Qwen3-30B-A3B TP=1/2/4, Qwen3-235B-A22B TP=4/8): attention, router and
    # LM head. Routed experts run in FusedMoE and are not affected.
    # 30B-A3B TP4 o.
    (2048, 1024): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, vector_width=4, static_k=1024),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1024),
        4: SkinnyGemmConfig(4, 64, 4, static_k=1024),
    },
    # 235B-A22B TP8 o.
    (4096, 1024): {
        1: SkinnyGemmConfig(1, 64, 4, vector_width=4, static_k=1024),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1024),
    },
    # 30B-A3B TP1 router, 30B-A3B TP2 router, 30B-A3B TP4 router.
    (128, 2048): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 1, static_k=2048),
        4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, static_k=2048),
    },
    # 30B-A3B TP4 qkv.
    (1280, 2048): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=2, static_k=2048),
        2: SkinnyGemmConfig(2, 256, 2, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=2),
    },
    # 30B-A3B TP2 o.
    (2048, 2048): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 64, 2, vector_width=4, static_k=2048),
        8: SkinnyGemmConfig(8, 128, 2),
    },
    # 30B-A3B TP2 qkv.
    (2560, 2048): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2048),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2048),
        4: SkinnyGemmConfig(4, 128, 2, static_k=2048),
    },
    # 30B-A3B TP4 lm_head.
    (37984, 2048): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
    },
    # 30B-A3B TP2 lm_head.
    (75968, 2048): {
        1: SkinnyGemmConfig(1, 256, 4, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=2),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
    },
    # 30B-A3B TP1 lm_head.
    (151936, 2048): {
        1: SkinnyGemmConfig(1, 256, 4),
        2: SkinnyGemmConfig(2, 32, 2, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 4),
    },
    # 235B-A22B TP4 router, 235B-A22B TP8 router.
    (128, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, static_k=4096),
        4: SkinnyGemmConfig(4, 256, 1, static_k=4096),
        8: SkinnyGemmConfig(8, 256, 1, static_k=4096),
    },
    # 235B-A22B TP8 qkv.
    (1280, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, static_k=4096),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1),
    },
    # 30B-A3B TP1 o.
    (2048, 4096): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4),
        8: SkinnyGemmConfig(8, 128, 2),
    },
    # 235B-A22B TP4 qkv.
    (2304, 4096): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=4096),
        2: SkinnyGemmConfig(2, 128, 1, static_k=4096),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=4),
    },
    # 235B-A22B TP8 lm_head.
    (18992, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4),
    },
    # 235B-A22B TP4 lm_head.
    (37984, 4096): {
        1: SkinnyGemmConfig(1, 256, 2, k_unroll=2),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4),
        4: SkinnyGemmConfig(4, 32, 2, k_unroll=4, vector_width=4),
    },
}

SKINNY_GEMM_PLANS_BY_CAPABILITY: dict[
    tuple[int, int] | None, dict[tuple[int, int], SkinnyGemmPlan]
] = {
    (10, 0): SM100_SKINNY_GEMM_PLANS,
    # B300: every entry also beats cuBLAS by at least 10% on SM103.
    (10, 3): SM100_SKINNY_GEMM_PLANS,
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
