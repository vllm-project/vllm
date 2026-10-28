# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as
# vendored by ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/config.py).
# The original source code was licensed under the Apache License 2.0 and included the
# following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths
#   rewritten to this package; reduced to the GLM-5 geometry the vLLM MonoKernel
#   uses.

"""GLM-5 geometry of the MonoKernel."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LayerConfig:
    """Compile-time geometry of one TP decode shard."""

    name: str
    hidden: int
    q_lora: int
    kv_lora: int
    pe_dim: int
    nope_dim: int
    v_dim: int
    n_experts: int
    top_k: int
    inter: int
    route_scale: float
    local_heads: int

    @property
    def qkv_a_rows(self) -> int:
        return self.q_lora + self.kv_lora + self.pe_dim

    @property
    def moe_slots(self) -> int:
        return 1 + self.top_k

    @property
    def softmax_scale(self) -> float:
        return (self.nope_dim + self.pe_dim) ** -0.5


GLM5_CONFIG = LayerConfig(
    name="glm5",
    hidden=6144,
    q_lora=2048,
    kv_lora=512,
    pe_dim=64,
    nope_dim=192,
    v_dim=256,
    n_experts=256,
    top_k=8,
    inter=256,
    route_scale=2.5,
    local_heads=8,
)

GLM5_REFERENCE_TP = 8
GLM5_TP_SIZES = (4, GLM5_REFERENCE_TP)
GLM5_KERNEL_SAMPLES = (1, 2, 4, 5, 6, 8, 10, 12)
GLM5_GLOBAL_HEADS = GLM5_CONFIG.local_heads * GLM5_REFERENCE_TP
GLM5_GLOBAL_INTER = GLM5_CONFIG.inter * GLM5_REFERENCE_TP


def glm5_tp_config(tp_size: int) -> LayerConfig:
    """Return the one GLM-5 shard geometry for ``tp_size``."""
    if tp_size not in GLM5_TP_SIZES:
        raise ValueError(
            f"GLM-5 tensor parallel size must be one of {GLM5_TP_SIZES}, got {tp_size}"
        )
    return replace(
        GLM5_CONFIG,
        local_heads=GLM5_GLOBAL_HEADS // tp_size,
        inter=GLM5_GLOBAL_INTER // tp_size,
    )


# Fixed GLM-5 geometry used by its performance-specialized MonoKernel.
HIDDEN = GLM5_CONFIG.hidden
Q_LORA = GLM5_CONFIG.q_lora
KV_LORA = GLM5_CONFIG.kv_lora
PE_DIM = GLM5_CONFIG.pe_dim
NOPE_DIM = GLM5_CONFIG.nope_dim
V_DIM = GLM5_CONFIG.v_dim
QKV_A_ROWS = GLM5_CONFIG.qkv_a_rows
N_EXPERTS = GLM5_CONFIG.n_experts
TOP_K = GLM5_CONFIG.top_k
MOE_SLOTS = GLM5_CONFIG.moe_slots
SHARED_EXPERT = N_EXPERTS
INTER = GLM5_CONFIG.inter
ROUTE_SCALE = GLM5_CONFIG.route_scale
EPS = 1e-5
FP8_MAX = 448.0
# attention matrices the kernel can read as checkpoint FP8 E4M3 with one F32 scale per
# output row (w_uk stays BF16: kv_b_proj's per-row scale runs along its K)
FP8_ATTENTION = ("qkv_a", "q_b", "uv", "o")
SOFTMAX_SCALE = GLM5_CONFIG.softmax_scale
MAX_LAYERS_PER_STEP = 128
