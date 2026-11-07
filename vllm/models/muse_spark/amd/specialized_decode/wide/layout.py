# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shapes the wide kernels are compiled for: Muse Spark at TP8 = EP8 on an
MI300X (gfx942), one rank's share. ``SHAPES`` is what the runtime checks the
checkpoint's text config against; every other constant follows from it."""

from dataclasses import dataclass

TP = 8

# the checkpoint's text config (``SHAPE_NAMES``), TP-independent
HIDDEN = 8192
POOLED = 4096  # moe_hidden_size: the pooler's output, the experts' K
INTER = 4096  # moe_intermediate_size
EXPERTS, TOPK = 256, 8
HEADS, KV_HEADS, HEAD_DIM = 128, 16, 64
LAYERS = 62

SHAPE_NAMES = (
    "hidden_size",
    "moe_hidden_size",
    "moe_intermediate_size",
    "num_local_experts",
    "num_experts_per_tok",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_hidden_layers",
)
SHAPES = (HIDDEN, POOLED, INTER, EXPERTS, TOPK, HEADS, KV_HEADS, HEAD_DIM, LAYERS)

# layers the kernels run; layer 0 (embedding input) and the last stay vLLM's
FIRST_LAYER, LAST_LAYER = 1, LAYERS - 2

# derived, one rank's
E_LOCAL = EXPERTS // TP  # 32 local experts (EP)
Q_ROWS = HEADS // TP * HEAD_DIM  # 1024: also o_proj's K
KV_ROWS = KV_HEADS // TP * HEAD_DIM  # 128
QKVG = 2 * Q_ROWS + 2 * KV_ROWS  # 2304 rows: [q | k | v | o_gate]
POOL_ROWS = POOLED // TP  # 512 pooler output rows a rank
UNPOOL_ROWS = HIDDEN // TP  # 1024 unpooler output rows a rank

# eps and constants of the stock modules
RMS_EPS = 1e-5  # rms_norm_eps: pre-norms, QK-norm, O-norm, pre_expert_norm
POST_EPS = 1e-8  # post_norm_eps: post_attn, post_ffn, post_expert norms
ROUTE_EPS = 1e-15
GAIN_CENTER = 1.0  # zero-centered gammas: weight + 1


@dataclass(frozen=True)
class Quant:
    """A per-token FP8 (e4m3fnuz) quant: ``scale = max(amax * (1/fp8_max) or
    amax / fp8_max, min_scale)``, ``q = clamp(x * rcp(scale))``."""

    fp8_max: float
    min_scale: float
    divide: bool  # the scale by true division (the codes: ``ops.quant_factor``)


# Each site matches the op that quantizes it in the stock forward, with
# VLLM_ROCM_USE_AITER=1. Dense inputs go through AITER's
# ``dynamic_per_token_scaled_quant`` (max 240, scale ``amax * (1/240)``); the
# expert input through vLLM's ``scaled_fp8_quant`` and the expert
# intermediate through ``silu_mul_token_quant`` (both max 224, floor
# ``1/(224*512)``). The GPU stage tests pin each against its op.
DENSE_QUANT = Quant(240.0, 0.0, divide=False)
MOE_IN_QUANT = Quant(224.0, 1.0 / (224.0 * 512.0), divide=False)
MOE_MID_QUANT = Quant(224.0, 1.0 / (224.0 * 512.0), divide=True)

# the MoE decomposition (tests/models/muse_spark/amd_specialized_decode)
PIECE = 32  # intermediate columns an up piece: 32 gate + 32 up weight rows
OWNER = 64  # output rows a down owner
UP_PIECES = INTER // PIECE  # 128 an expert
DOWN_OWNERS = POOLED // OWNER  # 64 an expert

# the execution model: one CTA per CU of an MI300X, all co-resident
XCDS = 8
CUS_PER_XCD = 38
BLOCKS = XCDS * CUS_PER_XCD  # 304; CTA ``b`` runs on XCD ``b % 8``
THREADS = 256
WAVES = THREADS // 64
LDS_BYTES = 64 * 1024

# rows a step: the CUDA graph capture sizes in this range get a build
MIN_TOKENS = 1
MAX_TOKENS = 128
