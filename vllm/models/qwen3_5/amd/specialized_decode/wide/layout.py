# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shapes the wide kernels are compiled for: Qwen3.8-2.4T-A95B at TP8, one
rank's share. `SHAPES` is what the runtime checks the checkpoint's text
config against; every other constant follows from it."""

TP = 8

# the checkpoint's text config (``SHAPE_NAMES``), TP-independent
HIDDEN = 8192
GDN_NK, GDN_NV, GDN_DK, GDN_DV, CONV_TAPS = 16, 128, 128, 128, 4
ATTN_HEADS, ATTN_KV_HEADS, ATTN_DIM = 64, 4, 256
EXPERTS, TOPK, MOE_INTER, SHARED_INTER = 512, 10, 2048, 2048
MAX_LAYERS = 92

SHAPE_NAMES = (
    "hidden_size",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)
SHAPES = (
    HIDDEN,
    GDN_NK,
    GDN_NV,
    GDN_DK,
    GDN_DV,
    CONV_TAPS,
    ATTN_HEADS,
    ATTN_KV_HEADS,
    ATTN_DIM,
    EXPERTS,
    TOPK,
    MOE_INTER,
    SHARED_INTER,
)

# derived, one rank's
NK, NV = GDN_NK // TP, GDN_NV // TP  # 2 key heads, 16 value heads
KEY_DIM, VALUE_DIM = NK * GDN_DK, NV * GDN_DV  # 256, 2048
QKVZ = 2 * KEY_DIM + 2 * VALUE_DIM  # in_proj_qkvz's rows: [q | k | v | z]
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # [q | k | v] go through the causal conv
BA = 2 * NV  # in_proj_ba's rows: [b | a]

HEADS = ATTN_HEADS // TP  # 8 query heads a rank
KV_HEADS = max(1, ATTN_KV_HEADS // TP)  # the KV head is replicated past TP 4
QKV = HEADS * 2 * ATTN_DIM + 2 * KV_HEADS * ATTN_DIM  # [q, gate per head | k | v]
ATTN_CORE = HEADS * ATTN_DIM  # o_proj's input width a rank

SLOTS = TOPK + 1  # a row's routed experts, then the shared one (expert 512)
E_ALL = EXPERTS + 1  # routed experts + the fused shared expert
RI = MOE_INTER // TP  # an expert's intermediate rows a rank: 256
assert SHARED_INTER == MOE_INTER  # the shared expert runs as routed expert 512

# the execution model: one CTA per CU of an MI355X, all co-resident
BLOCKS = 256
THREADS = 256
WAVES = THREADS // 64
LDS_BYTES = 160 * 1024

# rows a step: the CUDA graph capture sizes in this range get a build; wider
# steps run vLLM's decode, whose MoE is at the HBM floor there
MIN_TOKENS = 1
MAX_TOKENS = 64
