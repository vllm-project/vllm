# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shapes the wide kernels are compiled for: Qwen3.8-Flash-Next at TP2,
one rank's share. ``SHAPES`` is what the runtime checks the checkpoint's text
config against; every other constant follows from it."""

TP = 2

# the checkpoint's text config (``SHAPE_NAMES``), TP-independent
HIDDEN = 2560
HC = 4  # hyper-connection streams
LOWRANK = 320  # the gated residual's low-rank width
GDN_NK, GDN_NV, GDN_DK, GDN_DV, CONV_TAPS = 16, 48, 128, 128, 4
QSA_HEADS, QSA_KV_HEADS, QSA_DIM = 24, 2, 256
EXPERTS, TOPK, MOE_INTER, SHARED_INTER = 512, 10, 640, 640
MAX_LAYERS = 48

SHAPE_NAMES = (
    "hidden_size",
    "hc_count",
    "hc_lowrank",
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
    HC,
    LOWRANK,
    GDN_NK,
    GDN_NV,
    GDN_DK,
    GDN_DV,
    CONV_TAPS,
    QSA_HEADS,
    QSA_KV_HEADS,
    QSA_DIM,
    EXPERTS,
    TOPK,
    MOE_INTER,
    SHARED_INTER,
)

# derived, one rank's
HC_HIDDEN = HC * HIDDEN  # the multi-stream state's width
INJECT = LOWRANK  # first injection row of the merged down + inject weight
DOWN_ROWS = 336  # [down 320 | inject 4 | pad 12]: vLLM pads the merged rows to 16

NK, NV = GDN_NK // TP, GDN_NV // TP  # 8 key heads, 24 value heads
KEY_DIM, VALUE_DIM = NK * GDN_DK, NV * GDN_DV  # 1024, 3072
QKVZ = 2 * KEY_DIM + 2 * VALUE_DIM  # in_proj_qkvz's rows: [q | k | v | z]
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # [q | k | v] go through the causal conv
BA = 2 * NV  # in_proj_ba's rows: [b | a]
GDN_IN = QKVZ + BA  # the GDN input projection's rows, qkvz then ba

QSA_CORE = QSA_HEADS // TP * QSA_DIM  # o_proj's input width a rank

SLOTS = TOPK + 1  # a row's routed experts, then the shared one (expert 512)
E_ALL = EXPERTS + 1  # routed experts + the fused shared expert
RI = MOE_INTER // TP  # an expert's intermediate rows a rank: 320
assert SHARED_INTER == MOE_INTER  # the shared expert runs as routed expert 512

# the execution model: one CTA per CU of an MI355X, all co-resident
BLOCKS = 256
THREADS = 256
WAVES = THREADS // 64
LDS_BYTES = 160 * 1024

# rows a step: the CUDA graph capture sizes in this range get a build
MIN_TOKENS = 1
MAX_TOKENS = 128
