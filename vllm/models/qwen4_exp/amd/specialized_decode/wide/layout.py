# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shapes the wide kernels are compiled for: Qwen3.8-Flash-Next, one rank's
share at TP1 or TP2 (`shard`). `SHAPES` is what the runtime checks the
checkpoint's text config against; every other constant follows from it."""

import functools
from typing import NamedTuple

TPS = (1, 2)  # the tensor parallel sizes the kernels are built for

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

SLOTS = TOPK + 1  # a row's routed experts, then the shared one (expert 512)
E_ALL = EXPERTS + 1  # routed experts + the fused shared expert
assert SHARED_INTER == MOE_INTER  # the shared expert runs as routed expert 512

# the execution model: one CTA per CU of an MI355X, all co-resident
BLOCKS = 256
THREADS = 256
WAVES = THREADS // 64
LDS_BYTES = 160 * 1024

# rows a step: the CUDA graph capture sizes in `MIN_TOKENS .. Shard.MAX_TOKENS`
# get a build
MIN_TOKENS = 1
# TP1 stops at 64: the gain over vLLM's TP1 decode falls with the width, and an
# earlier TP1 build of these kernels was slower than it above 64 rows
_MAX_TOKENS = {1: 64, 2: 192}


class Shard(NamedTuple):
    """A rank's share of the model at tensor parallel size `TP` (`shard`), and
    the widest step the kernels run at it."""

    TP: int
    NK: int  # key heads
    NV: int  # value heads
    KEY_DIM: int
    VALUE_DIM: int
    QKVZ: int  # in_proj_qkvz's rows: [q | k | v | z]
    CONV_DIM: int  # [q | k | v] go through the causal conv
    BA: int  # in_proj_ba's rows: [b | a]
    QKV: int  # the QSA qkv projection's rows
    QSA_CORE: int  # o_proj's input width
    RI: int  # an expert's intermediate rows
    MAX_TOKENS: int  # rows a step: capture sizes up to it get a build


@functools.cache
def shard(tp: int) -> Shard:
    """The rank's share at `tp`; ValueError for a TP the kernels are not built
    for (`TPS`)."""
    if tp not in TPS:
        raise ValueError(f"TP {tp}: the kernels are built for TP in {TPS}")
    nk, nv = GDN_NK // tp, GDN_NV // tp
    key_dim, value_dim = nk * GDN_DK, nv * GDN_DV
    return Shard(
        TP=tp,
        NK=nk,
        NV=nv,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        QKVZ=2 * key_dim + 2 * value_dim,
        CONV_DIM=2 * key_dim + value_dim,
        BA=2 * nv,
        QKV=(2 * QSA_HEADS + 2 * QSA_KV_HEADS) // tp * QSA_DIM,
        QSA_CORE=QSA_HEADS // tp * QSA_DIM,
        RI=MOE_INTER // tp,
        MAX_TOKENS=_MAX_TOKENS[tp],
    )


# the widest step of any TP: it sizes the exchange mailbox and the MoE's split
# marks, which every build shares
MAX_TOKENS = max(shard(tp).MAX_TOKENS for tp in TPS)
