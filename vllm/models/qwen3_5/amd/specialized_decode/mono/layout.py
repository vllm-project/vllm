# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 (``Qwen3_5MoeForCausalLM``, amd/Qwen3.8-2.4T-A95B): the shapes the
kernels are built for, at TP8 (the only TP size they serve), so every shape
below is a compile-time constant of a kernel.

The routed experts and the online-MXFP4 linears (in_proj_qkvz, out_proj)
hold the layouts in ``mx_layout``.
"""

import torch

from vllm.models.qwen3_5.amd.specialized_decode.mono import mx_layout as QL

BLOCKS = 256  # CTAs a launch, all resident at once: one a CU
# the rows a launch serves: a step of more rows runs stock
MAX_TOKENS = 16
ROWS = QL.ROWS  # a weight row group
TP = 8

# the model (``Qwen3_5MoeTextConfig``)
HIDDEN = 8192
HD = 128  # GDN key and value head dim
CONV_W = 4
E = 512
TOPK = 10
MOE_I = 2048  # moe_intermediate_size
ATTN_HEADS = 64
ATTN_HD = 256
# the text config's fields the kernels are written for, and their values
SHAPE_NAMES = (
    "hidden_size",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "num_attention_heads",
    "head_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "hidden_act",
)
SHAPES = (
    HIDDEN,
    16,
    128,
    HD,
    HD,
    CONV_W,
    ATTN_HEADS,
    ATTN_HD,
    E,
    TOPK,
    MOE_I,
    MOE_I,
    "silu",
)
# decoder layers: a kernel's mailbox tag slots (``moe.TAG_SLOTS / 2``)
MAX_LAYERS = 128
# the GDN states the GDN block updates in place
CONV_DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32

# a rank's shapes
NK = 16 // TP  # GDN key heads: 2
NV = 128 // TP  # GDN value heads: 16
KEY = NK * HD
VAL = NV * HD
QKVZ = 2 * KEY + 2 * VAL  # in_proj_qkvz rows: [q | k | v | z]
BA = 2 * NV  # in_proj_ba rows: [b | a]
CONV = 2 * KEY + VAL  # conv channels: q | k | v
CORE = NV * HD  # out_proj / o_proj K (GDN: NV HD; attention: heads HD)
RI = MOE_I // TP  # routed intermediate a rank holds: 256
RIP = RI  # as the weights hold it (a multiple of the backend's 128)
EPR = E // TP  # router logits a rank computes: 64
MOE = QL.MoeDims(HIDDEN, RI, RIP)
GDN = QL.GdnDims(NK, NV, QKVZ)

assert ATTN_HEADS // TP * ATTN_HD == CORE and RIP % 128 == 0


def need(ok: bool, what: str) -> None:
    """A launch argument check that ``python -O`` keeps: the kernels read
    out-of-bounds memory silently where a fault would be wanted."""
    if not ok:
        raise ValueError(f"Qwen3.8 specialized decode: {what}")
