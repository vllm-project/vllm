# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next (``qwen4_exp``): the shapes the kernels are built for at
a TP size. A ``Layout`` is a build key's field, so every shape below is a
compile-time constant of a kernel.

The routed experts and the online-MXFP4 linears (in_proj_qkvz, out_proj)
hold the layouts in ``mx_layout``, the routed intermediate padded from RI to
RIP (``Layout.moe``).
"""

from dataclasses import dataclass

import torch

from vllm.models.qwen4_exp.amd.specialized_decode.mono import mx_layout as QL

BLOCKS = 256  # CTAs a launch, all resident at once: one a CU
# the rows a launch serves: a step of more rows runs stock (at 17..24 rows the
# MoE block spills registers and stock is faster)
MAX_TOKENS = 16
ROWS = QL.ROWS  # a weight row group
TPS = (2,)  # the TP sizes the kernels are built for

# the model (``Qwen4ExpTextConfig``)
HIDDEN = 2560
HC = 4  # hyper-connection streams
HH = HC * HIDDEN  # the multi-stream state's width
LORA = 320  # hc_lowrank
INJ = HC  # injection logits
DOWN_ROWS = 336  # down + inject rows: 320 + 4 + 12 pad (16-row aligned)
HD = 128  # GDN key and value head dim
CONV_W = 4
E = 512
TOPK = 10
MOE_I = 640  # moe_intermediate_size
QSA_HEADS = 24
QSA_HD = 256
# the text config's fields the kernels are written for, and their values
SHAPE_NAMES = (
    "hidden_size",
    "hc_count",
    "hc_lowrank",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "hidden_act",
)
SHAPES = (HIDDEN, HC, LORA, 16, 48, HD, HD, CONV_W, E, TOPK, MOE_I, MOE_I, "silu")
# decoder layers: a block kind's mailbox tag slots (``moe.TAG_SLOTS / 4``)
MAX_LAYERS = 64
# the GDN states the GDN block updates in place
CONV_DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32


def need(ok: bool, what: str) -> None:
    """A launch argument check that ``python -O`` keeps: the kernels read
    out-of-bounds memory silently where a fault would be wanted."""
    if not ok:
        raise ValueError(f"Qwen3.8-Flash-Next specialized decode: {what}")


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@dataclass(frozen=True)
class Layout:
    """The per-rank shapes at TP size ``tp``."""

    tp: int

    def __post_init__(self):
        need(self.tp in TPS, f"tensor parallel size {self.tp}, built for {TPS}")

    # GDN (linear attention): 16 key / 48 value heads
    @property
    def NK(self) -> int:
        return 16 // self.tp

    @property
    def NV(self) -> int:
        return 48 // self.tp

    @property
    def KEY(self) -> int:
        return self.NK * HD

    @property
    def VAL(self) -> int:
        return self.NV * HD

    @property
    def QKVZ(self) -> int:
        """in_proj_qkvz rows: [q | k | v | z]."""
        return 2 * self.KEY + 2 * self.VAL

    @property
    def BA(self) -> int:
        """in_proj_ba rows: [b | a]."""
        return 2 * self.NV

    @property
    def CONV(self) -> int:
        """Conv channels: q | k | v."""
        return 2 * self.KEY + self.VAL

    @property
    def CORE(self) -> int:
        """out_proj / o_proj K a rank holds (GDN: NV HD; QSA: heads HD)."""
        assert QSA_HEADS // self.tp * QSA_HD == self.NV * HD
        return self.NV * HD

    # MoE: experts TP-sharded (not EP); the router replicated
    @property
    def RI(self) -> int:
        """Routed intermediate a rank holds."""
        return MOE_I // self.tp

    @property
    def RIP(self) -> int:
        """RI as the weights hold it: padded to the backend's 128."""
        return _round_up(self.RI, 128)

    @property
    def moe(self) -> QL.MoeDims:
        return QL.MoeDims(HIDDEN, self.RI, self.RIP)

    @property
    def gdn(self) -> QL.GdnDims:
        return QL.GdnDims(self.NK, self.NV, self.QKVZ)
