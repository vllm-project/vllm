# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MonoKernel geometry (``amd/GLM-5.3-Flash-Quark-MXFP4``)."""

from dataclasses import dataclass

HIDDEN = 4096
N_EXPERTS = 288
TOP_K = 8
# the fused shared expert is the last expert slot of the routed tensor
SHARED_EXPERT = N_EXPERTS
MOE_SLOTS = TOP_K + 1
# router candidates per lane: experts lane + 64 i, i < ROUTER_CANDIDATES
ROUTER_CANDIDATES = (N_EXPERTS + 63) // 64
GLOBAL_INTER = 2048
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0
EPS = 1e-5
SCALE_BM = 128
FP8_MAX = 448.0
HC_MULT = 4
HC_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_ITERS = 20
# KDA linear attention
KDA_HEADS = 64
KDA_DIM = 128
KDA_PROJ = KDA_HEADS * KDA_DIM
CONV_K = 4
KDA_LOWER_BOUND = -5.0
L2_EPS = 1e-6
O_NORM_EPS = 1e-5
TP_SIZES = (2, 4, 8)
STEP_TOKENS = (1, 2, 4, 8)


@dataclass(frozen=True)
class FlashShard:
    """One TP rank's share of the FFN: every routed and the shared expert hold
    1/TP of the intermediate width."""

    tp: int

    def __post_init__(self) -> None:
        if self.tp not in TP_SIZES:
            raise ValueError(
                f"GLM-5.3-Flash tensor parallel size must be one of {TP_SIZES}"
            )

    @property
    def inter(self) -> int:
        return GLOBAL_INTER // self.tp

    @property
    def kda_heads(self) -> int:
        return KDA_HEADS // self.tp

    @property
    def kda_proj(self) -> int:
        return KDA_PROJ // self.tp

    @property
    def kda_in_rows(self) -> int:
        """Rows of this rank's merged q|k|v|beta|f_a|g_a projection."""
        return 3 * self.kda_proj + self.kda_heads + 2 * KDA_DIM
