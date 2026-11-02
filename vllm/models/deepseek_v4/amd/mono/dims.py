# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Pro's widths as the mono kernels read them, and ``Dims``, a TP
rank's share of them with every count the stages derive from it."""

from dataclasses import dataclass

from vllm.models.deepseek_v4.amd.mono.common.plan import Shard, cdiv, padded

HIDDEN = 7168
Q_RANK = 1536
N_HEADS = 128
O_GROUPS = 16
INDEX_TOPK = 1024  # a CSA token's top-k compressed keys
EXPERTS = 384
TOPK = 6
MOE_INTER = 3072  # a routed expert's intermediate width
SHARED_INTER = 3072
ROUTED_SCALE = 2.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-6  # every RMSNorm's (and the mHC seams')
HC_SINKHORN = 20
HC_EPS = 1e-6
HC_POST_MULT = 2.0
# a step's rows at most: 8 requests x (1 + 6 DSpark drafts) and spare
MAX_TOKENS = 64
# the step width from which the all-reduces take two hops (``stages.allreduce``)
TWO_HOP_ROWS = 42

HC = 4  # the mHC seams' streams
HEAD_DIM = 512
ROPE = 64
NOPE = HEAD_DIM - ROPE
HALF = ROPE // 2
KV_DIM = HEAD_DIM
O_RANK = 1024
WINDOW = 128
EXPERT_PAD = 128  # the loader pads a routed expert's rank share to this
ROWS = 16  # GEMV rows a task
TILE = 16  # tokens a GEMV pass (an MFMA's N)
HEAD_TILE = 16  # heads an attention MFMA takes
WOA_ROWS = 32  # a wo_a task's rows: one MX group of its output
UG_PART = 32  # intermediate columns an ug task part: one FP4 group
WQKV_KPARTS = 2  # wqkv's K split
QKV_ROWS = Q_RANK + KV_DIM
X_WORDS = HIDDEN // 4  # an MXFP8 hidden row's i32 words
X_GROUPS = HIDDEN // 32  # an MXFP8 hidden row's E8M0 groups
WQKV_TASKS = QKV_ROWS // ROWS
WOB_TASKS = HIDDEN // ROWS
DOWN_TASKS = HIDDEN // ROWS


def ld(k: int) -> int:
    """The row stride of a row-major FP8 weight of K = ``k`` (bytes): vLLM pads
    the rows of a multiple of 512 bytes by 256 (``VLLM_ROCM_FP8_PADDING``)."""
    return k + 256 if k % 512 == 0 else k


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of the model's widths."""

    tp: int
    # a token's compressed keys at most (0: the CSA top-k): an HCA layer's
    # attention over every compressed row of a long context reads more
    topk: int = 0

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(N_HEADS, "attention heads")
        shard.split(O_GROUPS, "output groups")
        shard.split(MOE_INTER, "routed intermediate")
        shard.split(SHARED_INTER, "shared intermediate")

    # ------------------------------------------------------------ attention
    @property
    def heads(self) -> int:
        return N_HEADS // self.tp

    @property
    def head_tiles(self) -> int:
        return self.heads // HEAD_TILE

    @property
    def groups(self) -> int:
        return O_GROUPS // self.tp

    @property
    def group_k(self) -> int:
        """wo_a's K a group: its heads' outputs."""
        return self.heads // self.groups * HEAD_DIM

    @property
    def o_rows(self) -> int:
        return self.groups * O_RANK

    @property
    def q_rows(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def wqb_tasks(self) -> int:
        return self.q_rows // ROWS

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def topk_width(self) -> int:
        """A token's compressed keys at most."""
        return self.topk or INDEX_TOPK

    @property
    def keys(self) -> int:
        """A token's attention keys at most: its top-k, then its window."""
        return self.topk_width + WINDOW

    # ------------------------------------------------------------ MoE
    @property
    def inter_real(self) -> int:
        """A routed expert's real intermediate width on this rank."""
        return MOE_INTER // self.tp

    @property
    def inter(self) -> int:
        """... padded as the loader pads it (the rest zero weights)."""
        return padded(self.inter_real, EXPERT_PAD)

    @property
    def sh_inter(self) -> int:
        """The shared expert's intermediate width on this rank."""
        return SHARED_INTER // self.tp

    @property
    def shared_tasks(self) -> int:
        return self.sh_inter // ROWS

    @property
    def mid_words(self) -> int:
        """MXFP8 words of a pick's routed intermediate."""
        return self.inter // 4

    @property
    def down_scale_cols(self) -> int:
        """w2_s's e8m0 columns: inter / 32, padded to a multiple of 8
        (aiter ``shuffle_scale``)."""
        return padded(self.inter // 32, 8)


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


__all__ = ["Dims", "cdiv", "ld", "tile_rows", "token_tiles"]
