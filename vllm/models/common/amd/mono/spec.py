# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a model is to the mono kernels: ``MonoSpec``, one model's decoder layer
(its widths, numerics and seam variant), and ``Dims``, a TP rank's share of it
with every count the stages derive from it. A build key carries the spec, so a
kernel reads its model from the key alone; a new model of the family is a new
``MonoSpec`` (README, "Adding a model")."""

from dataclasses import dataclass

from vllm.models.common.amd.mono.common.plan import Shard, cdiv, padded

# the family's fixed widths: the mHC seams' streams, the MLA head, the KV record
HC = 4
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


@dataclass(frozen=True)
class MonoSpec:
    """One model's decoder layer, as the kernels read it (scalars only: a build
    key carries it into FlyDSL's JIT cache key)."""

    name: str  # the kernels' symbol stem
    hidden: int
    q_rank: int
    n_heads: int
    o_groups: int
    index_topk: int  # a token's top-k compressed keys at most
    experts: int
    topk: int
    moe_inter: int  # a routed expert's intermediate width
    shared_inter: int
    routed_scale: float
    swiglu_limit: float
    norm_eps: float  # every RMSNorm's (and the mHC seams')
    # the mHC seam collapses the layer input with the previous seam's pre-mix
    # (V4.1) rather than with its own (V4)
    delayed_seam: bool
    max_tokens: int  # a step's rows at most
    # wqkv, wo_b and the shared expert's gate_up in aiter's (16, 16) shuffle
    # (vLLM preshuffles them in place for its block-scaled GEMMs), else row-major
    preshuffled: bool = False
    # a weightless RMSNorm over each query head ahead of the attention (V4)
    q_head_norm: bool = False
    # vLLM pads a row-major FP8 weight's rows of a multiple of 512 bytes by 256
    # (``VLLM_ROCM_FP8_PADDING``: its block-scaled linears), ``Dims.ld``
    padded_rows: bool = False
    # the step width from which the all-reduces take two hops
    # (``stages.allreduce``; 0: one hop always)
    two_hop_rows: int = 0
    hc_sinkhorn: int = 20
    hc_eps: float = 1e-6
    hc_post_mult: float = 2.0


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of ``spec``: every count a stage derives from the
    model's widths and the TP size."""

    spec: MonoSpec
    tp: int
    # a token's compressed keys at most (0: the model's top-k): a build's
    # attention over every compressed row of a long context reads more
    topk: int = 0

    def __post_init__(self):
        sp, shard = self.spec, Shard(self.tp)
        shard.split(sp.n_heads, "attention heads")
        shard.split(sp.o_groups, "output groups")
        shard.split(sp.moe_inter, "routed intermediate")
        shard.split(sp.shared_inter, "shared intermediate")

    # ------------------------------------------------------------ model
    @property
    def hidden(self) -> int:
        return self.spec.hidden

    @property
    def q_rank(self) -> int:
        return self.spec.q_rank

    @property
    def qkv_rows(self) -> int:
        return self.spec.q_rank + KV_DIM

    @property
    def x_words(self) -> int:
        """An MXFP8 hidden row's i32 words."""
        return self.spec.hidden // 4

    @property
    def x_groups(self) -> int:
        """An MXFP8 hidden row's E8M0 groups."""
        return self.spec.hidden // 32

    def ld(self, k: int) -> int:
        """The row stride of a row-major FP8 weight of K = ``k`` (bytes)."""
        return k + 256 if self.spec.padded_rows and k % 512 == 0 else k

    # ------------------------------------------------------------ attention
    @property
    def heads(self) -> int:
        return self.spec.n_heads // self.tp

    @property
    def head_tiles(self) -> int:
        return self.heads // HEAD_TILE

    @property
    def groups(self) -> int:
        return self.spec.o_groups // self.tp

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
    def wqkv_tasks(self) -> int:
        return self.qkv_rows // ROWS

    @property
    def wqb_tasks(self) -> int:
        return self.q_rows // ROWS

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def wob_tasks(self) -> int:
        return self.spec.hidden // ROWS

    @property
    def topk_width(self) -> int:
        """A token's compressed keys at most."""
        return self.topk or self.spec.index_topk

    @property
    def keys(self) -> int:
        """A token's attention keys at most: its top-k, then its window."""
        return self.topk_width + WINDOW

    # ------------------------------------------------------------ MoE
    @property
    def inter_real(self) -> int:
        """A routed expert's real intermediate width on this rank."""
        return self.spec.moe_inter // self.tp

    @property
    def inter(self) -> int:
        """... padded as the loader pads it (the rest zero weights)."""
        return padded(self.inter_real, EXPERT_PAD)

    @property
    def sh_inter(self) -> int:
        """The shared expert's intermediate width on this rank."""
        return self.spec.shared_inter // self.tp

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

    @property
    def down_tasks(self) -> int:
        return self.spec.hidden // ROWS


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


__all__ = [
    "EXPERT_PAD",
    "HALF",
    "HC",
    "HEAD_DIM",
    "HEAD_TILE",
    "KV_DIM",
    "NOPE",
    "O_RANK",
    "ROPE",
    "ROWS",
    "TILE",
    "UG_PART",
    "WINDOW",
    "WOA_ROWS",
    "WQKV_KPARTS",
    "Dims",
    "MonoSpec",
    "cdiv",
    "tile_rows",
    "token_tiles",
]
