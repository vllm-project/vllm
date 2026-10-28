# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/layout.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where the mono layer kernel (K4) puts its tasks and data, per decode token count.

Plain Python, evaluated at build time: which CTA runs each stage's tasks, the
scratch and peer-symmetric regions, and the LDS row pitches. Kept free of FlyDSL
so it is testable without a GPU.
"""

from .config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_SPARSE_KEYS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    PAGE16,
    SPARSE_BLOCK,
    TOP_K,
    TOPK_BLOCKS,
)
from .support.plan.check import (
    RegionDecl,
)
from .support.plan.execution import (
    BLOCKS,
    WAVES,
)
from .support.plan.layout import (
    DIAG_WORDS,
    PAIR_BYTES,
    pair_layout,
)
from .support.plan.trace import (
    Space,
)

H = LOCAL_Q_HEADS
SPLIT_KEYS = 256  # one gluon context partition per split task
N_SPLIT = MAX_SPARSE_KEYS // SPLIT_KEYS
O_K = H * HEAD_DIM
O_ROWS = 32  # hidden rows per o_proj / attention-reduce task
# from this many tokens the attention all-reduce is a reduce-scatter to the row
# group's owner rank (t % npes) plus an all-gather of the sums: half the bytes over
# each link for one more hop (smaller batches are latency-bound: one hop wins)
AG_RS_FROM = 12
N_O = HIDDEN // O_ROWS
PAGES_PER_BLOCK = SPARSE_BLOCK // PAGE16
N_ROUTER = N_ROUTED // WAVES  # one expert per wave
UG_ROWS = 16
UG_PER_SLOT = INTER // UG_ROWS
SH_ROWS = 8  # mid rows per shared-expert up / gate task (a half task)
SH_TASKS = INTER // SH_ROWS
DN_ROWS = 32
N_DN = HIDDEN // DN_ROWS
# The router logits are written SCORE_COPIES times and CTA b polls copy
# b % SCORE_COPIES: 240 expert CTAs spinning on one 1 KB region serialize on its
# lines. K4 on 4 GPUs 37.3 -> 35.9 us; copies of xn / mid measured neutral and of
# the split outputs +0.3 us (the split CTAs pay the extra stores).
SCORE_COPIES = 8


def _layout(items, align=256):
    """``pair_layout``'s byte offsets, each region on an ``align`` boundary, and
    ``_bytes``: the whole, a multiple of ``align``."""
    regions = pair_layout(items, align)
    end = max(o + n for o, n in regions.values())
    return {name: o for name, (o, _) in regions.items()} | {
        "_bytes": (end + align - 1) // align * align
    }


# plain words, not pairs: the selected pages then the context length, as the
# original selector emits them (read by the tests); one row per token
SPARSE_TABLE_WORDS = TOPK_BLOCKS * PAGES_PER_BLOCK + 2
# a context past THREADS blocks, one-head build: each split task's TOPK_BLOCKS
# best (key, block) (indexer context parallelism: the peers' sel_cp)
SEL_LONG_PAIRS = N_SPLIT * TOPK_BLOCKS * 2

# every region holds MAX_TOKENS rows (token k at k * the one-token size)
SCRATCH = _layout(
    [
        # indexer block scores, token k's block b at pair k * MAX_INDEX_BLOCKS + b
        ("iscore", MAX_INDEX_BLOCKS * MAX_TOKENS),
        ("sparse_table", SPARSE_TABLE_WORDS // 2 * MAX_TOKENS),
        ("sp_o", N_SPLIT * H * HEAD_DIM // 2 * MAX_TOKENS),  # packed bf16 pairs
        ("sp_m", N_SPLIT * H * MAX_TOKENS),
        ("sp_l", N_SPLIT * H * MAX_TOKENS),
        ("attnq", O_K // 4 * MAX_TOKENS // 2),  # merged attention, plain FP8 words
        ("attnq_s", MAX_TOKENS),  # and its per-token scale
        # the finished attention all-reduce + residual, bf16 pairs (o tasks -> router)
        ("a", HIDDEN // 2 * MAX_TOKENS),
        ("xn8", HIDDEN // 4 * MAX_TOKENS),  # MXFP8 xn, 4 fp8 per word
        ("xsc", HIDDEN // 32 * MAX_TOKENS),  # and its per-1x32 E8M0 scales
        # copy c, token k at pairs [(c * MAX_TOKENS + k) * N_ROUTED, ..)
        # per expert (routing key, sigmoid) pairs, see route_key
        ("scores", 2 * N_ROUTED * MAX_TOKENS * SCORE_COPIES),
        ("mid", MOE_SLOTS * INTER * MAX_TOKENS),
        # routing record read by the tests: token k's expert ids, then weights
        ("sel", 2 * MOE_SLOTS * MAX_TOKENS),
        # token k's split task p's candidates: rank r at pairs 2 ((k N_SPLIT + p)
        # TOPK_BLOCKS + r) + (0 key, 1 block)
        ("sel_long", SEL_LONG_PAIRS * MAX_TOKENS),
        # wide: the MXFP8 xn and scales as plain words (every token's row), and a
        # done flag per router task
        ("xn8p", HIDDEN // 4 * MAX_TOKENS // 2),
        ("xscp", HIDDEN // 32 * MAX_TOKENS // 2),
        ("xndone", N_ROUTER * MAX_TOKENS),
        # and the mid rows as MXFP8 words, their E8M0 scales a word each (the down
        # stage's LDS layout), a done flag per up / gate task pair
        ("mid8p", MOE_SLOTS * INTER * MAX_TOKENS // 8),
        ("midsp", MOE_SLOTS * INTER // 32 * MAX_TOKENS // 2),
        ("ugdone", (TOP_K * MAX_TOKENS + 1) * UG_PER_SLOT // 2),
        # a debug build's wait records, one a CTA; last, so the regions above
        # keep their offsets in every build
        ("diag", BLOCKS * DIAG_WORDS * 4 // PAIR_BYTES),
    ]
)
SCRATCH_BYTES = SCRATCH["_bytes"]


def sym_layout(npes: int):
    """Per-rank symmetric buffer: attn / ffn partial regions, [src rank][token][HIDDEN],
    and the peers' index-selection candidates (``sel_cp``).

    K4 pushes a layer's attention partials into every peer's ``attn`` region and
    its FFN partials into the ``ffn`` region, then polls its own. Regions are
    reused by every layer (the mailbox tag tells a layer's pairs from an earlier
    layer's) and no rank can overwrite a peer's unread pairs: to push layer L's
    FFN partials a rank must have passed layer L's attention reduce, which needs
    every peer's layer-L attention push, and a peer only pushes that after it
    finished reading layer L-1's FFN region. So ranks are at most half a layer
    apart and the two regions double-buffer each other."""
    rows = npes * MAX_TOKENS * HIDDEN // 2
    # a_ag / ffn_ag: the finished attention / FFN all-reduce rows, each written by
    # its owner rank (the reduce-scatter + all-gather form, S >= AG_RS_FROM)
    ag = MAX_TOKENS * HIDDEN // 2
    return _layout(
        [
            ("attn", rows),
            ("ffn", rows),
            ("a_ag", ag),
            ("ffn_ag", ag),
            # indexer context parallelism: token k's candidates for this rank's
            # index head, source rank s's half h's r-th best at pairs
            # 2 ((k N_SPLIT + 2 s + h) TOPK_BLOCKS + r) + (0 key, 1 block)
            ("sel_cp", SEL_LONG_PAIRS * MAX_TOKENS),
        ]
    )


def mailbox_regions(
    tokens: int, index_heads: int, fuse_k1: bool, index_topk: bool = True
) -> list[RegionDecl]:
    """The fused layer kernel's mailbox regions for a build: each one's space and
    the one stage that puts it (``support.plan.check``; the stages are the ones
    the kernel enters, in order: k1.norm (S > 2), k1.gemv, k1.head, k1.score,
    select_long, split, merge (S > 1), o, router, ug, down). Without
    ``index_topk`` (a build reusing a selection) the indexer's regions are gone."""
    S, P = Space.SCRATCH, Space.PEER
    regions = []
    if fuse_k1:
        regions += [
            RegionDecl("k1.qkv", S, "k1.gemv"),
            RegionDecl("k1.hdone", S, "k1.head"),
            # the residual: the norm tasks' past two tokens, else the GEMV tasks'
            RegionDecl("k1.rdone", S, "k1.norm" if tokens > 2 else "k1.gemv"),
        ]
        if index_topk:
            regions.append(RegionDecl("iscore", S, "k1.score"))
        if tokens > 2:
            regions.append(RegionDecl("k1.x8s", S, "k1.norm"))
    if index_topk:
        regions.append(
            RegionDecl("sel_cp", P, "select_long", exchange=True)
            if index_heads > 1
            else RegionDecl("sel_long", S, "select_long", exchange=True)
        )
    regions += [RegionDecl(n, S, "split") for n in ("sp_o", "sp_m", "sp_l")]
    if tokens > 1:
        regions.append(RegionDecl("attnq_s", S, "merge"))
    regions.append(RegionDecl("attn", P, "o", exchange=True))
    regions.append(
        RegionDecl("a_ag", P, "o") if tokens >= AG_RS_FROM else RegionDecl("a", S, "o")
    )
    regions.append(RegionDecl("scores", S, "router"))
    if wide_moe(tokens):
        regions += [RegionDecl("xndone", S, "router"), RegionDecl("ugdone", S, "ug")]
    else:
        regions += [RegionDecl(n, S, "router") for n in ("xn8", "xsc")]
        regions.append(RegionDecl("mid", S, "ug"))
    regions.append(RegionDecl("sel", S, "ug"))  # the routing record (tests)
    regions.append(RegionDecl("ffn", P, "down", exchange=True))
    if tokens >= AG_RS_FROM:
        regions.append(RegionDecl("ffn_ag", P, "down", exchange=True))
    return regions


def stage_bases(tokens: int):
    """Task t of a stage runs on CTA (base + t) % BLOCKS; the merge (S > 1), o and
    down tasks start past the tokens' split CTAs."""
    n_split = N_SPLIT * tokens
    n_merge = tokens if tokens > 1 else 0
    router = (n_split + n_merge + N_O) % BLOCKS
    return {
        "split": 0,
        "merge": n_split,
        "o": n_split + n_merge,
        "router": router,
        "shared": _shared_base(tokens, router),
        "ug": 0,
        "down": n_split,
    }


def ug_ctas_per_token(tokens: int):
    """CTAs per token of the routed up / gate tasks, token k on CTAs k C .. (k + 1)
    C - 1: a CTA stages one token's xn, not all (fused layer, S = 4: 57.9 -> 57.6
    us). None: round-robin -- S < 4 (S = 2: +0.4 us) or S not dividing the grid."""
    return BLOCKS // tokens if tokens >= 4 and BLOCKS % tokens == 0 else None


def ug_tasks_of(cta: int, tokens: int) -> int:
    """Routed up / gate tasks CTA ``cta`` runs."""
    per_token = TOP_K * UG_PER_SLOT
    cpt = ug_ctas_per_token(tokens)
    if cpt is None:
        return (per_token * tokens - cta + BLOCKS - 1) // BLOCKS
    return (per_token - cta % cpt + cpt - 1) // cpt


def _shared_base(tokens: int, router: int) -> int:
    """First of the SH_TASKS consecutive CTAs that run the shared expert's up /
    gate before the routing lands. Ranked by the most routed up / gate tasks a
    window CTA carries (they queue behind it); then, only if that is the most any
    CTA carries, by its router CTAs (they start it late and would delay the last
    routed tasks: S = 4, 4 us); then by its down CTAs (S = 2: 43.1 vs 43.7 us);
    then by its router CTAs anyway."""
    n_router = N_ROUTER * tokens
    n_split = N_SPLIT * tokens
    most = max(ug_tasks_of(c, tokens) for c in range(BLOCKS))

    def cost(base):
        ctas = [(base + j) % BLOCKS for j in range(SH_TASKS)]
        busiest = max(ug_tasks_of(c, tokens) for c in ctas)
        in_router = sum((c - router) % BLOCKS < n_router for c in ctas)
        in_down = sum(n_split <= c < n_split + N_DN for c in ctas)
        return busiest, in_router if busiest == most else 0, in_down, in_router

    return min(range(BLOCKS), key=cost)


def shared_after_routing(tokens: int) -> bool:
    """Every CTA carries the same number of routed up / gate tasks (S = 4): a
    shared-expert CTA routes first and runs its shared half task while its first
    routed task's weights are in flight, instead of holding its routing back
    (S = 4 54.51 -> 54.09 us; S = 2, where the shared window sits on CTAs with one
    task fewer, +0.1)."""
    loads = [ug_tasks_of(c, tokens) for c in range(BLOCKS)]
    return min(loads) == max(loads)


WIDE_FROM = 5  # token counts from here run the expert-grouped MoE


def wide_moe(tokens: int) -> bool:
    """The MoE stage groups its work by expert (every distinct expert's weights
    read once, a B column per token) instead of by (token, slot): past four tokens
    the per-token tasks re-read the experts the tokens share, and their operands
    no longer fit in LDS."""
    return tokens >= WIDE_FROM


# LDS rows of a token's MXFP8 xn and of its E8M0 scales, padded by 16 B: at the
# natural 1536 / 192-word rows the 16 tokens a MFMA B operand gathers sit in one
# bank group (16-way conflicts; ug spent ~3 us a task in them at S = 16)
XN8_ROW = HIDDEN // 4 + 4
XSC_ROW = HIDDEN // 32 + 4
# and the down stage's MXFP8 mid rows / their scales (16 lanes, 16 rows): 196 and
# 28 words put the rows' 16 B reads, and 4 B scale reads, in distinct banks
MID_ROW = INTER // 4 + 4
MIDSC_ROW = INTER // 32 + 4
# and the o stage's FP8 attention rows (512 words: the same 16-way conflict)
O_ROW = O_K // 4 + 4


def pool_words(tokens: int) -> int:
    """LDS words of the activation pool ``x``. Narrow: every token's bf16 xn (the
    largest stage operand). Wide: the largest of the split's PV partials, the up /
    gate stage's MXFP8 xn + scales + up partials + router sigmoids, and the down
    stage's MXFP8 mids + scales + a zero row."""
    if not wide_moe(tokens):
        return HIDDEN // 2 * tokens
    rows = MOE_SLOTS * tokens + 1
    return max(
        WAVES * O_K,
        tokens * (XN8_ROW + XSC_ROW + N_ROUTED) + WAVES * 64 * 4,
        rows * (MID_ROW + MIDSC_ROW),
    )
