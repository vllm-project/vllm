# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/portion.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Stage masks for the in-place portion ablation (TEP8 monokernel ablation, Task 1).

Build-key field ``only`` (env MK_ONLY for K3, MK5_ONLY for K5 HA / HM / HB):
  ""      every stage (the shipped kernel; the build is byte-identical to the unmasked one)
  "none"  no stage (the "nothing active" baseline: launch + persistent-kernel overhead)
  "a,b"   only these stages (comma list), plus the stages they need for CTA-local state (DEPS).

A turned-off stage does not run at all: no loads, no compute, no waits, no stores. Its outputs come
pre-staged from a full run of the same launch (same epoch and layer, so the tags match); see
bench_portion.py. The active stages keep their CTA set, placement and code.

Stage names follow the stage stamps (MK_TIMELINE):
  K3  (one kernel):  O AB A route h gate_up down final ar post G QKV finish
  K5  HA:            ng O5 AB5 A5 route_h
      HM:            m_setup m_gate_up m_down m_rstd m_final
      HB:            ar5 post5 G5 QKV5 finish5 gather5
"""

K3_STAGES = (
    "O",
    "AB",
    "A",
    "route",
    "h",
    "gate_up",
    "down",
    "final",
    "ar",
    "post",
    "G",
    "QKV",
    "finish",
)
K3_DEPS = {  # CTA-local state (LDS / registers) a stage reads from an earlier stage of the same CTA
    "gate_up": (
        "route",
        "h",
    ),  # routing tables + task range (route), the normalized h rows in LDS (h)
    "down": ("route",),  # routing tables, task range, picks
    "final": ("route",),  # routing ids / weights / pick index
}
K5A_STAGES = ("ng", "O5", "AB5", "A5", "route_h")
HM_STAGES = ("m_setup", "m_gate_up", "m_down", "m_rstd", "m_final")
K5B_STAGES = ("ar5", "post5", "G5", "QKV5", "finish5", "gather5")
K5_DEPS = {
    "m_gate_up": (
        "m_setup",
    ),  # the virtual-expert tables, the CTA's task range and picks
    "m_down": ("m_setup",),
    "m_rstd": ("m_setup",),
    "m_final": ("m_setup",),
}


def stage_set(only, stages, deps):
    """None (all stages) or the frozenset of active stages of this kernel (deps included)."""
    if only == "":
        return None
    names = [s for s in only.split(",") if s and s != "none"]
    known = set(K3_STAGES) | set(K5A_STAGES) | set(HM_STAGES) | set(K5B_STAGES)
    bad = [s for s in names if s not in known]
    assert not bad, (
        f"unknown stage(s) {bad}; K3 {K3_STAGES}, K5 {K5A_STAGES + HM_STAGES + K5B_STAGES}"
    )
    act = set(names)
    for s in list(act):
        act |= set(deps.get(s, ()))
    return frozenset(s for s in stages if s in act)


def name_tag(only):
    """Kernel-name suffix of a mask ("" for the full kernel, so its name and code stay unchanged)."""
    if only == "":
        return ""
    import hashlib

    return "_only" + hashlib.sha1(only.encode()).hexdigest()[:8]


# The linear stage order of a bucket (prefix masks in this order run the kernel causally) and its kernels.
ORDER = {"K3": K3_STAGES, "K5": K5A_STAGES + HM_STAGES + K5B_STAGES}
KERNEL_OF = {
    **{s: "K3" for s in K3_STAGES},
    **{s: "HA" for s in K5A_STAGES},
    **{s: "HM" for s in HM_STAGES},
    **{s: "HB" for s in K5B_STAGES},
}
# stage -> (start stamp, end stamp) of its MK_TIMELINE region in its kernel ([start, end) in address order;
# the same slots as tools/att_stages.py)
SPANS = {
    "K3": {
        "O": (0, 12),
        "AB": (12, 13),
        "A": (13, 2),
        "route": (2, 3),
        "h": (3, 4),
        "gate_up": (4, 5),
        "down": (5, 7),
        "final": (7, 8),
        "ar": (8, 9),
        "post": (9, 10),
        "G": (10, 11),
        "QKV": (11, 14),
        "finish": (14, 15),
    },
    "HA": {"ng": (0, 1), "O5": (1, 2), "AB5": (2, 3), "A5": (3, 4), "route_h": (4, 5)},
    "HM": {
        "m_setup": (0, 1),
        "m_gate_up": (1, 2),
        "m_down": (2, 4),
        "m_rstd": (4, 5),
        "m_final": (5, 6),
    },
    "HB": {
        "ar5": (6, 7),
        "post5": (7, 8),
        "G5": (8, 9),
        "QKV5": (9, 10),
        "finish5": (10, 11),
        "gather5": (11, 12),
    },
}


def active(only, bucket):
    """The stages of a bucket that run with mask `only`, in bucket order (all of them for "")."""
    order = ORDER[bucket]
    deps = K3_DEPS if bucket == "K3" else K5_DEPS
    s = stage_set(only, order, deps)
    return list(order) if s is None else [x for x in order if x in s]


# CTA set of each stage at W = 8 (K3 S8, K5 S32), from the code (the body and the stage functions' predicates).
# All 256 CTAs reach every stage (persistent kernels); "x of 256" = the CTAs that do the stage's work.
CTAS = {
    "O": "256 (o_proj rows 32b..; partial -> block owner b // 32)",
    "AB": "256 (block owners b // 32 == rank reduce + all-gather; every CTA polls its block)",
    "A": "256 (router group b // KQ, K part b % KQ)",
    "route": "256 (every CTA routes all tokens, redundant)",
    "h": "256 (h rows of all tokens into LDS)",
    "gate_up": "256 (virtual-expert tasks [b nu, (b+1) nu))",
    "down": "256 (same tasks as gate_up)",
    "final": "256 (16-column block b; push to block owner)",
    "ar": "32 of 256 (b // 32 == rank reduce; all-gather to every rank)",
    "post": "256 (tile b // KSP, K part b % KSP; S = 8 reducer CTAs 248..255 for rstd)",
    "G": "256 (hidden columns; per-token reduction)",
    "QKV": "256 (tile group b // 8, K eighth b % 8; NQT = 5 tiles)",
    "finish": "20 of 256 (head hh = b < HQ + 2 HKV = 16 + 4)",
    "ng": "256 (8 lanes a (token, head))",
    "O5": "256 (o_proj rows 32b..; partial -> owner b // 32)",
    "AB5": "4 of 256 (b < TS = S/W: one token row each)",
    "A5": "256 (router expert b // KQ, K part b % KQ)",
    "route_h": "32 of 256 (b < S: token b)",
    "m_setup": "256 (every CTA builds the virtual-expert tables)",
    "m_gate_up": "256 (tasks [b V, (b+1) V))",
    "m_down": "256 (same tasks)",
    "m_rstd": "256 (picks b, b + 256, ..)",
    "m_final": "256 (16-column block b; push to block owner)",
    "ar5": "32 of 256 (b // 32 == rank)",
    "post5": "256 (tile b // KSP, K part b % KSP)",
    "G5": "4 of 256 (b < TS)",
    "QKV5": "256",
    "finish5": "20 of 256 (b < HQ + 2 HKV)",
    "gather5": "32 of 256 (b < nrows)",
}

# The CTAs that do a stage's work (integer count and index set; W = 8, rows = 8 for K3, 32 for K5).
CTAS_SUB = {
    "ar": 32,
    "ar5": 32,
    "AB5": "TS",
    "G5": "TS",
    "route_h": "S",
    "gather5": "S",
    "finish": 20,
    "finish5": 20,
}


def cta_ids(stage, rank, rows, W=8, blocks=256):
    """Indices of the CTAs that do stage's work (the timeline summaries use only these)."""
    n = CTAS_SUB.get(stage)
    if n is None:
        return list(range(blocks))
    if stage in ("ar", "ar5"):
        c = blocks // W
        return list(range(c * rank, c * rank + c))
    if n == "TS":
        return list(range(rows // W))
    if n == "S":
        return list(range(rows))
    return list(range(n))


def ctas_n(stage, rows, W=8):
    return len(cta_ids(stage, 0, rows, W))


# World-1 validity rule: the stages that fail the world-1 check (tests/check_portion.py --world 1; measured on
# all 8 ranks, L1 and L4). check_portion skips them at world 1 by default.
CROSS_RANK = {
    "O",
    "AB",
    "A",
    "final",
    "ar",
    "post",
    "O5",
    "AB5",
    "A5",
    "m_final",
    "ar5",
    "post5",
    "G5",
}
# Upper-bound rule (separate): the stages that exchange with or wait on peers, i.e. whose code writes to another
# rank's buffer or polls the own peer buffer that other ranks write (system-scope mailbox ops on the peer layout;
# tests/check_upper_bound.py derives this set from the source). At world 8 their in-place time can include the
# start skew of the ranks (host barrier, no device gate), so the runner marks them upper_bound.
PEER_ACCESS = {
    "O",
    "AB",
    "A",
    "route",
    "h",
    "final",
    "ar",
    "post",
    "G",
    "O5",
    "AB5",
    "A5",
    "route_h",
    "m_final",
    "ar5",
    "post5",
    "G5",
    "QKV5",
    "gather5",
}


def upper_bound(active, world):
    """(upper_bound, the stages that cause it) of a run with these active stages."""
    st = sorted(set(active) & PEER_ACCESS)
    return bool(st) and world > 1, st


# CTA-local dependencies that a masked run includes, and the mask Task 3 subtracts to isolate the stage
DEP_SUBTRACT = {
    "gate_up": "route,h",
    "down": "route",
    "final": "route",
    "m_gate_up": "m_setup",
    "m_down": "m_setup",
    "m_rstd": "m_setup",
    "m_final": "m_setup",
}

# Timeline stamps inside a stage's code (absent when that stage is off) and the program order of the stamps of
# each kernel. A span whose start stamp is absent starts at the nearest earlier stamp that exists (the same
# fallback as the ISA region cut, ablation/isa_identity.py): K3 route starts at 13 when A is off.
STAMP_OWNER = {
    "K3": {1: "A", 2: "A", 6: "down"},
    "HA": {
        24: "O5",
        25: "O5",
        15: "AB5",
        16: "AB5",
        13: "A5",
        14: "A5",
        18: "A5",
        22: "A5",
        19: "A5",
        26: "A5",
        27: "A5",
        28: "A5",
        20: "A5",
        21: "A5",
    },
    "HM": {},
    "HB": {17: "G5", 23: "G5", 14: "QKV5"},
}
STAMP_ORDER = {
    "K3": [0, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15],
    "HA": [0, 1, 24, 25, 2, 15, 16, 3, 13, 14, 18, 22, 19, 26, 27, 28, 20, 21, 4, 5],
    "HM": [0, 1, 2, 3, 4, 5, 6],
    "HB": [0, 6, 7, 8, 17, 23, 9, 14, 10, 11, 12],
}


def span(kern, stage, active):
    """(start stamp, end stamp) of stage's timeline span in a build whose active stages are `active`."""
    a, b = SPANS[kern][stage]
    own, order = STAMP_OWNER[kern], STAMP_ORDER[kern]
    while a in own and own[a] not in active:
        a = order[order.index(a) - 1]
    return a, b
