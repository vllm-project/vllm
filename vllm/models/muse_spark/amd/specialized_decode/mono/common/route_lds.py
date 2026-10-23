# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K2 route table in LDS (``RouteLds``): a step's top-8 picks and the union
of their experts, every CTA's own copy, live from ``load_route`` to the owner
reduce.

    route   i32[MT 16]    t 16 + k: expert id (-1: dead pick); t 16 + 8 + k: the
                          fp32 bits of its route weight (0 on a pad row)
    flag    i32[256]      expert -> 0 / 1, then its union slot
    expert  i32[MAX_U]    union slot -> expert id (zeroed every launch)
    kof     i32[MAX_U 8]  slot 8 + t -> row t's top-k index k of the slot's
                          expert, -1 when row t did not pick it
    scan    i32[WAVES]    ``block_excl_scan``'s wave partials
    tot     f32[MT]       sum of squares of pooled row t (``load_xq``)

A pick is live iff ``0 <= id < EXPERTS``: pad rows carry -1, and a garbage id
(a poll that timed out returns whatever the pair held) is dead too, so no table
store runs out of bounds. Expert ids are clamped (``clamp_expert``) at every
address use.
"""

import flydsl.expr as fx

from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    EXPERTS,
    ROUTE_LDS,
    TOPK,
)

FIELDS = ("route", "flag", "expert", "kof", "scan", "tot")
assert set(FIELDS) == set(ROUTE_LDS), ROUTE_LDS
# words a field, by name (names, not subscripts, in the annotations below)
ROUTE_W, FLAG_W, EXPERT_W, KOF_W, SCAN_W, TOT_W = (ROUTE_LDS[f] for f in FIELDS)


@fx.struct
class RouteLds:
    route: fx.Array[fx.Int32, ROUTE_W, 16]
    flag: fx.Array[fx.Int32, FLAG_W, 16]
    expert: fx.Array[fx.Int32, EXPERT_W, 16]
    kof: fx.Array[fx.Int32, KOF_W, 16]
    scan: fx.Array[fx.Int32, SCAN_W, 16]
    tot: fx.Array[fx.Float32, TOT_W, 16]


def route_ptrs(rl) -> dict:
    """Field -> LDS pointer of a peeked ``RouteLds`` (the context's ``rt``)."""
    return {name: getattr(rl, name).ptr for name in FIELDS}


def id_index(t, k):
    """``route`` word of row t's pick k's expert id (``spec.route_id_pair``)."""
    return t * 16 + k


def weight_index(t, k):
    """``route`` word of row t's pick k's weight bits (``spec.route_weight_pair``)."""
    return t * 16 + TOPK + k


def load_weight(rt, t, k):
    """Row t's pick k's route weight, an fp32 (stored as its bits)."""
    return fx.ptr_load(rt["route"] + weight_index(t, k)).bitcast(fx.Float32)


def live(e):
    """Whether expert id ``e`` names a pick (pads: -1; garbage: out of range)."""
    return (e >= 0) & (e < EXPERTS)


def clamp_expert(e):
    """``e`` clamped into [0, EXPERTS): an in-bounds address for any id."""
    return fx.max(fx.min(e, fx.Int32(EXPERTS - 1)), fx.Int32(0))
