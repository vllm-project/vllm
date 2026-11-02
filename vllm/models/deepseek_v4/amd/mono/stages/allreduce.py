# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono layer's in-kernel all-reduce of a bf16 [S, hidden] partial over the
TP ranks (the attention output, the MoE output), into a peer region of each
rank's symmetric buffer, one of two ways:

    one hop   every rank pushes its partial to every rank (PART: [source rank]
              [token][hidden / 2] bf16 pairs); a reader sums the ranks' pairs
              in rank order
    two hops  a rank pushes each column to its owner (``owner``: hidden / tp
              columns a rank); the owner sums them in rank order and pushes
              the sums to every other rank (SUM: [token][hidden / 2] pairs),
              where the readers take them

Two hops carry 2 / tp of one hop's bytes a rank (a quarter at TP8) for one more
crossing: the wide steps of a wide TP group take them (``MonoBuild.two_hop``).
Both sum in rank order in fp32, so they agree bit for bit. Every value travels
as a tagged pair (``common.sync``)."""

from flydsl.expr import range_constexpr

from vllm.models.deepseek_v4.amd.mono.common.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    sum_partials,
    traced,
)
from vllm.models.deepseek_v4.amd.mono.common.sync import preg
from vllm.models.deepseek_v4.amd.mono.dims import HIDDEN, Dims

PAIR_BYTES = 8


def region_bytes(tokens: int, d: Dims) -> int:
    """A reduction's peer region: PART, then SUM (two hops)."""
    return (d.tp + 1) * tokens * HIDDEN // 2 * PAIR_BYTES


def part_pair(d: Dims, src, t, s, col):
    """PART's pair of rank ``src``, token t, columns col, col + 1."""
    return (src * s + t) * (HIDDEN // 2) + col // 2


def sum_pair(d: Dims, t, s, col):
    """SUM's pair of token t, columns col, col + 1 (past PART)."""
    return (d.tp * s + t) * (HIDDEN // 2) + col // 2


def owner(d: Dims, col):
    """The rank that sums column ``col`` on two hops."""
    return col // (HIDDEN // d.tp)


def peer_at(bases, p):
    """Peer ``p``'s base (``p`` traced): the select of the peers' bases."""
    base = bases[0]
    for q in range_constexpr(1, len(bases)):
        base = (p == q).select(bases[q], base)
    return base


def push(put, addr, off, name, rank, t, s, col, vals, d: Dims, two_hop: bool):
    """This rank's partial pair at (t, col) -- ``vals``, ``put(region, pair,
    vals)`` -- to every rank's PART (one hop) or to the column's owner's (two
    hops); ``addr(p)``: peer p's base, ``off``: the region's offset in it."""
    if two_hop:
        bases = [addr(p) for p in range(d.tp)]
        dst = preg(peer_at(bases, owner(d, col)), off, name)
        put(dst, part_pair(d, rank, t, s, col), vals)
    else:
        for p in range_constexpr(d.tp):
            dst = preg(addr(p), off, name)
            put(dst, part_pair(d, rank, t, s, col), vals)


def reduce_pair(c, own, t, s, col, d: Dims):
    """One hop: the ranks' PART pairs at (t, col) summed in rank order, fp32
    (lo, hi). Two hops, the owner's sum (``reduce_owned``)."""
    return sum_partials(c["poll"], own, lambda src: part_pair(d, src, t, s, col), d.tp)


def reduce_owned(c, own, off, name, t, s, col, d: Dims):
    """Two hops, on the column's owner: the ranks' sum (``reduce_pair``), its
    bf16 pushed to every other rank's SUM -> (lo, hi)."""
    acc0, acc1 = reduce_pair(c, own, t, s, col, d)
    lo, hi = bf16_round(acc0), bf16_round(acc1)
    at = 2 * sum_pair(d, t, s, col)
    for p in range_constexpr(d.tp):
        _put_other(c, p, preg(c["peer_addr"](p), off, name), at, [lo, hi])
    return acc0, acc1


@traced
def _put_other(c, p, region, at, vals):
    if p != c["rank"]:
        c["put_bf"](region, at, vals)


def reduce_gathered(c, own, t, s, col, d: Dims):
    """Two hops, off the column's owner: its sum from SUM -> (lo, hi)."""
    w = c["poll"]([(own, sum_pair(d, t, s, col), 1)])[0][0]
    return bf_lo(w), bf_hi(w)
