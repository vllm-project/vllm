# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every edge the subtractive passes dropped still has another route.

Claims and demotions both delete graph edges on purpose; each deletion is sound
only if coverage survives another way, so this is the check that the replacement
route is real. All invariants fail closed (exit 1):

1. No dropped leaf-origin lazy edge: a tests/ or benchmarks/ lazy import into a
   claimed target must never be finalized away (the flashinfer silent-unlinking
   class).
2. Every auto step naming a demoted member's gating literal still runs when that
   member changes (closes the loop on the sites that select the key).
3. Every gating literal either routes its own member or was refused on purpose;
   a member with no route must not read as a mere statistic.
4. Every literal the leaf-fanout bar drops was never a routing key or still
   routes through the key index (a high-fanout word like "pooling" is registered
   at fanout 76, yet looks like ordinary English).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..curated import CONFIG_KEY_MAX_TEST_FILES
from ..graph.dispatch import leaf_literal_fanout
from ..select import AnalyzerState, select


def leaf_origin_drops(state: AnalyzerState) -> list[tuple[str, str]]:
    return [
        (src, dst)
        for src, dst in state.full.graph.dropped_lazy
        if src.startswith(("tests/", "benchmarks/"))
    ]


def _member_literals(state: AnalyzerState) -> dict[str, set[str]]:
    claimed: dict[str, set[str]] = {}
    claims = state.full.dispatch.claims
    for (_, member), lits in state.full.dispatch.demotions.items():
        if member in claims:
            claimed.setdefault(member, set()).update(lits)
    return claimed


def key_selection_gaps(
    state: AnalyzerState,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], int]:
    """((member, step_id) misses, (member, literal) unrouted, members checked).

    A literal counts as routing a member only when it names THAT member --
    `lit in key_mechanism` also accepted a literal owned by some other file.
    A literal that routes nothing and was not refused on purpose is a gap, not
    a statistic. `checked` counts members that actually exercised the
    invariant, so a silent slide to all-skipped is visible."""
    gaps: list[tuple[str, str]] = []
    unrouted: list[tuple[str, str]] = []
    checked = 0
    for member, literals in sorted(_member_literals(state).items()):
        routed = literals & state.keys.for_file(member)
        unrouted += [
            (member, lit)
            for lit in sorted(literals - routed)
            if lit not in state.keys.refused
        ]
        steps = state.keys.steps_naming(routed) & state.auto_step_ids
        if not steps:
            continue
        checked += 1
        sel = select(state, [member])
        for sid in sorted(steps):
            if sid in sel.selected or sid.split(":", 1)[0] in sel.run_all:
                continue
            gaps.append((member, sid))
    return gaps, unrouted, checked


def fanout_dropped_literals(
    state: AnalyzerState,
) -> tuple[list[tuple[str, str]], int]:
    """((member, literal) losses, literals the bar dropped).

    The bar refuses a gating literal held by many leaf files, on the theory
    that it is ordinary English rather than a config key. That theory is only
    half true -- it drops registered keys too -- so the check is not "is this
    a key" but "does dropping it cost a route": a dropped literal is fine when
    it names no auto step through the key index either, and fine when it still
    does. It is a loss only when the key index says it routes somewhere and
    that somewhere is now unreachable from the member.
    """
    fanout = leaf_literal_fanout(state.full.graph)
    losses: list[tuple[str, str]] = []
    dropped: set[str] = set()
    for (_, member), literals in sorted(state.full.dispatch.demotions.items()):
        for lit in sorted(literals):
            if fanout[lit] <= CONFIG_KEY_MAX_TEST_FILES:
                continue
            dropped.add(lit)
            if lit not in state.keys.key_mechanism:
                continue  # never a routing key; the bar took nothing
            if state.keys.steps_naming({lit}) & state.auto_step_ids:
                continue  # the key index still routes it
            losses.append((member, lit))
    return losses, len(dropped)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)


def run(args) -> int:
    state = AnalyzerState.build(args.repo.resolve())
    drops = leaf_origin_drops(state)
    gaps, unrouted, checked = key_selection_gaps(state)
    losses, dropped = fanout_dropped_literals(state)
    members = _member_literals(state)
    demotions = len(state.full.dispatch.demotions)
    no_literals = sum(1 for lits in members.values() if not lits)
    print(
        f"dropped lazy edges: {len(state.full.graph.dropped_lazy)} "
        f"({len(drops)} leaf-origin); {demotions} demotions, "
        f"{len(members)} claimed members ({checked} checked, "
        f"{len(members) - checked} skipped -- {no_literals} with no literals, "
        f"the rest routed but naming no auto step); "
        f"{len(state.keys.refused)} literals refused on purpose; "
        f"{dropped} literals over the leaf-fanout bar ({len(losses)} costing a "
        f"route); {len(gaps)} key-selection gaps, {len(unrouted)} unrouted literals"
    )
    for src, dst in drops:
        print(f"  DROPPED LEAF EDGE {src} -> {dst}")
    for member, sid in gaps:
        print(f"  GAP {member}: key-selecting step {sid} not selected")
    for member, lit in unrouted:
        print(f"  GAP {member}: gating literal {lit!r} routes it to nothing")
    for member, lit in losses:
        print(
            f"  GAP {member}: gating literal {lit!r} is a registered key but "
            "the leaf-fanout bar dropped it and no key route survives"
        )
    # Detection floors: a collapsed detector makes every downstream check pass
    # vacuously. Zero demotions is the obvious shape; every member sliding to
    # skipped is the quiet one, and is how a severed key stayed invisible.
    if not demotions:
        print("  DETECTOR COLLAPSE: dispatch produced zero demotions")
        return 1
    if members and not checked:
        print("  DETECTOR COLLAPSE: no claimed member exercised the invariant")
        return 1
    return 1 if drops or gaps or unrouted or losses else 0
