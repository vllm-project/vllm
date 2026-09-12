# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test helpers: the drift-failure message, and the derivations behind
the subtractive invariants in test_demote.py.

Claims and demotions both delete graph edges on purpose, and each deletion is
sound only if the coverage it carried arrives another way. These work out
whether it did. They live here rather than in the package because nothing the
package ships calls them, and as CLI commands they only ran when someone
remembered to.
"""

from __future__ import annotations

from ci_selector.codemap.classify import select
from ci_selector.codemap.graph.build import FullGraph
from ci_selector.codemap.graph.demote import (
    CONFIG_KEY_MAX_TEST_FILES,
    leaf_literal_fanout,
)
from ci_selector.codemap.repo import is_test_basename
from ci_selector.codemap.state import RepoState


def leaf_origin_drops(state: RepoState) -> list[tuple[str, str]]:
    return [
        (src, dst)
        for src, dst in state.full.graph.dropped_lazy
        if src.startswith(("tests/", "benchmarks/"))
    ]


def _member_literals(state: RepoState) -> dict[str, set[str]]:
    claimed: dict[str, set[str]] = {}
    claims = state.full.dispatch.claims
    for (_, member), lits in state.full.dispatch.demotions.items():
        if member in claims:
            claimed.setdefault(member, set()).update(lits)
    return claimed


def key_selection_gaps(
    state: RepoState,
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
    state: RepoState,
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


def _tests(fg: FullGraph, file: str, include_boot: bool = True) -> set[str]:
    closure = fg.graph.reverse_closure({file}, include_boot=include_boot)
    return {f for f in closure if is_test_basename(f)}


def demoted_members(fg: FullGraph) -> set[str]:
    return {mem for _, mem in fg.dispatch.demotions}


def starved_members(fg: FullGraph) -> list[str]:
    """Demoted plugins whose key routing produced no test coverage."""
    return sorted(m for m in demoted_members(fg) if not _tests(fg, m))


# Where a drift fix almost always lands. Named once so every message spells it
# the same way and a rename is one edit.
HW = "ci_selector/handwritten.py"


def drift_message(what: str, cost: str, *fixes: str) -> str:
    """A drift failure written for whoever tripped it, who is usually a vLLM
    contributor with no idea this tool exists.

    Always three parts in this order: what moved, what it costs CI selection,
    and the exact edit. Anything shorter leaves the reader debugging a project
    they have never heard of, and a check nobody can act on gets switched off.
    """
    head = "Fix:" if len(fixes) == 1 else "Fix one of:"
    return "\n".join(["", what, "", cost, "", head, *(f"  - {f}" for f in fixes), ""])
