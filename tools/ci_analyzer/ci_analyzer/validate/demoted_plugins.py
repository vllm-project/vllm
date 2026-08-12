# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every demoted plugin still has test coverage.

Self-audit for the config-gated demotion pass. Demotion drops a plugin's eager
registration edge and re-routes coverage by key; the failure mode to catch is a
STARVED member, a demoted plugin whose key routing found zero tests, leaving the
run-all fail-open as its only coverage (the near-run-all regression demotion
exists to prevent). A non-empty starved list exits 1.

The trailing census is informational, not a gate: it nominates the next family
the dispatch parser should model, ranked by how many tests reach a file only
through the seam. Rank is not priority -- the head of the list is flat to
within a few percent -- so read it by cluster, not by position.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ..graph.build import FullGraph, build_full_graph
from ..repo import is_test_basename


@dataclass
class Amplifier:
    file: str
    seam_only: int
    outbound: int


def _tests(fg: FullGraph, file: str, include_gated: bool = True) -> set[str]:
    closure = fg.graph.reverse_closure({file}, include_gated=include_gated)
    return {f for f in closure if is_test_basename(f)}


def demoted_members(fg: FullGraph) -> set[str]:
    return {mem for _, mem in fg.dispatch.demotions}


def starved_members(fg: FullGraph) -> list[str]:
    """Demoted plugins whose key routing produced no test coverage."""
    return sorted(m for m in demoted_members(fg) if not _tests(fg, m))


def seam_census(fg: FullGraph) -> list[Amplifier]:
    """Unclaimed vllm/ files some test reaches ONLY through a gated edge.

    The boundary is derived, not tuned: seam-only counts are bimodal, with
    nothing at all between 0 and ~1000, so `> 0` splits the population exactly
    where the data does and needs no threshold to sit in the gap."""
    claims = fg.all_claims()
    sources = {imp for imp, _ in fg.dispatch.demotions}
    out: list[Amplifier] = []
    for file in fg.index.file_to_module:
        if not file.startswith("vllm/") or file in claims or file in sources:
            continue
        broad, narrow = len(_tests(fg, file)), len(_tests(fg, file, False))
        if broad > narrow:
            outbound = len(fg.graph.imports.get(file, ()))
            out.append(Amplifier(file, broad - narrow, outbound))
    out.sort(key=lambda a: a.seam_only, reverse=True)
    return out


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--census", action="store_true", help="also print the seam-amplifier census"
    )


def run(args) -> int:
    fg = build_full_graph(args.repo.resolve())
    members = demoted_members(fg)
    starved = starved_members(fg)
    print(
        f"demoted {len(fg.dispatch.demotions)} plugin edges over {len(members)} "
        f"members ({fg.dispatch.edges_added} routed test edges, "
        f"{len(fg.dispatch.claims)} claims, "
        f"{len(fg.graph.dropped_lazy)} dropped lazy edges); "
        f"{len(starved)} starved (zero routed tests)"
    )
    for m in starved:
        print(f"  STARVED {m} (demotion removed its only coverage)")
    if args.census:
        census = seam_census(fg)
        print(f"seam-amplified unclaimed files: {len(census)}")
        for a in census:
            print(f"  {a.file} (seam-only tests {a.seam_only}, imports {a.outbound})")
    # Detection floor: zero demotions = detector collapsed, so the starved
    # check would pass vacuously. Fail loud.
    if not fg.dispatch.demotions:
        print("  DETECTOR COLLAPSE: dispatch produced zero demotions")
        return 1
    return 1 if starved else 0
