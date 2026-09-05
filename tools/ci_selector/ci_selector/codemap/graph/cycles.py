# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dominant import cycle: where the import graph stops telling you anything.

vLLM has one very large group of files that all import each other. Every file
in it reaches the same set and selects the same steps, so the graph rule cannot
tell them apart however it is tuned.

This module only FINDS the cycle. What to do with a file inside it is policy,
and lives in `codemap/colocation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .imports import ImportGraph

# Below this, a component is an ordinary mutual-import pair rather than a knot
# worth substituting a different signal for, and the graph rule keeps the file.
MIN_CYCLE_SIZE = 100


@dataclass(frozen=True)
class ImportCycle:
    #: The strongly connected component itself.
    files: frozenset[str]
    #: `files` plus everything they import. A file the cycle imports inherits
    #: the cycle's importers, so its reverse closure is just as undiscriminating
    #: even though it is not itself in a cycle.
    reach_blind: frozenset[str]


EMPTY = ImportCycle(frozenset(), frozenset())


def _selection_edges(graph: ImportGraph) -> dict[str, set[str]]:
    """`graph.imports` minus demoted edges: the edge set the closures walk.

    The closure walks skip demoted edges but `graph.imports` still holds them,
    so a raw walk would see edges selection has disowned. Gated edges stay.
    """
    demoted = graph.demoted_edges
    return {
        src: {dst for dst in dsts if (src, dst) not in demoted}
        for src, dsts in graph.imports.items()
    }


def _largest_component(adj: dict[str, set[str]]) -> set[str]:
    """Tarjan, iterative. Recursion overflows the stack on a graph this size."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    best: set[str] = set()
    counter = 0

    nodes = set(adj) | {dst for dsts in adj.values() for dst in dsts}
    for root in sorted(nodes):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(sorted(adj.get(root, ()))))]
        while work:
            node, successors = work[-1]
            descended = False
            for succ in successors:
                if succ not in index:
                    index[succ] = low[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, iter(sorted(adj.get(succ, ())))))
                    descended = True
                    break
                if succ in on_stack:
                    low[node] = min(low[node], index[succ])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                if len(component) > len(best):
                    best = component
    return best


def dominant_cycle(graph: ImportGraph) -> ImportCycle:
    """The largest import cycle, or EMPTY when the graph has no knot.

    One O(V+E) pass. Callers should reach this through
    `FullGraph.import_cycle()`, which memoizes it per tree.
    """
    adj = _selection_edges(graph)
    component = _largest_component(adj)
    if len(component) < MIN_CYCLE_SIZE:
        return EMPTY

    reached = set(component)
    stack = list(component)
    while stack:
        node = stack.pop()
        for dst in adj.get(node, ()):
            if dst not in reached:
                reached.add(dst)
                stack.append(dst)
    return ImportCycle(frozenset(component), frozenset(reached))
