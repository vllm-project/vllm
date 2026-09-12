# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dominant-cycle detection, on synthetic graphs plus the real checkout.

The rule built on this replaces reachability wholesale, so what matters is the
trigger's edge set and its floor, not the SCC arithmetic.
"""

from ci_selector.codemap.graph.cycles import (
    MIN_CYCLE_SIZE,
    dominant_cycle,
)
from ci_selector.codemap.graph.imports import ImportGraph


def _ring(size: int, prefix: str = "vllm/m") -> ImportGraph:
    graph = ImportGraph()
    for i in range(size):
        graph.add_edge(f"{prefix}{i}.py", f"{prefix}{(i + 1) % size}.py")
    return graph


def test_an_acyclic_graph_has_no_cycle():
    graph = ImportGraph()
    graph.add_edge("a.py", "b.py")
    graph.add_edge("b.py", "c.py")
    assert dominant_cycle(graph).files == frozenset()


def test_a_cycle_below_the_floor_is_not_one():
    """Ordinary mutual imports are everywhere. Only a knot earns the substitute
    signal, so the floor is what keeps this rule from firing on a pair."""
    assert dominant_cycle(_ring(MIN_CYCLE_SIZE - 1)).files == frozenset()
    assert len(dominant_cycle(_ring(MIN_CYCLE_SIZE)).files) == MIN_CYCLE_SIZE


def test_reach_blind_adds_what_the_cycle_imports():
    """A file the cycle imports inherits the cycle's importers, so its reverse
    closure is just as undiscriminating even though it is in no cycle."""
    graph = _ring(MIN_CYCLE_SIZE)
    graph.add_edge("vllm/m0.py", "vllm/leaf.py")
    graph.add_edge("vllm/leaf.py", "vllm/deeper.py")
    cycle = dominant_cycle(graph)
    assert "vllm/leaf.py" not in cycle.files
    assert cycle.files < cycle.reach_blind
    assert {"vllm/leaf.py", "vllm/deeper.py"} <= cycle.reach_blind


def test_a_cycle_that_exists_only_through_a_demoted_edge_is_not_found():
    """The trigger has to walk what the closures walk. Demoted edges stay in
    `imports` and are skipped by reverse_closure, so a raw walk would find a
    knot selection does not believe in."""
    graph = _ring(MIN_CYCLE_SIZE)
    closing = (f"vllm/m{MIN_CYCLE_SIZE - 1}.py", "vllm/m0.py")
    graph.demoted_edges.add(closing)
    assert dominant_cycle(graph).files == frozenset()


def test_a_gated_edge_still_closes_a_cycle():
    """Gated edges are walked by default (`include_boot=True`), so the trigger
    keeps them; excluding them would shrink the cycle below what selection sees."""
    graph = ImportGraph()
    for i in range(MIN_CYCLE_SIZE - 1):
        graph.add_edge(f"vllm/m{i}.py", f"vllm/m{i + 1}.py")
    graph.add_boot_edge(f"vllm/m{MIN_CYCLE_SIZE - 1}.py", "vllm/m0.py")
    assert len(dominant_cycle(graph).files) == MIN_CYCLE_SIZE


def test_a_deep_chain_does_not_overflow_the_stack():
    """Tarjan is iterative because the real graph is thousands of nodes deep; a
    recursive one dies here rather than in production."""
    graph = ImportGraph()
    depth = 5000
    for i in range(depth):
        graph.add_edge(f"vllm/chain{i}.py", f"vllm/chain{i + 1}.py")
    for i in range(MIN_CYCLE_SIZE):
        graph.add_edge(f"vllm/r{i}.py", f"vllm/r{(i + 1) % MIN_CYCLE_SIZE}.py")
    graph.add_edge(f"vllm/chain{depth}.py", "vllm/r0.py")
    assert len(dominant_cycle(graph).files) == MIN_CYCLE_SIZE


def test_the_largest_component_wins_and_is_stable():
    graph = _ring(MIN_CYCLE_SIZE, prefix="vllm/small")
    for i in range(MIN_CYCLE_SIZE + 40):
        graph.add_edge(
            f"vllm/big{i}.py", f"vllm/big{(i + 1) % (MIN_CYCLE_SIZE + 40)}.py"
        )
    first = dominant_cycle(graph)
    assert len(first.files) == MIN_CYCLE_SIZE + 40
    assert all(f.startswith("vllm/big") for f in first.files)
    assert first == dominant_cycle(graph)


def test_the_real_checkout_has_one_large_knot(state):
    """The premise of the whole rule. If this ever collapses, co-location is
    substituting for a reachability signal that works."""
    cycle = state.full.import_cycle()
    assert len(cycle.files) > 500, len(cycle.files)
    assert cycle.files < cycle.reach_blind
    assert "vllm/config/__init__.py" in cycle.reach_blind


def test_the_cycle_is_memoized_per_graph(state):
    assert state.full.import_cycle() is state.full.import_cycle()
