import pandas as pd

from vllm.benchmarks.sweep.plot_pareto import _pareto_frontier


def test_pareto_frontier_tradeoffs():
    df = pd.DataFrame({
        "x": [10.0, 5.0, 2.0],
        "y": [100.0, 150.0, 200.0],
    })
    frontier = _pareto_frontier(df, "x", "y")
    assert len(frontier) == 3
    assert set(frontier.index) == {0, 1, 2}


def test_pareto_frontier_dominated():
    df = pd.DataFrame({
        "x": [10.0, 5.0],
        "y": [100.0, 90.0],
    })
    frontier = _pareto_frontier(df, "x", "y")
    assert len(frontier) == 1
    assert list(frontier.index) == [0]


def test_pareto_frontier_ties():
    # Issue #55561: Dominated throughput ties
    df = pd.DataFrame({
        "x": [10.0, 5.0],
        "y": [100.0, 100.0],
    })
    frontier = _pareto_frontier(df, "x", "y")
    assert len(frontier) == 1
    assert list(frontier.index) == [0]


def test_pareto_frontier_duplicates():
    # Equal coordinates should still retain all associated runs for labeling
    df = pd.DataFrame({
        "x": [10.0, 10.0],
        "y": [100.0, 100.0],
    })
    frontier = _pareto_frontier(df, "x", "y")
    assert len(frontier) == 2
    assert set(frontier.index) == {0, 1}


def test_pareto_frontier_mixed():
    df = pd.DataFrame({
        "x": [10.0, 10.0, 5.0, 5.0, 2.0, 2.0],
        "y": [100.0, 100.0, 150.0, 100.0, 140.0, 160.0],
    })
    frontier = _pareto_frontier(df, "x", "y")
    assert len(frontier) == 4
    # Expected:
    # (10.0, 100.0) index 0 and 1
    # (5.0, 150.0) index 2. (5.0, 100.0) index 3 is dominated by index 2.
    # (2.0, 160.0) index 5. (2.0, 140.0) index 4 is dominated by index 5.
    assert set(frontier.index) == {0, 1, 2, 5}
