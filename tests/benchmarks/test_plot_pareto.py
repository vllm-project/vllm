# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pandas as pd
import pytest

from vllm.benchmarks.sweep.plot_pareto import _pareto_frontier


@pytest.mark.parametrize(
    "points,expected_indices",
    [
        ([], []),
        ([(10, 5)], [0]),
        ([(5, 5), (10, 5)], [1]),
        ([(10, 3), (10, 5)], [1]),
        ([(10, 3), (5, 5), (3, 10)], [0, 1, 2]),
        ([(5, 5), (10, 10), (3, 3)], [1]),
        ([(10, 5), (10, 5), (5, 5)], [0, 1]),
        ([(10, 5), (5, 5 + 1e-10), (3, 6)], [0, 2]),
    ],
)
def test_pareto_frontier(points, expected_indices):
    df = pd.DataFrame(points, columns=["tokens_per_user", "tokens_per_gpu"])

    frontier = _pareto_frontier(df, "tokens_per_user", "tokens_per_gpu")

    pd.testing.assert_frame_equal(frontier, df.loc[expected_indices])
