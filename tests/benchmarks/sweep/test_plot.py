# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from unittest.mock import patch

import pytest

from vllm.benchmarks.sweep.plot import SweepPlotArgs, main


def test_sqrt_axes_on_sweep_plot(tmp_path):
    pytest.importorskip("seaborn")
    from matplotlib.figure import Figure

    records = [
        {"total_token_throughput": x, "median_ttft_ms": x, "group": g}
        for g in ("a", "b")
        for x in (0, 4, 9)
    ]
    (tmp_path / "summary.json").write_text(json.dumps(records))
    parser = SweepPlotArgs.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            str(tmp_path),
            "--col-by=group",
            "--scale-x=sqrt",
            "--scale-y=sqrt",
            "--fig-dpi=40",
        ]
    )

    with patch.object(
        Figure, "savefig", autospec=True, side_effect=Figure.savefig
    ) as save:
        main(args)

    assert (tmp_path / "FIGURE.png").is_file()
    figure = save.call_args.args[0]
    assert len(figure.axes) == 2
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            transform = axis.get_transform()
            assert list(transform.transform([-4, 0, 9])) == [-2, 0, 3]
            assert list(transform.inverted().transform([-2, 0, 3])) == [-4, 0, 9]
