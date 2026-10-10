# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.benchmarks.sweep.plot import SweepPlotArgs, main
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize("concurrencies", [(None, 32), (None,)])
@pytest.mark.parametrize(
    "option,group_by",
    [
        ("--curve-by", "max_concurrency"),
        ("--curve-by", "model,max_concurrency"),
        ("--curve-by", "model,dtype,max_concurrency"),
        ("--curve-by", "model,dtype,mode,max_concurrency"),
        ("--row-by", "max_concurrency"),
        ("--col-by", "max_concurrency"),
    ],
)
def test_plot_keeps_null_group_values(
    tmp_path, monkeypatch, option, group_by, concurrencies
):
    """Unset sweep parameters must not silently remove benchmark curves."""
    pytest.importorskip("seaborn")
    from matplotlib.figure import Figure

    records = [
        dict(
            model="model",
            dtype="auto",
            mode="default",
            max_concurrency=concurrency,
            total_token_throughput=100 * point,
            median_ttft_ms=3 * index + point,
        )
        for index, concurrency in enumerate(concurrencies)
        for point in (1, 2)
    ]
    (tmp_path / "summary.json").write_text(json.dumps(records))
    lines: list[list[list[float]]] = []
    savefig = Figure.savefig

    def capture_lines(figure, *args, **kwargs):
        lines.extend(
            line.get_xydata().tolist()
            for axis in figure.axes
            for line in axis.lines
            if len(line.get_xdata())
        )
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_lines)
    parser = SweepPlotArgs.add_cli_args(FlexibleArgumentParser())
    main(
        parser.parse_args(
            [
                str(tmp_path),
                option,
                group_by,
                "--no-error-bars",
                "--fig-dpi",
                "50",
            ]
        )
    )

    assert sorted(lines) == [
        [[100, 3 * index + 1], [200, 3 * index + 2]]
        for index in range(len(concurrencies))
    ]
    assert (tmp_path / "FIGURE.png").is_file()
