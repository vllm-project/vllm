# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from vllm.benchmarks.sweep.plot import SweepPlotArgs, main
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize("invalid_group", [0, 1])
def test_plot_propagates_worker_errors(tmp_path, invalid_group):
    records = [{"model": str(i), "tensor_parallel_size": 1} for i in range(2)]
    del records[invalid_group]["tensor_parallel_size"]
    (tmp_path / "summary.json").write_text(json.dumps(records))

    parser = SweepPlotArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            str(tmp_path),
            "--fig-by",
            "model",
            "--row-by",
            "tensor_parallel_size",
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="Cannot find metric 'tensor_parallel_size'"):
        main(args)
