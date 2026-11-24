# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import math
from pathlib import Path

import pytest

from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (math.nan, "nan"),
        ((1.5, -math.inf, None), [1.5, "-inf", None]),
    ],
)
def test_export_nonfinite_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting, expected
) -> None:
    monkeypatch.setenv("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "1")
    records = convert_to_pytorch_benchmark_format(
        argparse.Namespace(model="test"),
        metrics={"latency": [1.25]},
        extra_info={"setting": setting},
    )
    output = tmp_path / "run.pytorch.json"
    write_to_json(str(output), records)

    def reject_constant(value: str):
        pytest.fail(f"Invalid JSON constant in benchmark export: {value}")

    (record,) = json.loads(output.read_text(), parse_constant=reject_constant)
    assert record["metric"]["extra_info"]["setting"] == expected
    assert record["metric"]["benchmark_values"] == [1.25]
