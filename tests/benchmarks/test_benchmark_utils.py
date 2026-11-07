# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from enum import IntEnum

import pytest

from vllm.benchmarks.lib.utils import (
    convert_to_pytorch_benchmark_format,
    write_to_json,
)


class CompileMode(IntEnum):
    NONE = 0
    ENABLED = 3


@pytest.mark.parametrize("field", ["output_json", "result_filename"])
@pytest.mark.parametrize("filename", [None, "", "results.json", "eager.json"])
def test_export_optional_filename(monkeypatch, tmp_path, field, filename):
    monkeypatch.setenv("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "1")
    args = argparse.Namespace(model="test-model", **{field: filename})
    records = convert_to_pytorch_benchmark_format(args, {"latency": [1.0]}, {})
    output = tmp_path / "results.pytorch.json"
    write_to_json(str(output), records)
    record = json.loads(output.read_text())[0]
    assert record["benchmark"]["extra_info"]["use_compile"] == (
        filename != "eager.json"
    )
    assert record["metric"]["benchmark_values"] == [1.0]


@pytest.mark.parametrize("from_extra_info", [False, True])
@pytest.mark.parametrize(
    "mode,expected",
    [
        (0, False),
        ("0", False),
        (CompileMode.NONE, False),
        (3, True),
        ("3", True),
        (CompileMode.ENABLED, True),
    ],
)
def test_export_compilation_mode(monkeypatch, from_extra_info, mode, expected):
    monkeypatch.setenv("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", "1")
    args = argparse.Namespace(
        model="test-model",
        compilation_config=argparse.Namespace(
            mode=(0 if expected else 3) if from_extra_info else mode
        ),
    )
    extra_info = {"compilation_config.mode": mode} if from_extra_info else {}
    record = convert_to_pytorch_benchmark_format(args, {"latency": [1.0]}, extra_info)[
        0
    ]
    assert record["benchmark"]["extra_info"]["use_compile"] is expected


def test_export_disabled_with_unset_filename(monkeypatch):
    monkeypatch.delenv("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", raising=False)
    args = argparse.Namespace(model="test-model", result_filename=None)
    assert convert_to_pytorch_benchmark_format(args, {"latency": [1.0]}, {}) == []
