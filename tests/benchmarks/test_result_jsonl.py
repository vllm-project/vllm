# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math

from vllm.benchmarks.lib.utils import append_jsonl_record


def test_append_jsonl_record_writes_one_record_per_line(tmp_path):
    output_file = tmp_path / "benchmarks.jsonl"

    append_jsonl_record(output_file, {"backend": "vllm", "throughput": 1.0})
    append_jsonl_record(output_file, {"backend": "openai", "throughput": 2.0})

    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"backend": "vllm", "throughput": 1.0}
    assert json.loads(lines[1]) == {"backend": "openai", "throughput": 2.0}


def test_append_jsonl_record_handles_existing_file_without_trailing_newline(
    tmp_path,
):
    output_file = tmp_path / "benchmarks.jsonl"
    output_file.write_text('{"backend": "vllm"}', encoding="utf-8")

    append_jsonl_record(output_file, {"backend": "openai"})

    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"backend": "vllm"}
    assert json.loads(lines[1]) == {"backend": "openai"}


def test_append_jsonl_record_creates_parent_dirs(tmp_path):
    output_file = tmp_path / "nested" / "results" / "benchmarks.jsonl"

    append_jsonl_record(output_file, {"backend": "vllm"})

    assert json.loads(output_file.read_text(encoding="utf-8")) == {"backend": "vllm"}


def test_append_jsonl_record_encodes_infinity(tmp_path):
    output_file = tmp_path / "benchmarks.jsonl"

    append_jsonl_record(output_file, {"request_rate": math.inf})

    assert json.loads(output_file.read_text(encoding="utf-8")) == {
        "request_rate": "inf"
    }
