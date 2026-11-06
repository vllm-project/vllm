# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for custom dataset sampling when --num-prompts exceeds
the number of available samples.

The default path oversamples with an INFO log; the --no-oversample path
must log a warning instead of silently sending fewer requests.
"""

import argparse
import json
import logging
from pathlib import Path

import pytest

from vllm.benchmarks.datasets import get_samples


def _write_jsonl(path: Path, n_rows: int) -> None:
    with path.open("w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"prompt": f"row {i}: unique prompt content."}) + "\n")


def _args_for_custom(
    dataset_path: str, num_prompts: int, no_oversample: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="custom",
        dataset_path=dataset_path,
        disable_shuffle=False,
        num_prompts=num_prompts,
        custom_output_len=32,
        skip_chat_template=True,
        no_oversample=no_oversample,
        seed=42,
        request_id_prefix="",
    )


@pytest.mark.benchmark
def test_custom_dataset_num_prompts_larger_than_rows_oversamples_by_default(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """Without --no-oversample, missing rows are filled with duplicates."""
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=400)

    with caplog.at_level(logging.INFO):
        samples = get_samples(_args_for_custom(str(jsonl), 600, False), None)

    assert len(samples) == 600
    assert "Oversampled requests to reach 600 total samples." in caplog.text


@pytest.mark.benchmark
def test_custom_dataset_num_prompts_larger_than_rows_warns_with_no_oversample(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """--no-oversample must warn instead of silently sending fewer requests."""
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, n_rows=400)

    with caplog.at_level(logging.WARNING):
        samples = get_samples(_args_for_custom(str(jsonl), 600, True), None)

    assert len(samples) == 400
    assert "only 400 requests will be sent" in caplog.text
