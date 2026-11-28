# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

MOD = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/issues/69306/analyze_whisper_steps.py"
)
spec = importlib.util.spec_from_file_location("issue_69306", MOD)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_parse_and_analyze_issue_like_encoder_admission():
    lines = [
        (
            "INFO Engine 000: Iteration(1): 1 context requests, 40 context "
            "tokens, 3 generation requests, 3 generation tokens, iteration "
            "elapsed time: 33.00 ms, GPU KV cache usage: 23.0%, encoder inputs: "
            "2, encoder output embeddings: 100"
        ),
        (
            "INFO Engine 000: Iteration(2): 0 context requests, 0 context tokens, "
            "3 generation requests, 3 generation tokens, iteration elapsed time: "
            "1.90 ms, GPU KV cache usage: 23.0%"
        ),
        (
            "INFO Engine 000: Iteration(3): 0 context requests, 0 context tokens, "
            "3 generation requests, 3 generation tokens, iteration elapsed time: "
            "34.00 ms, GPU KV cache usage: 23.0%, encoder inputs: 1, encoder "
            "output embeddings: 10"
        ),
    ]
    steps = [mod.parse_line(line) for line in lines]
    assert all(step is not None for step in steps)
    report = mod.summarize(steps)
    assert report["total_steps"] == 3
    assert report["encoder_admission_steps"] == 2
    assert report["encoder_admission_fraction_pct"] == pytest.approx(66.667, rel=1e-4)
    assert report["median_admission_step_ms"] == 33.5
    assert report["median_non_admission_step_ms"] == 1.9
    assert report["observed_admission_step_time_fraction_pct"] > 90


def test_ignore_dummy_missing_timing_and_unrelated_logs():
    line = (
        "Engine 000: Iteration(4): 0 context requests, 0 context tokens, "
        "0 generation requests, 0 generation tokens (dummy), GPU KV cache usage: 0.0%"
    )
    assert mod.parse_line(line) is None
    assert mod.parse_line("not an iteration") is None
    line2 = (
        "Engine 000: Iteration(5): 0 context requests, 0 context tokens, "
        "4 generation requests, 4 generation tokens, GPU KV cache usage: 0.0%"
    )
    step = mod.parse_line(line2)
    assert step is not None
    assert step.elapsed_ms is None
    report = mod.summarize([step])
    assert report["steps_with_timing"] == 0
    assert report["observed_admission_step_time_fraction_pct"] is None


@pytest.mark.parametrize(
    "duration",
    ["", ", iteration elapsed time: NaN ms", ", iteration elapsed time: -2 ms"],
)
def test_reports_missing_admission_timing(duration):
    admission = mod.parse_line(
        "Iteration(1): 1 context requests, 42 context tokens, 2 generation "
        f"requests, 2 generation tokens{duration}, GPU KV cache usage: 1.0%, "
        "encoder inputs: 1, encoder output embeddings: 10"
    )
    decode = mod.parse_line(
        "Iteration(2): 0 context requests, 0 context tokens, 3 generation "
        "requests, 3 generation tokens, iteration elapsed time: 2.00 ms, "
        "GPU KV cache usage: 1.0%"
    )
    report = mod.summarize([admission, decode])
    assert report["encoder_admission_steps"] == 1
    assert report["admission_steps_missing_timing"] == 1
    assert report["non_admission_steps_missing_timing"] == 0
    assert report["timing_coverage_pct"] == 50
    assert report["observed_admission_step_time_fraction_pct"] is None


def test_parses_iteration_records_emitted_by_vllm(caplog):
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.v1.metrics.loggers import LoggingStatLogger
    from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats

    caplog.set_level(logging.INFO, logger="vllm.v1.metrics.loggers")
    logger = LoggingStatLogger(VllmConfig(device_config=DeviceConfig(device="cpu")))
    logger.record(
        scheduler_stats=SchedulerStats(
            kv_cache_usage=0.25,
            iteration_records=[
                SchedulerIterationDetails(
                    iteration_index=2,
                    num_ctx_requests=1,
                    num_ctx_tokens=4,
                    num_generation_requests=1,
                    num_generation_tokens=1,
                    num_encoder_inputs=1,
                    num_encoder_output_tokens=1500,
                ),
            ],
        ),
        iteration_stats=None,
    )

    steps = [step for line in caplog.messages if (step := mod.parse_line(line))]
    assert steps == [mod.Step(index=2, encoder_inputs=1, elapsed_ms=None)]
    assert mod.summarize(steps)["timing_coverage_pct"] == 0.0
