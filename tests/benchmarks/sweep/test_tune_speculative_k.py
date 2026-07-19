# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import cast

import pytest

from vllm.benchmarks.sweep.tune_speculative_k import (
    SpecKMeasurement,
    SweepTuneSpeculativeKArgs,
    _extract_k,
    aggregate_measurements,
    build_schedule,
    run_main,
    select_k_by_batch_size,
    tune_speculative_k,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _record(batch_size: int, k: int, throughput: float, acceptance: float = 2.0):
    return {
        "max_concurrency": batch_size,
        "output_throughput": throughput,
        "acceptance_length": acceptance,
        "speculative_config": {
            "num_speculative_tokens": 7,
            "num_speculative_tokens_per_batch_size": [[1, 64, k]],
        },
    }


def _measurement(batch_size: int, k: int, objective: float) -> SpecKMeasurement:
    return SpecKMeasurement(
        batch_size=batch_size,
        k=k,
        num_runs=3,
        median_objective=objective,
        mad_objective=0.0,
        conservative_objective=objective,
        objective_relative_mad=0.0,
    )


def test_extract_k_from_nested_schedule_with_gap_and_tail():
    record: dict[str, object] = {
        "speculative_config": {
            "num_speculative_tokens_per_batch_size": [
                [1, 4, 7],
                [9, 16, 3],
            ]
        }
    }

    assert (
        _extract_k(
            record, "speculative_config.num_speculative_tokens_per_batch_size", 6
        )
        == 7
    )
    assert (
        _extract_k(
            record, "speculative_config.num_speculative_tokens_per_batch_size", 32
        )
        == 3
    )


def test_aggregate_measurements_uses_median_and_mad():
    measurements = aggregate_measurements(
        [_record(1, 3, value) for value in [100.0, 102.0, 200.0]],
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="maximize",
        acceptance_var="acceptance_length",
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=1.0,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=True,
    )

    assert len(measurements) == 1
    assert measurements[0].median_objective == 102.0
    assert measurements[0].mad_objective == 2.0
    assert measurements[0].conservative_objective == pytest.approx(99.0348)


def test_aggregate_measurements_rejects_unstable_objective():
    with pytest.raises(ValueError, match="Unstable objective"):
        aggregate_measurements(
            [_record(1, 3, value) for value in [50.0, 100.0, 150.0]],
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
        )


def test_aggregate_measurements_can_require_stable_acceptance():
    records = [_record(1, 3, 100.0, acceptance) for acceptance in [1.0, 2.0, 3.0]]

    with pytest.raises(ValueError, match="Unstable acceptance"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var="acceptance_length",
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=True,
        )


def test_aggregate_measurements_accepts_stable_zero_acceptance():
    measurements = aggregate_measurements(
        [_record(1, 0, 100.0, acceptance=0.0) for _ in range(3)],
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="maximize",
        acceptance_var="acceptance_length",
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=0.1,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=True,
    )

    assert measurements[0].median_acceptance == 0.0
    assert measurements[0].acceptance_relative_mad == 0.0


@pytest.mark.parametrize("missing_value", [None, "absent"])
def test_aggregate_measurements_allows_undefined_k0_acceptance(missing_value):
    records = [_record(1, 0, 100.0) for _ in range(3)]
    for record in records:
        if missing_value is None:
            record["acceptance_length"] = None
        else:
            del record["acceptance_length"]

    measurements = aggregate_measurements(
        records,
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="maximize",
        acceptance_var="acceptance_length",
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=0.1,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=True,
    )

    assert measurements[0].median_acceptance is None
    assert measurements[0].acceptance_relative_mad is None


def test_aggregate_measurements_requires_positive_k_acceptance():
    records = [_record(1, 3, 100.0) for _ in range(3)]
    records[0]["acceptance_length"] = None

    with pytest.raises(ValueError, match="Expected numeric acceptance_length"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var="acceptance_length",
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=True,
        )


def test_maximize_objective_allows_zero_goodput_candidate():
    measurements = aggregate_measurements(
        [_record(1, 0, 0.0) for _ in range(3)]
        + [_record(1, 3, 10.0) for _ in range(3)],
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="maximize",
        acceptance_var=None,
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=0.1,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=False,
    )

    assert select_k_by_batch_size(
        measurements,
        objective_direction="maximize",
        relative_tolerance=0.01,
    ) == {1: 3}


def test_selection_rejects_all_zero_maximize_objectives():
    with pytest.raises(ValueError, match="No positive conservative objective"):
        select_k_by_batch_size(
            [_measurement(1, 0, 0.0), _measurement(1, 3, 0.0)],
            objective_direction="maximize",
            relative_tolerance=0.01,
        )


def test_aggregate_measurements_rejects_negative_acceptance():
    with pytest.raises(ValueError, match="non-negative acceptance"):
        aggregate_measurements(
            [_record(1, 3, 100.0, acceptance=-1.0) for _ in range(3)],
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var="acceptance_length",
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=True,
        )


def test_aggregate_measurements_rejects_failed_requests():
    records = [_record(1, 3, 100.0) for _ in range(3)]
    records[0]["failed"] = 1

    with pytest.raises(ValueError, match="contains 1 failed request"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
        )


def test_aggregate_measurements_rejects_incomplete_requests():
    records = [_record(1, 3, 100.0) for _ in range(3)]
    records[0].update({"failed": 0, "completed": 31, "num_prompts": 32})

    with pytest.raises(ValueError, match="completed 31 of 32 prompts"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
        )


def test_aggregate_measurements_rejects_k_above_global_maximum():
    records = [_record(1, 7, 100.0) for _ in range(3)]
    for record in records:
        record["speculative_config"]["num_speculative_tokens"] = 3

    with pytest.raises(ValueError, match="K=7 exceeds configured global K=3"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
        )


def test_aggregate_measurements_requires_fixed_global_maximum():
    records = [_record(1, 3, 100.0) for _ in range(3)]
    records[0]["speculative_config"]["num_speculative_tokens"] = 3

    with pytest.raises(ValueError, match=r"found \[3, 7\]"):
        aggregate_measurements(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
        )


def test_selection_prefers_smaller_k_within_tolerance():
    selected = select_k_by_batch_size(
        [_measurement(1, 0, 100.0), _measurement(1, 7, 100.5)],
        objective_direction="maximize",
        relative_tolerance=0.01,
    )

    assert selected == {1: 0}


def test_selection_can_minimize_objective():
    selected = select_k_by_batch_size(
        [_measurement(1, 0, 10.0), _measurement(1, 3, 8.0)],
        objective_direction="minimize",
        relative_tolerance=0.0,
    )

    assert selected == {1: 3}


def test_minimize_objective_penalizes_upward_uncertainty():
    measurements = aggregate_measurements(
        [_record(1, 3, value) for value in [100.0, 102.0, 200.0]],
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="minimize",
        acceptance_var=None,
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=1.0,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=False,
    )

    assert measurements[0].conservative_objective == pytest.approx(104.9652)


def test_selection_preserves_nonmonotonic_local_optima():
    measurements = [
        _measurement(1, 1, 100.0),
        _measurement(1, 3, 70.0),
        _measurement(8, 1, 80.0),
        _measurement(8, 3, 100.0),
    ]

    selected = select_k_by_batch_size(
        measurements,
        objective_direction="maximize",
        relative_tolerance=0.0,
    )

    assert selected == {1: 1, 8: 3}


def test_selection_requires_same_candidates_at_every_batch_size():
    with pytest.raises(ValueError, match=r"expected \[1, 3\]"):
        select_k_by_batch_size(
            [
                _measurement(1, 1, 100.0),
                _measurement(1, 3, 110.0),
                _measurement(8, 1, 120.0),
            ],
            objective_direction="maximize",
            relative_tolerance=0.01,
        )


def test_build_schedule_fills_gaps_tail_and_merges_equal_k():
    assert build_schedule({1: 7, 8: 7, 32: 3, 64: 0}, 128) == [
        [1, 31, 7],
        [32, 63, 3],
        [64, 128, 0],
    ]


def test_tune_speculative_k_emits_runtime_schedule():
    records: list[dict[str, object]] = []
    for batch_size, values in {
        1: {0: 90.0, 3: 100.0, 7: 120.0},
        8: {0: 100.0, 3: 130.0, 7: 125.0},
        32: {0: 140.0, 3: 135.0, 7: 100.0},
    }.items():
        for k, throughput in values.items():
            records.extend(_record(batch_size, k, throughput) for _ in range(3))

    result = tune_speculative_k(
        records,
        batch_size_var="max_concurrency",
        k_var="speculative_config.num_speculative_tokens_per_batch_size",
        objective_var="output_throughput",
        objective_direction="maximize",
        acceptance_var="acceptance_length",
        min_runs=3,
        uncertainty_penalty=1.0,
        max_objective_relative_mad=0.1,
        max_acceptance_relative_mad=0.1,
        strict_acceptance_stability=True,
        relative_tolerance=0.01,
        max_batch_size=32,
    )

    assert result["num_speculative_tokens_per_batch_size"] == [
        [1, 7, 7],
        [8, 31, 3],
        [32, 32, 0],
    ]
    assert result["configured_global_num_speculative_tokens"] == 7
    assert result["max_num_speculative_tokens_observed"] == 7
    batch_decisions = cast(list[dict[str, object]], result["batch_decisions"])
    selection = cast(dict[str, object], result["selection"])
    assert batch_decisions[0]["selected_vs_k0_percent"] == pytest.approx(
        100.0 * (120.0 / 90.0 - 1.0)
    )
    assert batch_decisions[1]["selected_vs_best_percent"] == 0.0
    assert selection["acceptance_metric"] == "acceptance_length"
    assert selection["min_runs"] == 3
    assert selection["strict_acceptance_stability"] is True
    assert selection["max_batch_size"] == 32


def test_tuner_rejects_unmeasured_tail_extrapolation():
    records: list[dict[str, object]] = []
    for batch_size in [1, 8]:
        for k in [0, 3]:
            records.extend(_record(batch_size, k, 100.0 + k) for _ in range(3))

    with pytest.raises(ValueError, match="largest tuned batch-size anchor"):
        tune_speculative_k(
            records,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
            relative_tolerance=0.01,
            max_batch_size=64,
        )


def test_run_main_reads_sweep_summaries_and_writes_json(tmp_path):
    records: list[dict[str, object]] = []
    for batch_size, values in {1: {0: 90.0, 3: 100.0}, 8: {0: 110.0, 3: 100.0}}.items():
        for k, throughput in values.items():
            records.extend(_record(batch_size, k, throughput) for _ in range(3))
    result_dir = tmp_path / "case"
    result_dir.mkdir()
    (result_dir / "summary.json").write_text(json.dumps(records))
    output_path = tmp_path / "schedule.json"

    result = run_main(
        SweepTuneSpeculativeKArgs(
            experiment_dir=tmp_path,
            output_json=output_path,
            batch_size_var="max_concurrency",
            k_var="speculative_config.num_speculative_tokens_per_batch_size",
            objective_var="output_throughput",
            objective_direction="maximize",
            acceptance_var=None,
            min_runs=3,
            uncertainty_penalty=1.0,
            max_objective_relative_mad=0.1,
            max_acceptance_relative_mad=0.1,
            strict_acceptance_stability=False,
            relative_tolerance=0.01,
            max_batch_size=8,
        )
    )

    assert result["num_speculative_tokens_per_batch_size"] == [
        [1, 7, 3],
        [8, 8, 0],
    ]
    assert result["input_summary_files"] == ["case/summary.json"]
    assert json.loads(output_path.read_text()) == result


def test_cli_can_minimize_tpot(tmp_path):
    parser = FlexibleArgumentParser()
    SweepTuneSpeculativeKArgs.add_cli_args(parser)

    namespace = parser.parse_args(
        [
            str(tmp_path),
            "--objective-var",
            "mean_tpot_ms",
            "--objective-direction",
            "minimize",
        ]
    )
    args = SweepTuneSpeculativeKArgs.from_cli_args(namespace)

    assert args.objective_var == "mean_tpot_ms"
    assert args.objective_direction == "minimize"
