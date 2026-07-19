# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import math
import statistics
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.spec_decode.dynamic.utils import (
    validate_and_normalize_dynamic_sd_schedule,
)

DEFAULT_K_VAR = "speculative_config.num_speculative_tokens_per_batch_size"
DEFAULT_GLOBAL_K_VAR = "speculative_config.num_speculative_tokens"


@dataclass(frozen=True)
class SpecKMeasurement:
    batch_size: int
    k: int
    num_runs: int
    median_objective: float
    mad_objective: float
    conservative_objective: float
    objective_relative_mad: float
    median_acceptance: float | None = None
    acceptance_relative_mad: float | None = None


def _get_nested(record: dict[str, object], key: str) -> object:
    if key in record:
        return record[key]

    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict):
            break
        for candidate in (part, part.replace("-", "_"), part.replace("_", "-")):
            if candidate in value:
                value = value[candidate]
                break
        else:
            break
    else:
        return value

    raise ValueError(f"Cannot find {key!r} in result keys {sorted(record)}")


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {name}, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {name}, got {value!r}")
    return result


def _positive_int(value: object, name: str) -> int:
    numeric = _finite_float(value, name)
    result = int(numeric)
    if result != numeric or result <= 0:
        raise ValueError(f"Expected positive integer {name}, got {value!r}")
    return result


def _non_negative_int(value: object, name: str) -> int:
    numeric = _finite_float(value, name)
    result = int(numeric)
    if result != numeric or result < 0:
        raise ValueError(f"Expected non-negative integer {name}, got {value!r}")
    return result


def _extract_k(record: dict[str, object], k_var: str, batch_size: int) -> int:
    value = _get_nested(record, k_var)
    if isinstance(value, str):
        with contextlib.suppress(json.JSONDecodeError):
            value = json.loads(value)

    if isinstance(value, int | float):
        numeric = _finite_float(value, k_var)
        k = int(numeric)
        if k != numeric or k < 0:
            raise ValueError(f"Expected non-negative integer K, got {value!r}")
        return k

    schedule = validate_and_normalize_dynamic_sd_schedule(value)
    selected_k = schedule[0][2]
    for range_start, _, range_k in schedule:
        if range_start > batch_size:
            break
        selected_k = range_k
    return selected_k


def _median_and_mad(values: list[float]) -> tuple[float, float, float]:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    relative_mad = mad / abs(median) if median != 0 else (0.0 if mad == 0 else math.inf)
    return median, mad, relative_mad


def aggregate_measurements(
    records: list[dict[str, object]],
    *,
    batch_size_var: str,
    k_var: str,
    objective_var: str,
    objective_direction: str,
    acceptance_var: str | None,
    min_runs: int,
    uncertainty_penalty: float,
    max_objective_relative_mad: float,
    max_acceptance_relative_mad: float,
    strict_acceptance_stability: bool,
) -> list[SpecKMeasurement]:
    if min_runs <= 0:
        raise ValueError("min_runs must be positive")
    if uncertainty_penalty < 0:
        raise ValueError("uncertainty_penalty must be non-negative")
    if objective_direction not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'")
    if max_objective_relative_mad < 0 or max_acceptance_relative_mad < 0:
        raise ValueError("relative MAD thresholds must be non-negative")

    grouped: dict[tuple[int, int], list[tuple[float, float | None]]] = defaultdict(list)
    global_ks: set[int] = set()
    for record in records:
        if "failed" in record:
            failed = _non_negative_int(record["failed"], "failed")
            if failed:
                raise ValueError(
                    f"Sweep record contains {failed} failed request(s); "
                    "only complete benchmark runs can be tuned"
                )
        if "completed" in record and "num_prompts" in record:
            completed = _non_negative_int(record["completed"], "completed")
            num_prompts = _positive_int(record["num_prompts"], "num_prompts")
            if completed != num_prompts:
                raise ValueError(
                    f"Sweep record completed {completed} of {num_prompts} prompts; "
                    "only complete benchmark runs can be tuned"
                )
        batch_size = _positive_int(_get_nested(record, batch_size_var), batch_size_var)
        k = _extract_k(record, k_var, batch_size)
        if k_var == DEFAULT_K_VAR:
            global_k = _positive_int(
                _get_nested(record, DEFAULT_GLOBAL_K_VAR), DEFAULT_GLOBAL_K_VAR
            )
            if k > global_k:
                raise ValueError(
                    f"Scheduled K={k} exceeds configured global K={global_k}"
                )
            global_ks.add(global_k)
        objective = _finite_float(_get_nested(record, objective_var), objective_var)
        if objective < 0 or (objective_direction == "minimize" and objective == 0):
            expectation = (
                "non-negative" if objective_direction == "maximize" else "positive"
            )
            raise ValueError(f"Expected {expectation} objective, got {objective!r}")
        acceptance = None
        if acceptance_var is not None:
            try:
                acceptance_value = _get_nested(record, acceptance_var)
            except ValueError:
                if k != 0:
                    raise
            else:
                if acceptance_value is None:
                    if k != 0:
                        raise ValueError(f"Expected numeric {acceptance_var}, got None")
                else:
                    acceptance = _finite_float(acceptance_value, acceptance_var)
        if acceptance is not None and acceptance < 0:
            raise ValueError(f"Expected non-negative acceptance, got {acceptance!r}")
        grouped[(batch_size, k)].append((objective, acceptance))

    if len(global_ks) > 1:
        raise ValueError(
            "All sweep records must use the same global num_speculative_tokens; "
            f"found {sorted(global_ks)}"
        )

    measurements = []
    for (batch_size, k), samples in sorted(grouped.items()):
        if len(samples) < min_runs:
            raise ValueError(
                f"batch_size={batch_size}, K={k} has {len(samples)} runs; "
                f"at least {min_runs} are required"
            )

        median_objective, mad_objective, objective_relative_mad = _median_and_mad(
            [sample[0] for sample in samples]
        )
        if objective_relative_mad > max_objective_relative_mad:
            raise ValueError(
                f"Unstable objective for batch_size={batch_size}, K={k}: "
                f"relative MAD {objective_relative_mad:.3f} exceeds "
                f"{max_objective_relative_mad:.3f}"
            )

        median_acceptance = None
        acceptance_relative_mad = None
        if acceptance_var is not None:
            acceptances = [sample[1] for sample in samples]
            if any(value is not None for value in acceptances):
                if not all(value is not None for value in acceptances):
                    raise ValueError(
                        f"Inconsistent missing {acceptance_var} values for "
                        f"batch_size={batch_size}, K={k}"
                    )
                median_acceptance, _, acceptance_relative_mad = _median_and_mad(
                    [float(value) for value in acceptances if value is not None]
                )
                if acceptance_relative_mad > max_acceptance_relative_mad:
                    message = (
                        f"Unstable acceptance for batch_size={batch_size}, K={k}: "
                        f"relative MAD {acceptance_relative_mad:.3f} exceeds "
                        f"{max_acceptance_relative_mad:.3f}"
                    )
                    if strict_acceptance_stability:
                        raise ValueError(message)
                    warnings.warn(message, stacklevel=2)

        # 1.4826 scales MAD to standard deviation for a normal distribution.
        uncertainty = uncertainty_penalty * 1.4826 * mad_objective
        conservative_objective = (
            max(median_objective - uncertainty, 0.0)
            if objective_direction == "maximize"
            else median_objective + uncertainty
        )
        measurements.append(
            SpecKMeasurement(
                batch_size=batch_size,
                k=k,
                num_runs=len(samples),
                median_objective=median_objective,
                mad_objective=mad_objective,
                conservative_objective=conservative_objective,
                objective_relative_mad=objective_relative_mad,
                median_acceptance=median_acceptance,
                acceptance_relative_mad=acceptance_relative_mad,
            )
        )

    if not measurements:
        raise ValueError("No benchmark measurements found")
    return measurements


def select_k_by_batch_size(
    measurements: list[SpecKMeasurement],
    *,
    objective_direction: str,
    relative_tolerance: float,
) -> dict[int, int]:
    if objective_direction not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'")
    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in [0, 1)")

    by_batch: dict[int, list[SpecKMeasurement]] = defaultdict(list)
    for measurement in measurements:
        by_batch[measurement.batch_size].append(measurement)

    batch_sizes = sorted(by_batch)
    if batch_sizes[0] != 1:
        raise ValueError("The smallest tuned batch size must be 1")
    expected_ks = {item.k for item in by_batch[batch_sizes[0]]}
    for batch_size in batch_sizes[1:]:
        actual_ks = {item.k for item in by_batch[batch_size]}
        if actual_ks != expected_ks:
            raise ValueError(
                f"batch_size={batch_size} has K candidates {sorted(actual_ks)}, "
                f"expected {sorted(expected_ks)}"
            )

    utility: dict[tuple[int, int], float] = {}
    for batch_size, candidates in by_batch.items():
        conservative_values = [
            candidate.conservative_objective for candidate in candidates
        ]
        if objective_direction == "maximize" and max(conservative_values) <= 0:
            raise ValueError(
                f"No positive conservative objective for batch_size={batch_size}"
            )
        if objective_direction == "minimize" and any(
            value <= 0 for value in conservative_values
        ):
            raise ValueError(
                f"No positive conservative objective for batch_size={batch_size}"
            )
        best_objective = (
            max(conservative_values)
            if objective_direction == "maximize"
            else min(conservative_values)
        )
        for candidate in candidates:
            ratio = (
                candidate.conservative_objective / best_objective
                if objective_direction == "maximize"
                else best_objective / candidate.conservative_objective
            )
            utility[(batch_size, candidate.k)] = (
                1.0 if ratio >= 1.0 - relative_tolerance else ratio
            )

    return {
        batch_size: min(
            by_batch[batch_size],
            key=lambda item: (-utility[(batch_size, item.k)], item.k),
        ).k
        for batch_size in batch_sizes
    }


def build_schedule(selected_k: dict[int, int], max_batch_size: int) -> list[list[int]]:
    batch_sizes = sorted(selected_k)
    if not batch_sizes or batch_sizes[0] != 1:
        raise ValueError("Selected batch sizes must start at 1")
    if max_batch_size < batch_sizes[-1]:
        raise ValueError("max_batch_size must cover the largest tuned batch size")

    schedule: list[list[int]] = []
    for index, range_start in enumerate(batch_sizes):
        range_end = (
            batch_sizes[index + 1] - 1
            if index + 1 < len(batch_sizes)
            else max_batch_size
        )
        k = selected_k[range_start]
        if schedule and schedule[-1][2] == k:
            schedule[-1][1] = range_end
        else:
            schedule.append([range_start, range_end, k])
    return schedule


def tune_speculative_k(
    records: list[dict[str, object]],
    *,
    batch_size_var: str,
    k_var: str,
    objective_var: str,
    objective_direction: str,
    acceptance_var: str | None,
    min_runs: int,
    uncertainty_penalty: float,
    max_objective_relative_mad: float,
    max_acceptance_relative_mad: float,
    strict_acceptance_stability: bool,
    relative_tolerance: float,
    max_batch_size: int | None,
) -> dict[str, object]:
    measurements = aggregate_measurements(
        records,
        batch_size_var=batch_size_var,
        k_var=k_var,
        objective_var=objective_var,
        objective_direction=objective_direction,
        acceptance_var=acceptance_var,
        min_runs=min_runs,
        uncertainty_penalty=uncertainty_penalty,
        max_objective_relative_mad=max_objective_relative_mad,
        max_acceptance_relative_mad=max_acceptance_relative_mad,
        strict_acceptance_stability=strict_acceptance_stability,
    )
    selected = select_k_by_batch_size(
        measurements,
        objective_direction=objective_direction,
        relative_tolerance=relative_tolerance,
    )
    resolved_max_batch_size = (
        max(selected) if max_batch_size is None else max_batch_size
    )
    if resolved_max_batch_size != max(selected):
        raise ValueError(
            "max_batch_size must be the largest tuned batch-size anchor; "
            "add a measurement at the deployment maximum instead of extrapolating"
        )
    schedule = build_schedule(selected, resolved_max_batch_size)
    configured_global_k = (
        _positive_int(
            _get_nested(records[0], DEFAULT_GLOBAL_K_VAR), DEFAULT_GLOBAL_K_VAR
        )
        if k_var == DEFAULT_K_VAR
        else None
    )

    by_batch: dict[int, list[SpecKMeasurement]] = defaultdict(list)
    for measurement in measurements:
        by_batch[measurement.batch_size].append(measurement)
    decisions = []
    for batch_size in sorted(selected):
        candidates = sorted(by_batch[batch_size], key=lambda item: item.k)
        selected_measurement = next(
            item for item in candidates if item.k == selected[batch_size]
        )
        best_objective = (
            max(item.conservative_objective for item in candidates)
            if objective_direction == "maximize"
            else min(item.conservative_objective for item in candidates)
        )
        k0_measurement = next((item for item in candidates if item.k == 0), None)
        decisions.append(
            {
                "batch_size": batch_size,
                "selected_k": selected[batch_size],
                "selected_median_objective": selected_measurement.median_objective,
                "selected_conservative_objective": (
                    selected_measurement.conservative_objective
                ),
                "selected_vs_best_percent": 100.0
                * (
                    (
                        selected_measurement.conservative_objective / best_objective
                        if objective_direction == "maximize"
                        else best_objective
                        / selected_measurement.conservative_objective
                    )
                    - 1.0
                ),
                "selected_vs_k0_percent": (
                    None
                    if k0_measurement is None
                    or k0_measurement.conservative_objective == 0
                    else 100.0
                    * (
                        (
                            selected_measurement.conservative_objective
                            / k0_measurement.conservative_objective
                            if objective_direction == "maximize"
                            else k0_measurement.conservative_objective
                            / selected_measurement.conservative_objective
                        )
                        - 1.0
                    )
                ),
                "candidates": [asdict(candidate) for candidate in candidates],
            }
        )

    return {
        "num_speculative_tokens_per_batch_size": schedule,
        "configured_global_num_speculative_tokens": configured_global_k,
        "max_num_speculative_tokens_observed": max(item.k for item in measurements),
        "selection": {
            "batch_size_metric": batch_size_var,
            "k_metric": k_var,
            "objective_metric": objective_var,
            "objective_direction": objective_direction,
            "acceptance_metric": acceptance_var,
            "min_runs": min_runs,
            "relative_tolerance": relative_tolerance,
            "uncertainty_penalty": uncertainty_penalty,
            "max_objective_relative_mad": max_objective_relative_mad,
            "max_acceptance_relative_mad": max_acceptance_relative_mad,
            "strict_acceptance_stability": strict_acceptance_stability,
            "max_batch_size": resolved_max_batch_size,
        },
        "batch_decisions": decisions,
    }


@dataclass
class SweepTuneSpeculativeKArgs:
    experiment_dir: Path
    output_json: Path
    batch_size_var: str
    k_var: str
    objective_var: str
    objective_direction: str
    acceptance_var: str | None
    min_runs: int
    uncertainty_penalty: float
    max_objective_relative_mad: float
    max_acceptance_relative_mad: float
    strict_acceptance_stability: bool
    relative_tolerance: float
    max_batch_size: int | None

    parser_name: ClassVar[str] = "tune_speculative_k"
    parser_help: ClassVar[str] = (
        "Generate a dynamic speculative-decoding K schedule from sweep results."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        experiment_dir = Path(args.EXPERIMENT_DIR)
        if not experiment_dir.exists():
            raise ValueError(f"No parameter sweep results under {experiment_dir}")
        output_json = (
            Path(args.output_json)
            if args.output_json
            else experiment_dir / "speculative_k_schedule.json"
        )
        return cls(
            experiment_dir=experiment_dir,
            output_json=output_json,
            batch_size_var=args.batch_size_var,
            k_var=args.k_var,
            objective_var=args.objective_var,
            objective_direction=args.objective_direction,
            acceptance_var=args.acceptance_var,
            min_runs=args.min_runs,
            uncertainty_penalty=args.uncertainty_penalty,
            max_objective_relative_mad=args.max_objective_relative_mad,
            max_acceptance_relative_mad=args.max_acceptance_relative_mad,
            strict_acceptance_stability=args.strict_acceptance_stability,
            relative_tolerance=args.relative_tolerance,
            max_batch_size=args.max_batch_size,
        )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser):
        parser.add_argument("EXPERIMENT_DIR", type=str)
        parser.add_argument("--output-json", type=str)
        parser.add_argument("--batch-size-var", default="max_concurrency")
        parser.add_argument(
            "--k-var",
            default=DEFAULT_K_VAR,
        )
        parser.add_argument("--objective-var", default="output_throughput")
        parser.add_argument(
            "--objective-direction",
            choices=("maximize", "minimize"),
            default="maximize",
        )
        parser.add_argument("--acceptance-var")
        parser.add_argument("--min-runs", type=int, default=3)
        parser.add_argument("--uncertainty-penalty", type=float, default=1.0)
        parser.add_argument("--max-objective-relative-mad", type=float, default=0.1)
        parser.add_argument("--max-acceptance-relative-mad", type=float, default=0.1)
        parser.add_argument("--strict-acceptance-stability", action="store_true")
        parser.add_argument("--relative-tolerance", type=float, default=0.01)
        parser.add_argument("--max-batch-size", type=int)
        return parser


def run_main(args: SweepTuneSpeculativeKArgs) -> dict[str, object]:
    summary_paths = sorted(args.experiment_dir.rglob("summary.json"))
    records = [
        record for path in summary_paths for record in json.loads(path.read_bytes())
    ]
    result = tune_speculative_k(
        records,
        batch_size_var=args.batch_size_var,
        k_var=args.k_var,
        objective_var=args.objective_var,
        objective_direction=args.objective_direction,
        acceptance_var=args.acceptance_var,
        min_runs=args.min_runs,
        uncertainty_penalty=args.uncertainty_penalty,
        max_objective_relative_mad=args.max_objective_relative_mad,
        max_acceptance_relative_mad=args.max_acceptance_relative_mad,
        strict_acceptance_stability=args.strict_acceptance_stability,
        relative_tolerance=args.relative_tolerance,
        max_batch_size=args.max_batch_size,
    )
    result["input_summary_files"] = [
        str(path.relative_to(args.experiment_dir)) for path in summary_paths
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result["num_speculative_tokens_per_batch_size"]))
    decisions = result["batch_decisions"]
    assert isinstance(decisions, list)
    for decision in decisions:
        assert isinstance(decision, dict)
        print(
            "batch_size={batch_size}: K={selected_k}, "
            "conservative_objective={selected_conservative_objective:.2f}, "
            "vs_best={selected_vs_best_percent:.2f}%, "
            "vs_k0={selected_vs_k0_percent}".format(**decision)
        )
    print(f"Wrote {args.output_json}")
    return result


def main(args: argparse.Namespace):
    run_main(SweepTuneSpeculativeKArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=SweepTuneSpeculativeKArgs.parser_help)
    SweepTuneSpeculativeKArgs.add_cli_args(parser)
    main(parser.parse_args())
