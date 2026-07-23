#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenarioSelection:
    scenario: str
    reason: str
    mode: str
    scenarios: tuple[str, ...] = ()
    trigger_label: str = ""
    dataset_path: str = ""
    constraints_file: str = ""
    same_spec_spec_file: str = ""
    same_spec_constraints_file: str = ""
    perfgate_mode: str = ""


LABEL_SCENARIOS: dict[str, ScenarioSelection] = {
    "ascend-targeted-test": ScenarioSelection(
        scenario="random-online",
        reason=(
            "PR label ascend-targeted-test selected random-online targeted verification"
        ),
        mode="l2-targeted",
        trigger_label="ascend-targeted-test",
    ),
    "ascend-targeted-required": ScenarioSelection(
        scenario="random-online",
        reason=(
            "PR label ascend-targeted-required selected required "
            "random-online targeted verification"
        ),
        mode="l2-required",
        trigger_label="ascend-targeted-required",
        perfgate_mode="enforce",
    ),
    "ascend-benchmark:l2-random": ScenarioSelection(
        scenario="random-online",
        reason=(
            "PR label ascend-benchmark:l2-random selected random-online "
            "targeted verification"
        ),
        mode="l2-targeted",
        trigger_label="ascend-benchmark:l2-random",
    ),
    "ascend-benchmark:l2-sharegpt": ScenarioSelection(
        scenario="sharegpt-online",
        reason=(
            "PR label ascend-benchmark:l2-sharegpt selected sharegpt-online "
            "targeted verification"
        ),
        mode="l2-targeted",
        trigger_label="ascend-benchmark:l2-sharegpt",
    ),
}

SUPPORTED_MANUAL_SCENARIOS = {"random-online", "sharegpt-online"}
L2_LABEL_PREFIX = "ascend-benchmark:l2-"
TARGETED_LABEL_PREFIX = "ascend-targeted-"


def parse_labels(raw_labels: str) -> list[str]:
    raw_labels = raw_labels.strip()
    if not raw_labels:
        return []
    try:
        payload = json.loads(raw_labels)
    except json.JSONDecodeError:
        return [item.strip() for item in raw_labels.split(",") if item.strip()]
    if not isinstance(payload, list):
        raise ValueError(
            "PR labels payload must be a JSON array or comma-separated list"
        )
    labels: list[str] = []
    for item in payload:
        if isinstance(item, str):
            label = item.strip()
        elif isinstance(item, dict):
            label = str(item.get("name") or "").strip()
        else:
            label = ""
        if label:
            labels.append(label)
    return labels


def parse_scenarios(raw_scenarios: str) -> tuple[str, ...]:
    scenarios = tuple(
        scenario.strip()
        for scenario in raw_scenarios.replace("\n", ",").split(",")
        if scenario.strip()
    )
    if len(set(scenarios)) != len(scenarios):
        duplicates = sorted(
            {scenario for scenario in scenarios if scenarios.count(scenario) > 1}
        )
        raise ValueError(
            "duplicate benchmark scenario(s): " + ", ".join(duplicates)
        )
    return scenarios


def _load_registered_scenarios(
    benchmark_repo: str,
    hardware_chip_model: str,
) -> set[str]:
    registry_file = (
        Path(benchmark_repo)
        / "src"
        / "vllm_hust_benchmark"
        / "data"
        / "perfgate_spec_registry.json"
    )
    if not registry_file.is_file():
        raise ValueError(f"perfgate spec registry not found: {registry_file}")
    payload = json.loads(registry_file.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{registry_file}: perfgate spec registry must contain entries")
    scenarios: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("hardware_chip_model") or "").strip() != hardware_chip_model:
            continue
        scenario = str(entry.get("scenario") or "").strip()
        if scenario:
            scenarios.add(scenario)
    return scenarios


def _validate_configured_scenarios(
    scenarios: tuple[str, ...],
    *,
    benchmark_repo: str,
    hardware_chip_model: str,
) -> None:
    if not scenarios or not benchmark_repo:
        return
    registered = _load_registered_scenarios(benchmark_repo, hardware_chip_model)
    unsupported = sorted(scenario for scenario in scenarios if scenario not in registered)
    if unsupported:
        supported = ", ".join(sorted(registered)) or "none"
        raise ValueError(
            "No perfgate spec registered for configured benchmark scenario(s): "
            + ", ".join(unsupported)
            + f" on hardware_chip_model={hardware_chip_model}; registered: {supported}"
        )


def _selection_scenarios(selection: ScenarioSelection) -> tuple[str, ...]:
    return selection.scenarios or (selection.scenario,)


def _validate_selection(
    selection: ScenarioSelection,
    *,
    require_sharegpt_inputs: bool = True,
) -> ScenarioSelection:
    if not require_sharegpt_inputs:
        return selection
    if "sharegpt-online" not in _selection_scenarios(selection):
        return selection
    missing = []
    if not selection.dataset_path:
        missing.append("BENCH_DATASET_PATH")
    if not selection.constraints_file:
        missing.append("BENCH_CONSTRAINTS_FILE")
    if missing:
        raise ValueError("sharegpt-online requires " + ", ".join(missing))
    return selection


def select_scenario(
    *,
    event_name: str,
    manual_scenario: str,
    pr_labels: list[str],
    default_scenario: str,
    default_dataset_path: str,
    default_constraints_file: str,
    same_spec_spec_file: str,
    same_spec_constraints_file: str,
    configured_scenarios: str = "",
    benchmark_repo: str = "",
    hardware_chip_model: str = "910B2",
) -> ScenarioSelection:
    hardware_chip_model = hardware_chip_model or "910B2"
    scenario_list = parse_scenarios(configured_scenarios)
    if len(scenario_list) > 1 and same_spec_spec_file:
        raise ValueError(
            "configured benchmark scenario list cannot be combined with "
            "SAME_SPEC_SPEC_FILE because each scenario must resolve its own "
            "perfgate spec from the registry"
        )
    _validate_configured_scenarios(
        scenario_list,
        benchmark_repo=benchmark_repo,
        hardware_chip_model=hardware_chip_model,
    )
    if scenario_list:
        return _validate_selection(
            ScenarioSelection(
                scenario=scenario_list[0],
                scenarios=scenario_list,
                reason=(
                    f"{event_name or 'unknown'} event selected configured "
                    "benchmark scenario list"
                ),
                mode="multi-scenario",
                dataset_path=default_dataset_path,
                constraints_file=default_constraints_file,
                same_spec_spec_file=same_spec_spec_file,
                same_spec_constraints_file=same_spec_constraints_file,
            ),
            require_sharegpt_inputs=False,
        )

    if event_name == "workflow_dispatch":
        scenario = manual_scenario.strip() or default_scenario
        if scenario not in SUPPORTED_MANUAL_SCENARIOS:
            supported = ", ".join(sorted(SUPPORTED_MANUAL_SCENARIOS))
            raise ValueError(
                f"unsupported manual benchmark_scenario={scenario!r}; "
                f"supported: {supported}"
            )
        return _validate_selection(
            ScenarioSelection(
                scenario=scenario,
                reason=f"workflow_dispatch input selected {scenario}",
                mode="manual",
                dataset_path=default_dataset_path,
                constraints_file=default_constraints_file,
                same_spec_spec_file=same_spec_spec_file,
                same_spec_constraints_file=same_spec_constraints_file,
            )
        )

    if event_name == "pull_request":
        unknown_l2_labels = [
            label
            for label in pr_labels
            if (
                label.startswith(L2_LABEL_PREFIX)
                or label.startswith(TARGETED_LABEL_PREFIX)
            )
            and label not in LABEL_SCENARIOS
        ]
        if unknown_l2_labels:
            supported = ", ".join(sorted(LABEL_SCENARIOS))
            raise ValueError(
                "unsupported L2 benchmark scenario label(s): "
                + ", ".join(unknown_l2_labels)
                + f"; supported: {supported}"
            )
        matches = [label for label in pr_labels if label in LABEL_SCENARIOS]
        if len(matches) > 1:
            raise ValueError(
                "multiple L2 benchmark scenario labels found: " + ", ".join(matches)
            )
        if matches:
            selected = LABEL_SCENARIOS[matches[0]]
            return _validate_selection(
                ScenarioSelection(
                    scenario=selected.scenario,
                    reason=selected.reason,
                    mode=selected.mode,
                    trigger_label=selected.trigger_label,
                    dataset_path=default_dataset_path,
                    constraints_file=default_constraints_file,
                    same_spec_spec_file=same_spec_spec_file,
                    same_spec_constraints_file=same_spec_constraints_file,
                    perfgate_mode=selected.perfgate_mode,
                )
            )
        return _validate_selection(
            ScenarioSelection(
                scenario=default_scenario,
                reason=(
                    "no L2 benchmark label found; using default PR benchmark scenario"
                ),
                mode="l1-smoke",
                dataset_path=default_dataset_path,
                constraints_file=default_constraints_file,
                same_spec_spec_file=same_spec_spec_file,
                same_spec_constraints_file=same_spec_constraints_file,
            )
        )

    return _validate_selection(
        ScenarioSelection(
            scenario=default_scenario,
            reason=(
                f"{event_name or 'unknown'} event uses configured default "
                "benchmark scenario"
            ),
            mode="default",
            dataset_path=default_dataset_path,
            constraints_file=default_constraints_file,
            same_spec_spec_file=same_spec_spec_file,
            same_spec_constraints_file=same_spec_constraints_file,
        )
    )


def _github_multiline_delimiter(key: str, value: str) -> str:
    delimiter = f"__{key}_EOF__"
    while delimiter in value:
        delimiter = f"_{delimiter}_"
    return delimiter


def write_env_file(path: str, values: dict[str, str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            delimiter = _github_multiline_delimiter(key, value)
            handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve Ascend benchmark scenario for L1/L2 CI runs."
    )
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument(
        "--manual-scenario",
        default=os.environ.get("INPUT_BENCHMARK_SCENARIO", ""),
    )
    parser.add_argument(
        "--configured-scenarios",
        default=os.environ.get("BENCH_SCENARIOS", ""),
    )
    parser.add_argument("--pr-labels", default=os.environ.get("PR_LABELS", ""))
    parser.add_argument(
        "--default-scenario",
        default=os.environ.get("BENCH_SCENARIO", "random-online"),
    )
    parser.add_argument(
        "--default-dataset-path",
        default=os.environ.get("BENCH_DATASET_PATH", ""),
    )
    parser.add_argument(
        "--default-constraints-file",
        default=os.environ.get("BENCH_CONSTRAINTS_FILE", ""),
    )
    parser.add_argument(
        "--same-spec-spec-file",
        default=os.environ.get("SAME_SPEC_SPEC_FILE", ""),
    )
    parser.add_argument(
        "--same-spec-constraints-file",
        default=os.environ.get("SAME_SPEC_CONSTRAINTS_FILE", ""),
    )
    parser.add_argument(
        "--benchmark-repo",
        default=os.environ.get("VLLM_HUST_BENCHMARK_REPO", ""),
    )
    parser.add_argument(
        "--hardware-chip-model",
        default=os.environ.get("HARDWARE_CHIP_MODEL") or "910B2",
    )
    parser.add_argument("--github-env", default=os.environ.get("GITHUB_ENV", ""))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    args = parser.parse_args()

    try:
        labels = parse_labels(args.pr_labels)
        selection = select_scenario(
            event_name=args.event_name,
            manual_scenario=args.manual_scenario,
            configured_scenarios=args.configured_scenarios,
            pr_labels=labels,
            default_scenario=args.default_scenario,
            default_dataset_path=args.default_dataset_path,
            default_constraints_file=args.default_constraints_file,
            same_spec_spec_file=args.same_spec_spec_file,
            same_spec_constraints_file=args.same_spec_constraints_file,
            benchmark_repo=args.benchmark_repo,
            hardware_chip_model=args.hardware_chip_model,
        )
    except ValueError as exc:
        print(f"Ascend benchmark scenario resolution failed: {exc}", file=sys.stderr)
        return 2

    scenarios = _selection_scenarios(selection)
    values = {
        "BENCH_SCENARIO": selection.scenario,
        "BENCH_SCENARIOS": ",".join(scenarios),
        "BENCH_SCENARIO_COUNT": str(len(scenarios)),
        "BENCH_DATASET_PATH": selection.dataset_path,
        "BENCH_CONSTRAINTS_FILE": selection.constraints_file,
        "SAME_SPEC_SPEC_FILE": selection.same_spec_spec_file,
        "SAME_SPEC_CONSTRAINTS_FILE": selection.same_spec_constraints_file,
        "L2_SCENARIO_MODE": selection.mode,
        "L2_SCENARIO_LABEL": selection.trigger_label,
        "L2_SCENARIO_REASON": selection.reason,
    }
    if selection.perfgate_mode:
        values["PERFGATE_MODE"] = selection.perfgate_mode
    write_env_file(args.github_env, values)
    write_env_file(args.github_output, values)
    print(f"Resolved Ascend benchmark scenario: {selection.scenario}")
    if len(scenarios) > 1:
        print("Resolved Ascend benchmark scenarios: " + ", ".join(scenarios))
    print(f"Mode: {selection.mode}")
    if selection.trigger_label:
        print(f"Trigger label: {selection.trigger_label}")
    print(f"Reason: {selection.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
