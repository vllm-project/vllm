# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/resolve_ascend_benchmark_scenario.py"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location(
        "resolve_ascend_benchmark_scenario",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pr_without_l2_label_uses_default_scenario():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    assert selected.scenario == "random-online"
    assert selected.mode == "l1-smoke"
    assert selected.trigger_label == ""
    assert "no L2 benchmark label" in selected.reason


def test_issue_comment_event_falls_back_to_default_scenario():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="issue_comment",
        manual_scenario="",
        pr_labels=[],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    assert selected.scenario == "random-online"
    assert selected.mode == "default"
    assert selected.trigger_label == ""
    assert (
        "issue_comment event uses configured default benchmark scenario"
        in selected.reason
    )


def test_parse_labels_accepts_github_label_objects():
    resolver = load_resolver()

    labels = resolver.parse_labels(
        '[{"name":"ready"},{"name":"ascend-benchmark:l2-random"}]'
    )

    assert labels == ["ready", "ascend-benchmark:l2-random"]


def test_parse_labels_rejects_non_list_json_payload():
    resolver = load_resolver()

    try:
        resolver.parse_labels('{"name":"ready"}')
    except ValueError as exc:
        assert "JSON array" in str(exc)
    else:
        raise AssertionError("expected non-list label JSON to be rejected")


def test_parse_scenarios_accepts_comma_and_newline_separated_values():
    resolver = load_resolver()

    scenarios = resolver.parse_scenarios(
        "random-online, sharegpt-throughput\nagent-research-online"
    )

    assert scenarios == (
        "random-online",
        "sharegpt-throughput",
        "agent-research-online",
    )


def test_parse_scenarios_rejects_duplicates():
    resolver = load_resolver()

    try:
        resolver.parse_scenarios("random-online,random-online")
    except ValueError as exc:
        assert "duplicate benchmark scenario" in str(exc)
        assert "random-online" in str(exc)
    else:
        raise AssertionError("expected duplicate scenarios to be rejected")


def test_configured_scenario_list_rejects_registry_misses(tmp_path):
    resolver = load_resolver()
    registry_dir = tmp_path / "src" / "vllm_hust_benchmark" / "data"
    registry_dir.mkdir(parents=True)
    (registry_dir / "perfgate_spec_registry.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "scenario": "random-online",
                        "hardware_chip_model": "910B2",
                        "spec_file": "docs/official-baselines/random.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        resolver.select_scenario(
            event_name="pull_request",
            manual_scenario="",
            pr_labels=["ready"],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="",
            same_spec_constraints_file="",
            configured_scenarios="random-online,unknown-scenario",
            benchmark_repo=str(tmp_path),
            hardware_chip_model="910B2",
        )
    except ValueError as exc:
        message = str(exc)
        assert "No perfgate spec registered" in message
        assert "unknown-scenario" in message
        assert "random-online" in message
    else:
        raise AssertionError("expected unsupported scenarios to be rejected")


def test_configured_scenario_list_rejects_explicit_same_spec_override():
    resolver = load_resolver()

    try:
        resolver.select_scenario(
            event_name="pull_request",
            manual_scenario="",
            pr_labels=["ready"],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="/tmp/spec.json",
            same_spec_constraints_file="",
            configured_scenarios="random-online,sharegpt-throughput",
        )
    except ValueError as exc:
        message = str(exc)
        assert "cannot be combined with SAME_SPEC_SPEC_FILE" in message
        assert "each scenario must resolve its own perfgate spec" in message
    else:
        raise AssertionError("expected explicit same-spec override to be rejected")


def test_configured_scenario_list_allows_sharegpt_without_legacy_inputs(tmp_path):
    resolver = load_resolver()
    registry_dir = tmp_path / "src" / "vllm_hust_benchmark" / "data"
    registry_dir.mkdir(parents=True)
    (registry_dir / "perfgate_spec_registry.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "scenario": "random-online",
                        "hardware_chip_model": "910B2",
                        "spec_file": "docs/official-baselines/random.json",
                    },
                    {
                        "scenario": "sharegpt-online",
                        "hardware_chip_model": "910B2",
                        "spec_file": "docs/official-baselines/sharegpt.json",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
        configured_scenarios="random-online,sharegpt-online",
        benchmark_repo=str(tmp_path),
        hardware_chip_model="910B2",
    )

    assert selected.scenarios == ("random-online", "sharegpt-online")
    assert selected.mode == "multi-scenario"


def test_configured_scenario_list_uses_default_chip_for_empty_value(tmp_path):
    resolver = load_resolver()
    registry_dir = tmp_path / "src" / "vllm_hust_benchmark" / "data"
    registry_dir.mkdir(parents=True)
    (registry_dir / "perfgate_spec_registry.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "scenario": "random-online",
                        "hardware_chip_model": "910B2",
                        "spec_file": "docs/official-baselines/random.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
        configured_scenarios="random-online",
        benchmark_repo=str(tmp_path),
        hardware_chip_model="",
    )

    assert selected.scenario == "random-online"
    assert selected.scenarios == ("random-online",)


def test_configured_scenario_list_takes_precedence_over_labels():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ascend-targeted-required"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
        configured_scenarios="random-online,sharegpt-throughput",
    )

    assert selected.scenario == "random-online"
    assert selected.scenarios == ("random-online", "sharegpt-throughput")
    assert selected.mode == "multi-scenario"
    assert selected.perfgate_mode == ""
    assert "configured benchmark scenario list" in selected.reason


def test_pr_l2_sharegpt_label_selects_sharegpt_scenario():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready", "ascend-benchmark:l2-sharegpt"],
        default_scenario="random-online",
        default_dataset_path="/data/sharegpt.jsonl",
        default_constraints_file="/data/constraints.json",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    assert selected.scenario == "sharegpt-online"
    assert selected.mode == "l2-targeted"
    assert selected.trigger_label == "ascend-benchmark:l2-sharegpt"
    assert selected.dataset_path == "/data/sharegpt.jsonl"
    assert "l2-sharegpt" in selected.reason


def test_pr_targeted_test_label_selects_l2_targeted_mode():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready", "ascend-targeted-test"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    assert selected.scenario == "random-online"
    assert selected.mode == "l2-targeted"
    assert selected.trigger_label == "ascend-targeted-test"
    assert selected.perfgate_mode == ""


def test_pr_targeted_required_label_selects_required_mode():
    resolver = load_resolver()
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ready", "ascend-targeted-required"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    assert selected.scenario == "random-online"
    assert selected.mode == "l2-required"
    assert selected.trigger_label == "ascend-targeted-required"
    assert selected.perfgate_mode == "enforce"


def test_pr_l2_sharegpt_requires_dataset_and_constraints():
    resolver = load_resolver()

    try:
        resolver.select_scenario(
            event_name="pull_request",
            manual_scenario="",
            pr_labels=["ascend-benchmark:l2-sharegpt"],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="",
            same_spec_constraints_file="",
        )
    except ValueError as exc:
        message = str(exc)
        assert "sharegpt-online requires" in message
        assert "BENCH_DATASET_PATH" in message
        assert "BENCH_CONSTRAINTS_FILE" in message
    else:
        raise AssertionError("expected sharegpt-online missing config to be rejected")


def test_multiple_l2_labels_are_rejected():
    resolver = load_resolver()

    try:
        resolver.select_scenario(
            event_name="pull_request",
            manual_scenario="",
            pr_labels=["ascend-benchmark:l2-random", "ascend-benchmark:l2-sharegpt"],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="",
            same_spec_constraints_file="",
        )
    except ValueError as exc:
        assert "multiple L2 benchmark scenario labels" in str(exc)
    else:
        raise AssertionError("expected multiple L2 labels to be rejected")


def test_unknown_l2_label_is_rejected():
    resolver = load_resolver()

    try:
        resolver.select_scenario(
            event_name="pull_request",
            manual_scenario="",
            pr_labels=["ascend-targeted-moe"],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="",
            same_spec_constraints_file="",
        )
    except ValueError as exc:
        assert "unsupported L2 benchmark scenario label" in str(exc)
        assert "ascend-targeted-test" in str(exc)
    else:
        raise AssertionError("expected unknown L2 label to be rejected")


def test_manual_unsupported_scenario_is_rejected():
    resolver = load_resolver()

    try:
        resolver.select_scenario(
            event_name="workflow_dispatch",
            manual_scenario="unknown-scenario",
            pr_labels=[],
            default_scenario="random-online",
            default_dataset_path="",
            default_constraints_file="",
            same_spec_spec_file="",
            same_spec_constraints_file="",
        )
    except ValueError as exc:
        assert "unsupported manual benchmark_scenario" in str(exc)
    else:
        raise AssertionError("expected unsupported manual scenario to be rejected")


def test_write_env_file_uses_collision_safe_multiline_format(tmp_path):
    resolver = load_resolver()
    env_file = tmp_path / "github-env"

    resolver.write_env_file(
        str(env_file),
        {
            "L2_SCENARIO_REASON": "first line\ncontains __L2_SCENARIO_REASON_EOF__",
        },
    )

    content = env_file.read_text(encoding="utf-8")

    assert content.startswith("L2_SCENARIO_REASON<<_")
    assert "first line\ncontains __L2_SCENARIO_REASON_EOF__\n" in content
    assert content.rstrip().endswith("_")


def test_required_label_writes_perfgate_enforce_mode(tmp_path):
    resolver = load_resolver()
    env_file = tmp_path / "github-env"
    selected = resolver.select_scenario(
        event_name="pull_request",
        manual_scenario="",
        pr_labels=["ascend-targeted-required"],
        default_scenario="random-online",
        default_dataset_path="",
        default_constraints_file="",
        same_spec_spec_file="",
        same_spec_constraints_file="",
    )

    values = {
        "L2_SCENARIO_MODE": selected.mode,
        "L2_SCENARIO_LABEL": selected.trigger_label,
    }
    if selected.perfgate_mode:
        values["PERFGATE_MODE"] = selected.perfgate_mode
    resolver.write_env_file(str(env_file), values)

    content = env_file.read_text(encoding="utf-8")

    assert "L2_SCENARIO_MODE<<__L2_SCENARIO_MODE_EOF__\nl2-required\n" in content
    assert (
        "L2_SCENARIO_LABEL<<__L2_SCENARIO_LABEL_EOF__\nascend-targeted-required\n"
        in content
    )
    assert "PERFGATE_MODE<<__PERFGATE_MODE_EOF__\nenforce\n" in content
