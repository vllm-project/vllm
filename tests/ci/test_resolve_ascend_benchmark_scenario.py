# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

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
