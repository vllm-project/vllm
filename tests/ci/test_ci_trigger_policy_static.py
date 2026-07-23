# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NPU_WORKFLOWS = (
    REPO_ROOT / ".github/workflows/linux-ascend-inference-smoke.yml",
    REPO_ROOT / ".github/workflows/linux-ascend-inference-regression.yml",
    REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml",
)


def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_npu_pr_workflows_run_when_labeled_and_after_new_commits():
    for workflow_path in NPU_WORKFLOWS:
        workflow = load_workflow(workflow_path)
        pull_request = workflow[True]["pull_request"]

        assert set(pull_request["types"]) == {
            "labeled",
            "synchronize",
            "reopened",
        }


def test_npu_pr_workflows_gate_on_current_labels_not_label_event():
    for workflow_path in NPU_WORKFLOWS:
        text = workflow_path.read_text(encoding="utf-8")

        assert "contains(github.event.pull_request.labels.*.name, 'ready')" in text
        assert "contains(github.event.pull_request.labels.*.name, 'verified')" in text
        assert "github.event.label.name" not in text


def test_pre_commit_uses_available_github_hosted_runner():
    workflow_path = REPO_ROOT / ".github/workflows/pre-commit.yml"
    workflow = load_workflow(workflow_path)
    pre_commit = workflow["jobs"]["pre-commit"]
    text = workflow_path.read_text(encoding="utf-8")

    assert pre_commit["if"] != "false"
    assert pre_commit["runs-on"] == "ubuntu-latest"
    assert "fetch-depth: 0" in text
    assert "--from-ref" in text
    assert "--to-ref" in text
    assert "--all-files" not in text
    assert "--hook-stage manual" not in text
    assert "PRE_COMMIT_FROM_REF:" in text
    assert "PRE_COMMIT_TO_REF:" in text


def test_benchmark_detection_does_not_disable_idle_device_selection():
    workflow_path = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"
    text = workflow_path.read_text(encoding="utf-8")
    detection_step = text[
        text.index("      - name: Detect runner user and Ascend devices") : text.index(
            "      - name: Pre-clean stale checkout state"
        )
    ]

    assert "ASCEND_CI_DETECTED_DEVICES=" in detection_step
    assert 'echo "ASCEND_VISIBLE_DEVICES=0"' in detection_step
    assert 'echo "ASCEND_RT_VISIBLE_DEVICES=0"' in detection_step
    assert '"${#runner_devnodes[@]}" -eq 1' in detection_step


def test_actionlint_knows_ascend_runner_labels():
    config_path = REPO_ROOT / ".github/actionlint.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labels = set(config["self-hosted-runner"]["labels"])

    assert {"ascend", "910b", "docker"} <= labels
