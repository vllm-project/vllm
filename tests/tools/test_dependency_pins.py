# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("hashed", [False, True])
def test_use_existing_torch_removes_complete_requirements(tmp_path, prefix, hashed):
    """Removing torch must not attach its archive hashes to another package."""
    suffix = " \\\n    --hash=sha256:" + "a" * 64 if hashed else ""
    entries = {
        name: f"{name}==1.0.0{suffix}\n"
        for name in ("torch", "torchvision", "torchmetrics", "other")
    }
    requirements = tmp_path / "requirements" / "test.txt"
    requirements.parent.mkdir()
    requirements.write_text("".join(entries.values()))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/use_existing_torch.py"),
            *(["--prefix"] if prefix else []),
        ],
        cwd=tmp_path,
        check=True,
    )

    expected = (entries["torchmetrics"] if prefix else "") + entries["other"]
    assert requirements.read_text() == expected


@pytest.mark.parametrize(
    "reference",
    [
        "actions/checkout@v7",
        "actions/checkout@main",
        "actions/checkout@1234567",
        "owner/repo/.github/workflows/test.yml@v1",
        "docker://alpine:latest",
    ],
)
def test_action_pin_check_rejects_mutable_references(tmp_path, reference):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(json.dumps({"jobs": {"test": {"uses": reference}}}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/pre_commit/check_github_actions_pins.py"),
            str(workflow),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert reference in result.stdout


def test_action_pin_check_accepts_immutable_and_local_actions(tmp_path):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        json.dumps(
            {
                "jobs": {
                    "reuse": {
                        "uses": "owner/repo/.github/workflows/test.yml@" + "a" * 40
                    },
                    "test": {
                        "steps": [
                            {"uses": "actions/checkout@" + "b" * 40},
                            {"uses": "docker://alpine@sha256:" + "c" * 64},
                            {"uses": "./.github/actions/local"},
                            {"run": "echo uses: actions/checkout@main"},
                        ]
                    },
                }
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/pre_commit/check_github_actions_pins.py"),
            str(workflow),
        ],
        check=True,
    )
