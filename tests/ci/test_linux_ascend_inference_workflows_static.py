# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATHS = (
    REPO_ROOT / ".github/workflows/linux-ascend-inference-smoke.yml",
    REPO_ROOT / ".github/workflows/linux-ascend-inference-regression.yml",
)


def test_pr_inference_workflows_checkout_with_https_without_ssh_known_hosts():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")

        assert "Checkout target repo with retry" in text
        assert "github.event_name == 'pull_request'" in text
        assert "format('https://github.com/{0}.git', github.repository)" in text
        assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text
        assert "https://github.com/vLLM-HUST/ascend-runtime-manager.git" in text

        env_block = text[
            text.index("    env:") : text.index("    steps:")
        ]
        assert "TARGET_REPO_URL: git@github.com:${{ github.repository }}.git" not in (
            env_block
        )
        assert (
            "VLLM_ASCEND_HUST_REPO_URL: git@github.com:vLLM-HUST/"
            "vllm-ascend-hust.git"
        ) not in env_block
        assert (
            "HUST_ASCEND_MANAGER_REPO_URL: git@github.com:vLLM-HUST/"
            "ascend-runtime-manager.git"
        ) not in env_block


def test_inference_workflows_install_no_build_isolation_build_dependencies():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")

        assert "--no-build-isolation" in text
        assert '"setuptools-rust>=1.9.0"' in text
