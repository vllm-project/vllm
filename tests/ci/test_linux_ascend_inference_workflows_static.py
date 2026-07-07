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
        assert "TARGET_REPO_URL: https://github.com/${{ github.repository }}.git" in text
        assert "format('refs/pull/{0}/merge', github.event.pull_request.number)" in text
        assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text
        assert "https://github.com/vLLM-HUST/ascend-runtime-manager.git" in text

        env_block = text[
            text.index("    env:") : text.index("    steps:")
        ]
        assert "git@github.com" not in env_block
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


def test_inference_workflows_remove_stale_torchvision_and_verify_server_import():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        install_step = text[
            text.index("      - name: Prepare Ascend runtime and install current checkout") :
            text.index("      - name: Verify installation")
        ]
        verify_step = text[
            text.index("      - name: Verify installation") :
            text.index("      - name: Run real")
        ]

        assert '"$PYTHON_BIN" -m pip uninstall -y torchvision' in install_step
        assert "import vllm.entrypoints.openai.api_server" in verify_step
        assert "torchvision importable:" in verify_step


def test_inference_workflows_install_matching_ascend_torch_stack():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")

        assert "Install Ascend torch stack for preflight" in text
        assert '"$PYTHON_BIN" -c "import torch, torch_npu;' in text
        assert '"$PYTHON_BIN" -m pip install "torch==2.9.0" "torch-npu==2.9.0"' in text


def test_inference_workflows_disable_backend_autoload_during_install_checks():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        install_step = text[
            text.index("      - name: Prepare Ascend runtime and install current checkout") :
            text.index("      - name: Verify installation")
        ]
        verify_step = text[
            text.index("      - name: Verify installation") :
            text.index("      - name: Run real")
        ]

        assert 'TORCH_DEVICE_BACKEND_AUTOLOAD: "0"' in install_step
        assert 'TORCH_DEVICE_BACKEND_AUTOLOAD: "0"' in verify_step


def test_inference_workflows_fetch_target_sha_without_default_branch_clone():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        checkout_step = text[
            text.index("      - name: Checkout target repo with retry") :
            text.index("      - name: Prepare Hugging Face cache directories")
        ]

        assert "git clone --depth 1 \"$repo_url\" \"$temp_dir\"" not in checkout_step
        assert "git -C \"$temp_dir\" init" in checkout_step
        assert "git -C \"$temp_dir\" remote add origin \"$repo_url\"" in checkout_step
        assert (
            'git -C "$temp_dir" -c protocol.version=2 fetch --no-tags --depth 1 '
            'origin "$target_ref"'
        ) in checkout_step
        assert 'git -C "$temp_dir" checkout --force "$target_sha"' in checkout_step
        assert (
            'clone_with_retry "$TARGET_REPO_URL" "$GITHUB_WORKSPACE" '
            '"$TARGET_REPO_REF" "$TARGET_REPO_SHA"'
        ) in checkout_step


def test_inference_workflows_cleanup_tolerates_checkout_failure():
    for workflow_path in WORKFLOW_PATHS:
        text = workflow_path.read_text(encoding="utf-8")
        cleanup_step = text[
            text.index("      - name: Cleanup leftover Ascend CI processes") :
        ]

        assert '[[ -f "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh" ]]' in cleanup_step
        assert '[[ -f ".github/workflows/scripts/cleanup_ascend_ci_processes.sh" ]]' in cleanup_step
        assert "target checkout is unavailable" in cleanup_step
