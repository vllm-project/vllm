# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/l3_benchmark_publish_preflight.sh"
)


def run_preflight(
    tmp_path: Path, extra_env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], str]:
    github_env = tmp_path / "github-env"
    env = os.environ.copy()
    env.update(
        {
            "GITHUB_ENV": str(github_env),
            "BENCHMARK_REPO_SLUG": "vLLM-HUST/vllm-hust-benchmark",
        }
    )
    env.update(extra_env)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    env_text = github_env.read_text(encoding="utf-8") if github_env.exists() else ""
    return result, env_text


def test_l3_publish_preflight_skips_when_publish_disabled(tmp_path):
    result, env_text = run_preflight(tmp_path, {"PUBLISH_TO_BENCHMARK_REPO": "0"})

    assert result.returncode == 0
    assert "L3 benchmark repository publish preflight: skipped" in result.stdout
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=skipped" in env_text


def test_l3_publish_preflight_fails_when_credential_missing(tmp_path):
    result, env_text = run_preflight(tmp_path, {"PUBLISH_TO_BENCHMARK_REPO": "1"})

    assert result.returncode == 2
    assert "no cross-repository write credential" in result.stderr
    assert "Target: vLLM-HUST/vllm-hust-benchmark@main" in result.stderr
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=credential-missing" in env_text
    assert "L3_BENCHMARK_PUBLISH_TARGET=vLLM-HUST/vllm-hust-benchmark@main" in env_text


def test_l3_publish_preflight_accepts_ssh_key(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_SSH_KEY": "fake-key",
        },
    )

    assert result.returncode == 0
    assert "L3 benchmark repository publish preflight: ok" in result.stdout
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=ok" in env_text
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL=ssh-key" in env_text


def test_l3_publish_preflight_accepts_github_token(tmp_path):
    result, env_text = run_preflight(
        tmp_path,
        {
            "PUBLISH_TO_BENCHMARK_REPO": "1",
            "BENCHMARK_REPO_GH_TOKEN": "fake-token",
        },
    )

    assert result.returncode == 0
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT=ok" in env_text
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL=token" in env_text
