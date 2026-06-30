# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/sync_benchmark_snapshots_to_github.sh"
)


def run(
    cmd: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def init_bare_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    remote.mkdir()
    run(["git", "init", "--bare", str(remote)], tmp_path)
    run(["git", "clone", str(remote), str(seed)], tmp_path)
    run(["git", "config", "user.name", "Test"], seed)
    run(["git", "config", "user.email", "test@example.com"], seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    run(["git", "add", "README.md"], seed)
    run(["git", "commit", "-m", "seed"], seed)
    run(["git", "push", "origin", "HEAD:main"], seed)
    return remote, seed


def write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$1" != "-m" \
  || "$2" != "vllm_hust_benchmark.cli" \
  || "$3" != "publish-website" ]]; then
  echo "unexpected fake python invocation: $*" >&2
  exit 2
fi
shift 3
output_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$output_dir"
printf '{}\\n' > "$output_dir/leaderboard_single.json"
printf '{}\\n' > "$output_dir/leaderboard_multi.json"
printf '{}\\n' > "$output_dir/leaderboard_compare.json"
printf '{}\\n' > "$output_dir/last_updated.json"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def test_sync_benchmark_snapshots_verifies_published_commit(tmp_path):
    remote, _seed = init_bare_remote(tmp_path)
    benchmark_repo = tmp_path / "benchmark"
    website_repo = tmp_path / "website"
    vllm_hust_repo = tmp_path / "vllm-hust"
    submission = tmp_path / "submission"
    github_env = tmp_path / "github-env"
    fake_python = write_fake_python(tmp_path)

    run(["git", "clone", str(remote), str(benchmark_repo)], tmp_path)
    run(["git", "config", "user.name", "Test"], benchmark_repo)
    run(["git", "config", "user.email", "test@example.com"], benchmark_repo)
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts/aggregate_results.py").write_text(
        "# fake\n", encoding="utf-8"
    )
    vllm_hust_repo.mkdir()
    (vllm_hust_repo / "pyproject.toml").write_text(
        "[project]\nname='fake'\n", encoding="utf-8"
    )
    submission.mkdir()
    (submission / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")
    (submission / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ALLOW_LOCAL_GIT_RESET": "1",
            "BENCHMARK_REPO_DIR": str(benchmark_repo),
            "BENCHMARK_REPO_REMOTE": "origin",
            "BENCHMARK_REPO_SLUG": "local/benchmark",
            "CURRENT_SUBMISSION_DIR": str(submission),
            "GITHUB_ENV": str(github_env),
            "LOCAL_SNAPSHOT_OUTPUT_DIR": str(tmp_path / "local-snapshots"),
            "PYTHON_BIN": str(fake_python),
            "RUN_ID": "ci-test",
            "SNAPSHOT_TARGET_BRANCH": "main",
            "VLLM_HUST_REPO_DIR": str(vllm_hust_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    env_text = github_env.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified benchmark publication" in result.stdout
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=verified" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT=" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH=submissions/ci-test" in env_text
    assert "GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH=leaderboard-data/snapshots" in env_text
