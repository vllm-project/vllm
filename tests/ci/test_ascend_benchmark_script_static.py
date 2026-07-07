# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github/workflows/scripts"


def script_text(name: str) -> str:
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def test_run_ascend_benchmark_propagates_benchmark_repo_publish_env():
    text = script_text("run_ascend_benchmark_ci.sh")

    assert "PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}" in text
    sudo_env_block = text[text.index("export_sudo_preserved_env_vars()") :]
    assert "PUBLISH_TO_BENCHMARK_REPO" in sudo_env_block
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in text
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then' in text
    assert 'BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \\' in text
    assert 'BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \\' in text


def test_perfgate_store_baseline_cleans_worktree_on_exit():
    text = script_text("perfgate_store_baseline.sh")

    assert "cleanup() {" in text
    assert 'git worktree remove "$WORKTREE_DIR" --force' in text
    assert 'rm -rf "$WORKTREE_DIR"' in text
    assert "trap cleanup EXIT" in text


def test_benchmark_snapshot_sync_explains_missing_write_credentials():
    text = script_text("sync_benchmark_snapshots_to_github.sh")

    assert "L3 benchmark repository publication is enabled" in text
    assert "no cross-repository write credential is available" in text
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in text
    assert "VLLM_HUST_BENCHMARK_GH_TOKEN" in text
    assert "Benchmark repo publish target:" in text


def test_e2e_inference_scripts_retry_http_requests_and_print_server_log():
    for script_name in (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
    ):
        text = script_text(script_name)

        assert "print_server_log_tail() {" in text
        assert "curl_with_server_log() {" in text
        assert "E2E_HTTP_REQUEST_ATTEMPTS" in text
        assert "failed after ${max_attempts} attempts" in text
        assert (
            'done\n\ncurl -fsS "http://$HOST:$PORT/v1/models" >/dev/null'
            not in text
        )
        assert "vLLM models endpoint readiness confirmation" in text
        assert "curl_with_server_log" in text[text.index("completion_response=$(mktemp)") :]
