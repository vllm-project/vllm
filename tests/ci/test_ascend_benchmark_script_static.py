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


def test_same_spec_benchmark_failure_prints_server_log_tail():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "same_spec_server_log=$RESULT_ROOT/server.stdout.log" in same_spec_block
    assert "print_same_spec_server_log_tail() {" in same_spec_block
    assert "current same-spec vLLM server log tail" in same_spec_block
    assert 'collect_ascend_diagnostics "same-spec-current-failure"' in same_spec_block
    assert 'return "$same_spec_status"' in same_spec_block


def test_same_spec_pr_preview_uses_ascend_compatibility_overlay():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}" in text
    assert "prepare_same_spec_pr_preview_compat_file() {" in same_spec_block
    assert 'server_parameters["no_enable_chunked_prefill"] = True' in same_spec_block
    assert 'server_parameters["no_enable_prefix_caching"] = True' in same_spec_block
    assert 'client_parameters.setdefault("temperature", 0)' in same_spec_block
    assert '${GITHUB_EVENT_NAME:-}" == "pull_request"' in same_spec_block
    assert '${GITHUB_EVENT_NAME:-}" == "issue_comment"' in same_spec_block
    assert '"$effective_same_spec_file"' in same_spec_block


def test_same_spec_runner_resolves_spec_from_shared_registry():
    text = script_text("run_ascend_benchmark_ci.sh")
    same_spec_block = text[text.index("run_same_spec_current_benchmark() {") :]

    assert "SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}" in text
    assert "vllm_hust_benchmark.perfgate_specs resolve" in same_spec_block
    assert '--scenario "$BENCH_SCENARIO"' in same_spec_block
    assert '--hardware-chip-model "$HARDWARE_CHIP_MODEL"' in same_spec_block
    assert '--repo-root "$VLLM_HUST_BENCHMARK_REPO"' in same_spec_block


def test_e2e_inference_scripts_use_python_http_probe_with_server_log():
    for script_name in (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
    ):
        text = script_text(script_name)

        assert "print_server_log_tail() {" in text
        assert "http_with_server_log() {" in text
        assert "e2e_http_request.py" in text
        assert '"$PYTHON_BIN" "$HTTP_REQUEST_SCRIPT"' in text
        assert "E2E_HTTP_REQUEST_ATTEMPTS" in text
        assert "E2E_HTTP_REQUEST_TIMEOUT_SECONDS" in text
        assert "else\n      rc=$?\n    fi" in text
        assert "failed after ${max_attempts} attempts" in text
        assert "curl -fsS" not in text
        assert "vLLM models endpoint readiness confirmation" in text
        assert (
            "http_with_server_log"
            in text[text.index("completion_response=$(mktemp)") :]
        )


def test_ascend_server_readiness_windows_allow_cold_start():
    for script_name in (
        "run_e2e_serve_smoke.sh",
        "run_e2e_inference_regression.sh",
    ):
        text = script_text(script_name)

        assert "SERVER_READY_MAX_ATTEMPTS=${SERVER_READY_MAX_ATTEMPTS:-300}" in text
        assert 'seq 1 "$SERVER_READY_MAX_ATTEMPTS"' in text
        assert '"$attempt" -eq "$SERVER_READY_MAX_ATTEMPTS"' in text

    benchmark_text = script_text("run_ascend_benchmark_ci.sh")
    assert (
        "SAME_SPEC_READY_TIMEOUT_SECONDS=${SAME_SPEC_READY_TIMEOUT_SECONDS:-1200}"
    ) in benchmark_text


def test_benchmark_pins_named_runner_to_its_npu():
    text = script_text("run_ascend_benchmark_ci.sh")

    assert '"${RUNNER_NAME:-}" =~ npu([0-9]+)$' in text
    assert 'runner_physical_device="${BASH_REMATCH[1]}"' in text
    assert "runner_devnodes=(/dev/davinci[0-9]*)" in text
    assert "export ASCEND_RT_VISIBLE_DEVICES=0" in text
    assert 'export ASCEND_RT_VISIBLE_DEVICES="$runner_physical_device"' in text


def test_perfgate_baseline_fetch_bounds_git_network_waits():
    text = script_text("perfgate_fetch_baseline.sh")

    assert "GIT_NETWORK_ATTEMPTS=${GIT_NETWORK_ATTEMPTS:-3}" in text
    assert "GIT_NETWORK_TIMEOUT_SECONDS=${GIT_NETWORK_TIMEOUT_SECONDS:-90}" in text
    assert 'timeout --foreground "${GIT_NETWORK_TIMEOUT_SECONDS}s"' in text
    assert "run_git_network ls-remote" in text
    assert "run_git_network clone" in text
