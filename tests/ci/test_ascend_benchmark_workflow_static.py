# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend-benchmark-leaderboard.yml"
)


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def workflow_yaml() -> dict:
    return yaml.safe_load(workflow_text())


def test_workflow_dispatch_input_count_stays_within_github_limit():
    inputs = workflow_yaml()[True]["workflow_dispatch"]["inputs"]

    assert len(inputs) <= 10
    assert "metadata_lengths" in inputs
    assert "input_length" not in inputs
    assert "output_length" not in inputs


def test_pr_comment_update_job_has_job_level_issues_write_permission():
    text = workflow_text()

    comment_step = text.index("      - name: Update PR benchmark comment")
    job_permissions = text.rindex("    permissions:", 0, comment_step)
    permissions_block = text[
        job_permissions : text.index("    runs-on:", job_permissions)
    ]

    assert "issues: write" in permissions_block
    assert "pull-requests: write" in permissions_block


def test_issue_comment_non_pr_commands_receive_denial_feedback():
    text = workflow_text()

    assert "if: ${{ github.event.comment.body != '' }}" in text
    assert '--event-payload "$GITHUB_EVENT_PATH"' in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "persist-credentials: false" in text
    assert "const safeReason = reason.replace" in text
    assert "`Reason: ${safeReason}`" in text


def test_fork_pr_security_note_is_blocking():
    text = workflow_text()

    assert "Skipping Ascend benchmark on fork PRs" in text
    assert (
        "exit 1"
        in text[
            text.index("fork-pr-security-note:") : text.index("  ascend-benchmark:")
        ]
    )


def test_main_baseline_store_has_spec_file_and_benchmark_repo_checkout():
    text = workflow_text()
    store_job = text[text.index("  store-main-perfgate-baseline:") :]

    assert "TARGET_REPO_SHA: ${{ github.sha }}" in store_job
    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}"
    ) in store_job
    assert (
        "RESULT_ROOT: ${{ github.workspace }}/.benchmarks/ci/ci-"
        "${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}"
    ) in store_job
    assert "PERFGATE_SPEC_FILE:" not in store_job
    assert "MAIN_SAME_SPEC_SPEC_FILE:" not in store_job
    assert (
        "BENCHMARK_REPO_URL: https://github.com/vLLM-HUST/vllm-hust-benchmark.git"
        in store_job
    )
    assert "BENCHMARK_REPO_REF:" in store_job
    assert "Checkout benchmark repo" in store_job
    assert "git@github.com:vLLM-HUST/vllm-hust-benchmark.git" not in store_job
    assert (
        "vllm-hust-benchmark/${{ vars.VLLM_HUST_SAME_SPEC_SPEC_FILE || "
        "vars.VLLM_HUST_MAIN_SAME_SPEC_SPEC_FILE || "
        "'docs/official-baselines/official-ascend-jan-2026-v0180-random-online-"
        "qwen25-14b-910b2.json' }}"
    ) in store_job


def test_benchmark_repo_default_ref_is_main():
    text = workflow_text()

    assert "feature/perfgate-two-stage" not in text
    assert (
        "BENCHMARK_REPO_REF: ${{ vars.VLLM_HUST_BENCHMARK_REPO_REF || 'main' }}"
    ) in text


def test_pr_checkout_urls_use_https_without_publish_ssh_key():
    text = workflow_text()

    assert "format('https://github.com/{0}.git', github.repository)" in text
    assert "format('git@github.com:{0}.git', github.repository)" in text
    assert (
        "github.event_name == 'pull_request' || github.event_name == 'issue_comment'"
    ) in text
    assert "https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in text
    assert "https://github.com/vLLM-HUST/vllm-ascend-hust.git" in text


def test_main_benchmark_defaults_match_ascend_main_config():
    text = workflow_text()

    assert "default: Qwen/Qwen2.5-14B-Instruct" in text
    assert (
        "github.event_name == 'pull_request' || github.event_name == 'issue_comment'"
    ) in text
    assert "&& '3B' || '14B'" in text
    assert "&& 'BF16' || 'FP16'" in text
    assert 'MAX_MODEL_LEN: ""' in text
    assert "&& '64' || '1024'" in text
    assert "&& '16' || '256'" in text
    assert "perfgate-ascend-qwen25-3b-910b3.json" in text
    assert "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" in text


def test_benchmark_script_does_not_force_max_model_len():
    script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/run_ascend_benchmark_ci.sh"
    ).read_text(encoding="utf-8")

    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}" in script
    assert "max_model_len_args=()" in script
    assert '"${max_model_len_args[@]}"' in script
    assert script.count('"${max_model_len_args[@]}"') == 2


def test_issue_comment_uses_ubuntu_gate_before_self_hosted_runner():
    text = workflow_text()

    assert "issue_comment:" in text
    assert "issue-comment-command:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '1'" in text


def test_issue_comment_path_uses_pr_head_sha_and_base_sha():
    text = workflow_text()

    assert (
        "TARGET_REPO_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_head_sha || github.sha }}" in text
    )
    assert (
        "PR_HEAD_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_head_sha || "
        "github.event.pull_request.head.sha }}" in text
    )
    assert (
        "PR_BASE_SHA: ${{ github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.pr_base_sha || "
        "github.event.pull_request.base.sha }}" in text
    )


def test_benchmark_run_id_and_summary_use_target_repo_sha():
    text = workflow_text()

    assert (
        "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-"
        "${{ env.TARGET_REPO_SHA }}" in text
    )
    assert (
        "target_repo_sha = os.environ.get('TARGET_REPO_SHA') or "
        "os.environ['GITHUB_SHA']" in text
    )
    assert (
        "const targetRepoSha = process.env.TARGET_REPO_SHA || process.env.GITHUB_SHA;"
        in text
    )
    assert (
        "ci-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}-"
        "${targetRepoSha}" in text
    )
    assert "f'- Commit: `{target_repo_sha}`'" in text


def test_issue_comment_path_keeps_publish_secrets_disabled():
    text = workflow_text()

    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf) "
        "&& secrets.HF_TOKEN" in text
    )
    assert (
        "github.event_name != 'issue_comment') && "
        "secrets.VLLM_HUST_BENCHMARK_GH_TOKEN" in text
    )
    assert (
        "github.event_name != 'issue_comment') && "
        "secrets.VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in text
    )


def test_pr_comment_update_has_issues_write_permission():
    text = workflow_text()

    assert "issues: write" in text
    assert "github.rest.issues.createComment" in text
    assert "github.rest.issues.updateComment" in text


def test_issue_comment_denial_feedback_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert (
        "deny_reason: ${{ steps.parse-command.outputs.ASCEND_COMMENT_DENY_REASON }}"
        in text
    )
    assert "issue-comment-denied:" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '0'" in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "runs-on: ubuntu-latest" in text
    assert "github.rest.issues.createComment" in text


def test_issue_comment_help_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert (
        "help_requested: ${{ steps.parse-command.outputs.ASCEND_COMMENT_HELP }}" in text
    )
    assert "issue-comment-help:" in text
    assert "needs.issue-comment-command.outputs.help_requested == '1'" in text
    assert "<!-- ascend-benchmark-command-help -->" in text
    assert "Supported same-repository PR preview commands:" in text
    assert "`/ascend smoke`" in text
    assert "`/ascend scenario random`" in text
    assert "`/ascend group smoke`" in text
    assert (
        "Comment-triggered runs are optional preview checks and are not "
        "required checks." in text
    )
    assert (
        "`/ascend official ...` is reserved for the future formal "
        "leaderboard path and is not supported yet." in text
    )


def test_workflow_dispatch_publish_inputs_are_split():
    text = workflow_text()

    assert "publish_to_benchmark_repo:" in text
    assert "description: Publish benchmark result to HF" in text
    assert (
        "description: Publish benchmark result to the benchmark repo and "
        "refresh leaderboard snapshots" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && "
        "inputs.publish_to_benchmark_repo" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf "
        "&& secrets.HF_TOKEN != ''" in text
    )
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.publish_to_hf)) "
        "&& '1' || '0'" not in text
    )


def test_workflow_dispatch_metadata_lengths_are_parsed_from_single_input():
    text = workflow_text()

    assert "metadata_lengths:" in text
    assert "BENCH_METADATA_LENGTHS:" in text
    assert "inputs.metadata_lengths" in text
    assert "Resolve workflow dispatch metadata lengths" in text
    assert "BENCH_INPUT_LEN=$input_len" in text
    assert "BENCH_OUTPUT_LEN=$output_len" in text
    assert "inputs.input_length" not in text
    assert "inputs.output_length" not in text


def test_l3_benchmark_publish_preflight_runs_before_benchmark():
    text = workflow_text()

    preflight_step = text.index("      - name: L3 benchmark publication preflight")
    checkout_step = text.index("      - name: Checkout target repo with retry")
    benchmark_repo_checkout_step = text.index("      - name: Checkout benchmark repo")
    benchmark_step = text.index(
        "      - name: Runner health preflight (before benchmark)"
    )
    summary_step = text.index("      - name: Build benchmark summary artifacts")

    assert "bash .github/workflows/scripts/l3_benchmark_publish_preflight.sh" in text
    assert checkout_step < preflight_step
    assert preflight_step < benchmark_repo_checkout_step
    assert preflight_step < benchmark_step
    assert "L3_BENCHMARK_PUBLISH_PREFLIGHT" in text[summary_step:]
    assert "L3_BENCHMARK_PUBLISH_TARGET" in text[summary_step:]
    assert "L3_BENCHMARK_PUBLISH_CREDENTIAL" in text[summary_step:]
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION" in text[summary_step:]
    assert "GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT" in text[summary_step:]


def test_ascend_torch_stack_is_installed_before_preinstall_preflight():
    text = workflow_text()

    install_step = text.index("      - name: Install Ascend torch stack for preflight")
    preinstall_preflight_step = text.index(
        "      - name: Runner health preflight (before install)"
    )

    assert install_step < preinstall_preflight_step
    assert '"torch==2.9.0" "torch-npu==2.9.0"' in text
    assert "import torch, torch_npu" in text


def test_l2_targeted_scenario_registry_is_covered_by_parser_tests():
    parser_script = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/scripts/parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")
    parser_tests = (
        Path(__file__).resolve().parent / "test_parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")
    registry_tests = (
        Path(__file__).resolve().parent / "test_ascend_targeted_scenarios.py"
    ).read_text(encoding="utf-8")

    assert "load_targeted_scenario_registry" in parser_script
    assert "test_parse_group_command_maps_to_supported_group" in parser_tests
    assert "test_load_targeted_scenario_registry_from_repo_file" in registry_tests


def test_issue_comment_non_pr_and_fork_pr_are_not_allowed_by_parser_tests():
    parser_tests = (
        Path(__file__).resolve().parent / "test_parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")

    assert "test_resolve_issue_comment_pr_context_rejects_non_pr_issue" in parser_tests
    assert "test_resolve_issue_comment_pr_context_rejects_fork_pr" in parser_tests
