from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend-benchmark-leaderboard.yml"
)


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_pr_comment_update_job_has_job_level_issues_write_permission():
    text = workflow_text()

    comment_step = text.index("      - name: Update PR benchmark comment")
    job_permissions = text.rindex("    permissions:", 0, comment_step)
    permissions_block = text[job_permissions:text.index("    runs-on:", job_permissions)]

    assert "issues: write" in permissions_block
    assert "pull-requests: write" in permissions_block


def test_issue_comment_non_pr_commands_receive_denial_feedback():
    text = workflow_text()

    assert "if: ${{ github.event.comment.body != '' }}" in text
    assert "--event-payload \"$GITHUB_EVENT_PATH\"" in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "persist-credentials: false" in text
    assert "const safeReason = reason.replace" in text
    assert "`Reason: ${safeReason}`" in text


def test_fork_pr_security_note_is_blocking():
    text = workflow_text()

    assert "Skipping Ascend benchmark on fork PRs" in text
    assert "exit 1" in text[text.index("fork-pr-security-note:"):text.index("  ascend-benchmark:")]


def test_main_baseline_store_has_spec_file_and_benchmark_repo_checkout():
    text = workflow_text()
    store_job = text[text.index("  store-main-perfgate-baseline:"):]

    assert "TARGET_REPO_SHA: ${{ github.sha }}" in store_job
    assert "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-${{ env.TARGET_REPO_SHA }}" in store_job
    assert "RESULT_ROOT: ${{ github.workspace }}/.benchmarks/ci/ci-${{ github.run_id }}-${{ github.run_attempt }}-${{ env.TARGET_REPO_SHA }}" in store_job
    assert "PERFGATE_SPEC_FILE:" in store_job
    assert "BENCHMARK_REPO_URL: https://github.com/vLLM-HUST/vllm-hust-benchmark.git" in store_job
    assert "BENCHMARK_REPO_REF:" in store_job
    assert "Checkout benchmark repo" in store_job
    assert "git@github.com:vLLM-HUST/vllm-hust-benchmark.git" not in store_job
    assert "vllm-hust-benchmark/${{ env.PERFGATE_SPEC_FILE }}" in store_job


def test_issue_comment_uses_ubuntu_gate_before_self_hosted_runner():
    text = workflow_text()

    assert "issue_comment:" in text
    assert "issue-comment-command:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '1'" in text


def test_issue_comment_path_uses_pr_head_sha_and_base_sha():
    text = workflow_text()

    assert "TARGET_REPO_SHA: ${{ github.event_name == 'issue_comment' && needs.issue-comment-command.outputs.pr_head_sha || github.sha }}" in text
    assert "PR_HEAD_SHA: ${{ github.event_name == 'issue_comment' && needs.issue-comment-command.outputs.pr_head_sha || github.event.pull_request.head.sha }}" in text
    assert "PR_BASE_SHA: ${{ github.event_name == 'issue_comment' && needs.issue-comment-command.outputs.pr_base_sha || github.event.pull_request.base.sha }}" in text


def test_benchmark_run_id_and_summary_use_target_repo_sha():
    text = workflow_text()

    assert "RUN_ID: ci-${{ github.run_id }}-${{ github.run_attempt }}-${{ env.TARGET_REPO_SHA }}" in text
    assert "target_repo_sha = os.environ.get('TARGET_REPO_SHA') or os.environ['GITHUB_SHA']" in text
    assert "const targetRepoSha = process.env.TARGET_REPO_SHA || process.env.GITHUB_SHA;" in text
    assert "ci-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}-${targetRepoSha}" in text
    assert "f'- Commit: `{target_repo_sha}`'" in text


def test_issue_comment_path_keeps_publish_secrets_disabled():
    text = workflow_text()

    assert "github.event_name != 'issue_comment') && secrets.HF_TOKEN" in text
    assert "github.event_name != 'issue_comment') && secrets.VLLM_HUST_BENCHMARK_GH_TOKEN" in text
    assert "github.event_name != 'issue_comment') && secrets.VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in text


def test_pr_comment_update_has_issues_write_permission():
    text = workflow_text()

    assert "issues: write" in text
    assert "github.rest.issues.createComment" in text
    assert "github.rest.issues.updateComment" in text


def test_issue_comment_denial_feedback_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert "deny_reason: ${{ steps.parse-command.outputs.ASCEND_COMMENT_DENY_REASON }}" in text
    assert "issue-comment-denied:" in text
    assert "needs: [issue-comment-command]" in text
    assert "needs.issue-comment-command.outputs.should_run == '0'" in text
    assert "needs.issue-comment-command.outputs.deny_reason != ''" in text
    assert "runs-on: ubuntu-latest" in text
    assert "github.rest.issues.createComment" in text


def test_issue_comment_help_is_posted_without_self_hosted_runner():
    text = workflow_text()

    assert "help_requested: ${{ steps.parse-command.outputs.ASCEND_COMMENT_HELP }}" in text
    assert "issue-comment-help:" in text
    assert "needs.issue-comment-command.outputs.help_requested == '1'" in text
    assert "<!-- ascend-benchmark-command-help -->" in text
    assert "Supported same-repository PR preview commands:" in text
    assert "Comment-triggered runs are optional preview checks and are not required checks." in text
    assert "`/ascend smoke`, `/ascend group ...`, `/ascend scenario ...`, and `/ascend official ...` are planned but not supported yet." in text


def test_issue_comment_non_pr_and_fork_pr_are_not_allowed_by_parser_tests():
    parser_tests = (
        Path(__file__).resolve().parent / "test_parse_ascend_comment_command.py"
    ).read_text(encoding="utf-8")

    assert "test_resolve_issue_comment_pr_context_rejects_non_pr_issue" in parser_tests
    assert "test_resolve_issue_comment_pr_context_rejects_fork_pr" in parser_tests
