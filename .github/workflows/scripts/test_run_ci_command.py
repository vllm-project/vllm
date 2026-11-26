# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import subprocess
import unittest
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

from run_ci_command import (
    CANCELABLE_BUILD_STATES,
    CI_AUTHORIZED_COMMENT_MARKER,
    COMMAND_CANCEL_AMD_CI,
    COMMAND_CANCEL_CI,
    COMMAND_RETRY_AMD_FAILED,
    COMMAND_RETRY_FAILED,
    COMMAND_RUN_AMD_CI,
    COMMAND_RUN_AMD_CI_ALL,
    COMMAND_RUN_AMD_CI_NIGHTLY,
    COMMAND_RUN_CI,
    COMMAND_RUN_CI_ALL,
    COMMAND_RUN_CI_NIGHTLY,
    RETRY_STATES,
    ROCM_BUILD_SH,
    ROCM_CACHE_CHECK_KEY,
    ROCM_CACHE_STEP_KEY,
    ROCM_CACHE_YAML,
    ROCM_REFRESH_STEP_KEY,
    ROCM_STACK_INPUTS,
    RUN_CI_COMMAND_ENV,
    ApiError,
    BuildkiteClient,
    GitHubClient,
    HttpTransport,
    authorize,
    codeowners_for,
    create_build_payload,
    defines_step_key,
    handle_buildkite_status,
    has_trusted_approval,
    is_active_build,
    is_build_for_pr,
    is_ci_command_attempt,
    iter_builds,
    notify_authorized,
    parse_command,
    parse_trusted_users,
    pipeline_for_command,
    recheck_rocm_stack_prs,
    reconcile_rocm_cache,
    reply_unrecognized_command,
    resolve_workflow_run_pr,
    rocm_key_format,
    run,
    select_latest_build,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CODEOWNERS_FILE = REPO_ROOT / ".github" / "CODEOWNERS"


def make_pr(**overrides: Any) -> dict[str, Any]:
    pr = {
        "base": {"ref": "main"},
        "created_at": "2026-01-01T00:00:00Z",
        "draft": False,
        "head": {
            "label": "contributor:feature",
            "ref": "feature",
            "repo": {"clone_url": "https://github.com/contributor/vllm.git"},
            "sha": "0123456789abcdef",
        },
        "labels": [],
        "number": 42,
        "state": "open",
        "user": {"login": "author"},
    }
    pr.update(overrides)
    return pr


def make_event(command: str, actor: str = "reviewer") -> dict[str, Any]:
    return {
        "comment": {
            "body": command,
            "id": 99,
            "user": {"login": actor},
        },
        "issue": {
            "number": 42,
            "pull_request": {},
        },
    }


class FakeGitHub:
    def __init__(
        self,
        *,
        permission: str = "write",
        permissions: dict[str, str] | None = None,
        pr: dict[str, Any] | None = None,
        review_decision: str = "REVIEW_REQUIRED",
        reviews: list[dict[str, Any]] | None = None,
        comments: list[str] | None = None,
        pulls_for_commit: list[dict[str, Any]] | None = None,
        prs: dict[int, dict[str, Any]] | None = None,
        behind: int = 0,
        files: list[str] | None = None,
    ) -> None:
        self.comments = comments or []
        self.files = files or []
        # (path, ref) -> blob; every ref shares every blob by default.
        self.file_shas: dict[tuple[str, str], str] = {}
        self.permission = permission
        self.permissions = permissions or {}
        self.pr = pr or make_pr()
        self.reactions: list[str] = []
        self.review_decision = review_decision
        self.reviews = reviews or []
        self.pulls_for_commit = pulls_for_commit or []
        self.prs = prs or {self.pr["number"]: self.pr}
        self.behind = behind
        self.lag_queries: list[tuple[str, str]] = []

    def get_commits_behind_base(self, base_ref: str, head_sha: str) -> int:
        self.lag_queries.append((base_ref, head_sha))
        return self.behind

    def get_pr(self, number: int) -> dict[str, Any]:
        try:
            return self.prs[number]
        except KeyError as error:
            raise ApiError(404, "Not found") from error

    def get_permission(self, actor: str) -> str:
        return self.permissions.get(actor, self.permission)

    def list_pulls_for_commit(self, commit: str) -> list[dict[str, Any]]:
        return self.pulls_for_commit

    def get_review_decision(self, number: int) -> str:
        return self.review_decision

    def list_reviews(self, number: int) -> list[dict[str, Any]]:
        return self.reviews

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        return [
            {
                "body": body,
                "user": {"login": "github-actions[bot]"},
            }
            for body in self.comments
        ]

    def list_reactions(self, comment_id: int) -> list[dict[str, Any]]:
        return []

    def add_reaction(self, comment_id: int, content: str) -> None:
        self.reactions.append(content)

    def add_comment(self, issue_number: int, body: str) -> None:
        self.comments.append(body)

    def list_pr_files(self, number: int) -> list[str]:
        return self.files

    def get_file_sha(self, path: str, ref: str) -> str:
        return self.file_shas.get((path, ref), "blob")


class FakeBranchUpdater:
    def __init__(self, github: FakeGitHub, error: ApiError | None = None) -> None:
        self.calls: list[tuple[int, str]] = []
        self.error = error
        self.github = github

    def update_branch(self, number: int, expected_head_sha: str) -> None:
        self.calls.append((number, expected_head_sha))
        if self.error:
            raise self.error
        self.github.prs[number] = {
            **self.github.prs[number],
            "head": {**self.github.prs[number]["head"], "sha": "merge-commit-sha"},
        }
        self.github.behind = 0


class FakeBuildkite:
    def __init__(
        self,
        build_lists: list[list[dict[str, Any]]] | None = None,
        failed_job_lists: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.build_lists = build_lists or []
        self.created_builds: list[dict[str, Any]] = []
        self.failed_job_lists = failed_job_lists or []
        self.job_list_calls: list[int] = []
        self.list_calls: list[tuple[str | None, tuple[str, str] | None]] = []
        self.list_requests: list[dict[str, Any]] = []
        self.retry_calls: list[tuple[int, str]] = []
        self.cancel_calls: list[int] = []

    def list_builds(
        self,
        commit: str | None,
        *,
        branch: str | None = None,
        metadata: tuple[str, str] | None = None,
        states: tuple[str, ...] = (),
        page: int = 1,
        created_from: str | None = None,
    ) -> list[dict[str, Any]]:
        self.list_calls.append((commit, metadata))
        self.list_requests.append(
            {
                "branch": branch,
                "commit": commit,
                "created_from": created_from,
                "metadata": metadata,
                "page": page,
                "states": states,
            }
        )
        return self.build_lists.pop(0)

    def create_build(self, body: dict[str, Any]) -> dict[str, Any]:
        self.created_builds.append(body)
        return {
            "number": 123,
            "web_url": "https://buildkite.example/builds/123",
        }

    def retry_failed_jobs(
        self,
        build_number: int,
        states: str,
    ) -> dict[str, Any]:
        self.retry_calls.append((build_number, states))
        return {"retried_jobs_count": 3}

    def cancel_build(self, build_number: int) -> dict[str, Any]:
        self.cancel_calls.append(build_number)
        return {"number": build_number, "state": "canceling"}

    def list_failed_jobs(self, build_number: int) -> list[dict[str, Any]]:
        self.job_list_calls.append(build_number)
        return self.failed_job_lists.pop(0)


class FakeTransport:
    def __init__(self, response: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

    def request(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.response, tuple):
            return self.response[len(self.calls) - 1]
        return self.response


class FakeHttpResponse:
    def __init__(self, response: Any) -> None:
        self.response = response

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.response).encode()


class RunCiCommandTest(unittest.TestCase):
    @patch("run_ci_command.urllib.request.urlopen")
    def test_http_transport_retries_buildkite_rate_limit(self, urlopen: Any) -> None:
        body = {
            "message": "Please wait 9 seconds before making more requests.",
            "reset": 9,
            "scope": "rest",
        }
        rate_limit_error = urllib.error.HTTPError(
            "https://api.buildkite.com/v2/builds",
            429,
            "Too Many Requests",
            {
                "RateLimit-Limit": "400",
                "RateLimit-Remaining": "0",
                "RateLimit-Reset": "9",
            },
            io.BytesIO(json.dumps(body).encode()),
        )
        urlopen.side_effect = [rate_limit_error, FakeHttpResponse({"ok": True})]
        delays: list[float] = []

        response = HttpTransport(
            jitter=lambda: 2.5,
            sleep=delays.append,
        ).request("https://api.buildkite.com/v2/builds")

        self.assertEqual(response, {"ok": True})
        self.assertEqual(delays, [11.5])
        self.assertEqual(urlopen.call_count, 2)

    @patch("run_ci_command.urllib.request.urlopen")
    def test_http_transport_retries_rate_limit_three_times(self, urlopen: Any) -> None:
        def rate_limit_error() -> urllib.error.HTTPError:
            body = {
                "message": "Please wait 9 seconds before making more requests.",
                "reset": 9,
                "scope": "rest",
            }
            return urllib.error.HTTPError(
                "https://api.buildkite.com/v2/builds",
                429,
                "Too Many Requests",
                {
                    "RateLimit-Limit": "400",
                    "RateLimit-Remaining": "0",
                    "RateLimit-Reset": "9",
                },
                io.BytesIO(json.dumps(body).encode()),
            )

        urlopen.side_effect = [rate_limit_error() for _ in range(4)]
        delays: list[float] = []

        with self.assertRaisesRegex(RuntimeError, "API returned 429"):
            HttpTransport(
                jitter=lambda: 2,
                sleep=delays.append,
            ).request("https://api.buildkite.com/v2/builds")

        self.assertEqual(delays, [11, 11, 11])
        self.assertEqual(urlopen.call_count, 4)

    @patch("run_ci_command.urllib.request.urlopen")
    def test_http_transport_does_not_retry_permission_error(self, urlopen: Any) -> None:
        permission_error = urllib.error.HTTPError(
            "https://api.github.com/repos/vllm-project/vllm/issues/1/comments",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"message":"Resource not accessible by integration"}'),
        )
        urlopen.side_effect = permission_error
        delays: list[float] = []

        with self.assertRaisesRegex(
            RuntimeError,
            "Resource not accessible by integration",
        ):
            HttpTransport(
                jitter=lambda: 2,
                sleep=delays.append,
            ).request(
                "https://api.github.com/repos/vllm-project/vllm/issues/1/comments"
            )

        self.assertEqual(delays, [])
        self.assertEqual(urlopen.call_count, 1)

    def test_only_exact_ci_commands_are_accepted(self) -> None:
        commands = (
            COMMAND_RUN_CI,
            COMMAND_RUN_CI_ALL,
            COMMAND_RUN_CI_NIGHTLY,
            COMMAND_RETRY_FAILED,
            COMMAND_CANCEL_CI,
            COMMAND_RUN_AMD_CI,
            COMMAND_RUN_AMD_CI_ALL,
            COMMAND_RUN_AMD_CI_NIGHTLY,
            COMMAND_RETRY_AMD_FAILED,
            COMMAND_CANCEL_AMD_CI,
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(parse_command(command), command)
                override = f"{command} --allow-stale"
                if " run" in command:
                    self.assertEqual(parse_command(override), override)
                else:
                    self.assertIsNone(parse_command(override))

        self.assertIsNone(parse_command("/ci run please"))
        self.assertIsNone(parse_command("/ci run all please"))
        self.assertIsNone(parse_command("/ci cancel please"))
        self.assertIsNone(parse_command("/amd-ci run please"))
        self.assertIsNone(parse_command("/AMD-CI run"))
        self.assertIsNone(parse_command("/amdci run"))
        self.assertIsNone(parse_command("/ci run --allow-stale all"))
        self.assertIsNone(parse_command("/amd-ci run --allow-stale please"))

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertEqual(parse_command("/ci run\r\n"), COMMAND_RUN_CI)
        self.assertEqual(parse_command(" /ci run "), COMMAND_RUN_CI)
        self.assertEqual(parse_command("/amd-ci retry\n"), COMMAND_RETRY_AMD_FAILED)

    def test_ci_command_attempts_are_detected(self) -> None:
        for body in ("/ci", "/ci rn", "/CI run", "/amd-ci run please", " /ci\nrun"):
            with self.subTest(body=body):
                self.assertTrue(is_ci_command_attempt(body))
        for body in ("", "LGTM", "/cifoo", "/amdci run", "please /ci run"):
            with self.subTest(body=body):
                self.assertFalse(is_ci_command_attempt(body))

    def test_unrecognized_command_gets_feedback(self) -> None:
        github = FakeGitHub()

        reply_unrecognized_command(make_event("/ci rn", actor="author"), github)

        self.assertEqual(github.reactions, ["-1"])
        self.assertEqual(len(github.comments), 1)
        self.assertIn(
            "@author, that is not a recognized CI command", github.comments[0]
        )
        self.assertIn(f"`{COMMAND_RUN_CI}`", github.comments[0])
        self.assertIn("<!-- vllm-ci-command:99 -->", github.comments[0])

    def test_unrecognized_command_feedback_is_not_repeated(self) -> None:
        github = FakeGitHub(comments=["❌ earlier\n\n<!-- vllm-ci-command:99 -->"])

        reply_unrecognized_command(make_event("/ci rn"), github)

        self.assertEqual(len(github.comments), 1)
        self.assertEqual(github.reactions, [])

    def test_commands_select_only_their_configured_pipeline(self) -> None:
        cases = (
            (COMMAND_RUN_CI, "ci"),
            (COMMAND_RUN_CI_ALL, "ci"),
            (COMMAND_RUN_CI_NIGHTLY, "ci"),
            (COMMAND_RETRY_FAILED, "ci"),
            (COMMAND_CANCEL_CI, "ci"),
            (COMMAND_RUN_AMD_CI, "amd-ci"),
            (COMMAND_RUN_AMD_CI_ALL, "amd-ci"),
            (COMMAND_RUN_AMD_CI_NIGHTLY, "amd-ci"),
            (COMMAND_RETRY_AMD_FAILED, "amd-ci"),
            (COMMAND_CANCEL_AMD_CI, "amd-ci"),
        )
        for command, expected_pipeline in cases:
            with self.subTest(command=command):
                self.assertEqual(
                    pipeline_for_command(command),
                    expected_pipeline,
                )
        with self.assertRaisesRegex(ValueError, "Unsupported CI command"):
            pipeline_for_command("/amd-ci run arbitrary-pipeline")

    def test_write_access_authorizes_reviewers_and_authors(self) -> None:
        allowed, _ = authorize(
            actor="reviewer",
            permission="write",
            pr=make_pr(),
        )
        self.assertTrue(allowed)

    def test_configured_trusted_contributors_can_run_ci(self) -> None:
        trusted_users = parse_trusted_users("trusted-one, TRUSTED-TWO")
        allowed, _ = authorize(
            actor="trusted-two",
            permission="read",
            pr=make_pr(),
            trusted_users=trusted_users,
        )
        self.assertTrue(allowed)

    def test_authors_need_an_approval_or_ready_label(self) -> None:
        pending, _ = authorize(
            actor="author",
            permission="read",
            pr=make_pr(),
        )
        approved, _ = authorize(
            actor="author",
            permission="read",
            pr=make_pr(),
            trusted_approval=True,
        )
        ready, _ = authorize(
            actor="author",
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        self.assertFalse(pending)
        self.assertTrue(approved)
        self.assertTrue(ready)

    def test_non_author_contributors_without_write_are_denied(self) -> None:
        allowed, _ = authorize(
            actor="contributor",
            permission="read",
            pr=make_pr(),
            trusted_approval=True,
        )
        self.assertFalse(allowed)

    def test_authors_cannot_use_ready_state_on_draft_prs(self) -> None:
        allowed, _ = authorize(
            actor="author",
            permission="read",
            pr=make_pr(draft=True, labels=[{"name": "ready"}]),
            trusted_approval=True,
        )
        self.assertFalse(allowed)

    def test_only_trusted_reviewers_can_delegate_through_approval(self) -> None:
        approved_review = {
            "state": "APPROVED",
            "user": {"login": "reviewer"},
        }
        trusted = FakeGitHub(
            permission="read",
            permissions={"reviewer": "write"},
            review_decision="APPROVED",
            reviews=[approved_review],
        )
        untrusted = FakeGitHub(
            permission="read",
            review_decision="APPROVED",
            reviews=[approved_review],
        )
        self.assertTrue(has_trusted_approval(trusted, 42, set()))
        self.assertFalse(has_trusted_approval(untrusted, 42, set()))

    def test_build_matching_is_scoped_to_the_pr(self) -> None:
        self.assertTrue(is_build_for_pr({"pull_request": {"id": 42}}, 42))
        self.assertFalse(is_build_for_pr({"pull_request": {"id": 43}}, 42))
        self.assertTrue(
            is_build_for_pr(
                {"meta_data": {"github-pr-number": "42"}},
                42,
            )
        )

    def test_latest_build_selection_ignores_other_prs(self) -> None:
        latest = select_latest_build(
            [
                {
                    "created_at": "2026-07-28T02:00:00Z",
                    "number": 3,
                    "pull_request": {"id": 43},
                },
                {
                    "created_at": "2026-07-28T01:00:00Z",
                    "number": 2,
                    "pull_request": {"id": 42},
                },
                {
                    "created_at": "2026-07-28T00:00:00Z",
                    "number": 1,
                    "pull_request": {"id": 42},
                },
            ],
            42,
        )
        self.assertEqual(latest["number"], 2)

    def test_active_build_states_prevent_duplicate_runs(self) -> None:
        self.assertTrue(is_active_build({"state": "scheduled"}))
        self.assertTrue(is_active_build({"state": "running"}))
        self.assertTrue(is_active_build({"state": "waiting"}))
        self.assertTrue(is_active_build({"blocked": True, "state": "passed"}))
        self.assertTrue(is_active_build({"blocked": True, "state": "blocked"}))
        # Every job finished; only an optional picker step is still blocked.
        finished = "2026-11-23T23:04:32Z"
        self.assertFalse(
            is_active_build(
                {"blocked": True, "finished_at": finished, "state": "blocked"}
            )
        )
        self.assertFalse(
            is_active_build(
                {"blocked": True, "finished_at": finished, "state": "passed"}
            )
        )
        self.assertTrue(
            is_active_build(
                {"blocked": True, "finished_at": finished, "state": "failing"}
            )
        )
        self.assertFalse(is_active_build({"state": "failed"}))

    def test_build_payload_preserves_pr_context(self) -> None:
        payload = create_build_payload(
            actor="reviewer",
            comment_id=99,
            pr=make_pr(labels=[{"name": "ready"}, {"name": "v1"}]),
        )
        self.assertEqual(
            payload,
            {
                "commit": "0123456789abcdef",
                "branch": "contributor:feature",
                "message": "PR #42 /ci run by @reviewer",
                "pull_request_id": 42,
                "pull_request_base_branch": "main",
                "pull_request_repository": ("https://github.com/contributor/vllm.git"),
                "pull_request_labels": ["ready", "v1"],
                "env": {
                    "VLLM_CI_GITHUB_COMMENT_ID": "99",
                    "VLLM_CI_TRIGGERED_BY": "reviewer",
                },
                "meta_data": {
                    "github-comment-id": "99",
                    "github-pr-number": "42",
                    "github-triggered-by": "reviewer",
                },
            },
        )

    def test_build_payload_uses_head_label_to_avoid_branch_collision(self) -> None:
        pr = make_pr()
        pr["head"]["ref"] = "main"
        pr["head"]["label"] = "contributor:main"

        payload = create_build_payload(actor="reviewer", comment_id=99, pr=pr)

        self.assertEqual(payload["branch"], "contributor:main")

    def test_build_payload_falls_back_to_ref_without_label(self) -> None:
        pr = make_pr()
        del pr["head"]["label"]

        payload = create_build_payload(actor="reviewer", comment_id=99, pr=pr)

        self.assertEqual(payload["branch"], "feature")

    def test_write_reviewer_runs_ci_without_delegation(self) -> None:
        github = FakeGitHub()
        buildkite = FakeBuildkite([[], []])
        run(make_event(COMMAND_RUN_CI), github, buildkite)

        self.assertEqual(len(buildkite.created_builds), 1)
        self.assertEqual(
            buildkite.created_builds[0]["message"],
            "PR #42 /ci run by @reviewer",
        )
        self.assertEqual(github.reactions, ["eyes", "rocket"])
        self.assertTrue(github.comments[0].startswith("✅ "))
        self.assertIn("Buildkite CI #123", github.comments[0])

    def test_run_variants_require_current_base_without_changing_commit(self) -> None:
        for command in RUN_CI_COMMAND_ENV:
            if command.endswith(" --allow-stale"):
                continue
            for behind in (0, 1, 5):
                with self.subTest(command=command, behind=behind):
                    github = FakeGitHub(
                        behind=behind, pr=make_pr(base={"ref": "release/1.0"})
                    )
                    buildkite = FakeBuildkite([[], []])
                    run(make_event(command), github, buildkite)

                    self.assertEqual(
                        github.lag_queries,
                        [("release/1.0", github.pr["head"]["sha"])],
                    )
                    if behind == 0:
                        self.assertEqual(
                            buildkite.created_builds[0]["commit"],
                            github.pr["head"]["sha"],
                        )
                        self.assertIn(
                            "](https://buildkite.example/builds/123)",
                            github.comments[0],
                        )
                    else:
                        self.assertEqual(buildkite.created_builds, [])
                        self.assertTrue(github.comments[0].startswith("❌ "))
                        self.assertIn(str(behind), github.comments[0])
                        self.assertIn("upstream `release/1.0`", github.comments[0])
                        self.assertIn("every commit", github.comments[0])
                        self.assertIn(
                            "No new CI build was started.", github.comments[0]
                        )
                        self.assertIn(command, github.comments[0])

    def test_run_merges_base_into_outdated_branch_before_building(self) -> None:
        github = FakeGitHub(behind=5)
        updater = FakeBranchUpdater(github)
        buildkite = FakeBuildkite([[], []])

        with patch("run_ci_command.time.sleep"):
            run(make_event(COMMAND_RUN_CI), github, buildkite, "", updater)

        self.assertEqual(updater.calls, [(42, "0123456789abcdef")])
        self.assertEqual(buildkite.created_builds[0]["commit"], "merge-commit-sha")
        self.assertTrue(github.comments[0].startswith("✅ "))
        self.assertIn("5 commits behind", github.comments[0])
        self.assertIn("Merged upstream `main`", github.comments[0])

    def test_only_committers_can_rerun_a_commit_that_already_ran_ci(self) -> None:
        finished_build = {
            "created_at": "2026-07-28T01:00:00Z",
            "pull_request": {"id": 42},
            "state": "failed",
            "web_url": "https://buildkite.example/builds/122",
        }
        for actor, permission, command, allowed in (
            ("author", "read", COMMAND_RUN_CI, False),
            ("author", "read", f"{COMMAND_RUN_CI} --allow-stale", False),
            ("reviewer", "write", COMMAND_RUN_CI, True),
        ):
            with self.subTest(actor=actor, command=command):
                github = FakeGitHub(
                    behind=5,
                    permission=permission,
                    pr=make_pr(labels=[{"name": "ready"}]),
                )
                updater = FakeBranchUpdater(github)
                buildkite = FakeBuildkite([[], [finished_build]])

                with patch("run_ci_command.time.sleep"):
                    run(make_event(command, actor), github, buildkite, "", updater)

                self.assertEqual(bool(buildkite.created_builds), allowed)
                if not allowed:
                    self.assertEqual(updater.calls, [])
                    self.assertIn("CI already ran on this commit", github.comments[0])
                    self.assertIn("`/ci retry`", github.comments[0])

    def test_run_waits_for_fork_merge_commit_to_reach_base_repo(self) -> None:
        github = FakeGitHub(behind=5)
        updater = FakeBranchUpdater(github)
        buildkite = FakeBuildkite([[], []])

        with (
            patch("run_ci_command.time.sleep"),
            patch.object(
                github,
                "get_commits_behind_base",
                side_effect=[5, ApiError(404, "Not Found"), 0],
            ),
        ):
            run(make_event(COMMAND_RUN_CI), github, buildkite, "", updater)

        self.assertEqual(buildkite.created_builds[0]["commit"], "merge-commit-sha")
        self.assertTrue(github.comments[0].startswith("✅ "))

    def test_run_reports_failed_merge_without_building(self) -> None:
        github = FakeGitHub(behind=5)
        updater = FakeBranchUpdater(github, error=ApiError(422, "merge conflict"))
        buildkite = FakeBuildkite([[], []])

        run(make_event(COMMAND_RUN_CI), github, buildkite, "", updater)

        self.assertEqual(buildkite.created_builds, [])
        self.assertTrue(github.comments[0].startswith("❌ "))
        self.assertIn("could not merge upstream `main`", github.comments[0])
        self.assertIn("merge conflict", github.comments[0])
        self.assertIn("No new CI build was started.", github.comments[0])
        self.assertIn("`/ci run --allow-stale`", github.comments[0])

    def test_allow_stale_preserves_author_permissions_and_run_configuration(
        self,
    ) -> None:
        cases = (
            (COMMAND_RUN_CI, "ci", {}),
            (COMMAND_RUN_CI_ALL, "ci", {"RUN_ALL": "1"}),
            (COMMAND_RUN_CI_NIGHTLY, "ci", {"RUN_ALL": "1", "NIGHTLY": "1"}),
            (COMMAND_RUN_AMD_CI, "amd-ci", {}),
            (COMMAND_RUN_AMD_CI_ALL, "amd-ci", {"RUN_ALL": "1"}),
            (
                COMMAND_RUN_AMD_CI_NIGHTLY,
                "amd-ci",
                {"RUN_ALL": "1", "NIGHTLY": "1"},
            ),
        )
        for original_command, pipeline, expected_env in cases:
            command = f"{original_command} --allow-stale"
            for ready, behind in ((False, 100), (True, 0), (True, 100)):
                with self.subTest(command=command, ready=ready, behind=behind):
                    github = FakeGitHub(
                        permission="read",
                        behind=behind,
                        pr=make_pr(
                            base={"ref": "feature/stack"},
                            labels=[{"name": "ready"}] if ready else [],
                        ),
                    )
                    buildkite = FakeBuildkite([[], []])

                    run(make_event(command, "author"), github, buildkite)

                    self.assertEqual(pipeline_for_command(command), pipeline)
                    if not ready:
                        self.assertEqual(buildkite.created_builds, [])
                        self.assertEqual(buildkite.list_calls, [])
                        self.assertEqual(github.lag_queries, [])
                        self.assertTrue(github.comments[0].startswith("❌ "))
                        continue
                    self.assertEqual(len(buildkite.created_builds), 1)
                    payload = buildkite.created_builds[0]
                    self.assertEqual(payload["commit"], github.pr["head"]["sha"])
                    self.assertEqual(
                        payload["pull_request_base_branch"], "feature/stack"
                    )
                    self.assertEqual(payload["message"], f"PR #42 {command} by @author")
                    self.assertEqual(
                        payload["env"],
                        {
                            "VLLM_CI_GITHUB_COMMENT_ID": "99",
                            "VLLM_CI_TRIGGERED_BY": "author",
                            **expected_env,
                        },
                    )
                    self.assertEqual(
                        github.lag_queries,
                        [("feature/stack", github.pr["head"]["sha"])],
                    )
                    self.assertIn(
                        "Buildkite AMD CI #123"
                        if pipeline == "amd-ci"
                        else "Buildkite CI #123",
                        github.comments[0],
                    )
                    if behind:
                        self.assertIn("100 commits behind", github.comments[0])
                        self.assertIn("upstream `feature/stack`", github.comments[0])
                        self.assertIn("Before merging", github.comments[0])
                        self.assertIn("latest PR commit", github.comments[0])
                        self.assertIn(original_command, github.comments[0])

    def test_base_lag_query_qualifies_branch_refs_and_validates_count(self) -> None:
        for base_ref, encoded_ref in (
            ("main", "main"),
            ("release/1.0", "release%2F1.0"),
            ("feature/stack", "feature%2Fstack"),
        ):
            for count in (0, 5, None, -1, True, "5"):
                with self.subTest(base_ref=base_ref, count=count):
                    transport = FakeTransport(
                        {} if count is None else {"behind_by": count}
                    )
                    github = GitHubClient("token", "vllm-project/vllm", transport)
                    if type(count) is int and count >= 0:
                        self.assertEqual(
                            github.get_commits_behind_base(base_ref, "head-sha"), count
                        )
                    else:
                        with self.assertRaises(ApiError):
                            github.get_commits_behind_base(base_ref, "head-sha")
                    self.assertEqual(
                        transport.calls[0]["url"],
                        "https://api.github.com/repos/vllm-project/vllm/compare/"
                        f"refs%2Fheads%2F{encoded_ref}...head-sha?per_page=1",
                    )
                    self.assertEqual(transport.calls[0]["method"], "GET")

    def test_unknown_lag_or_changed_pr_prevents_runs(self) -> None:
        for command in (
            COMMAND_RUN_CI,
            COMMAND_RUN_AMD_CI,
            f"{COMMAND_RUN_CI} --allow-stale",
            f"{COMMAND_RUN_AMD_CI} --allow-stale",
        ):
            for fault in ("api", "head", "closed", "base"):
                with self.subTest(command=command, fault=fault):
                    github = FakeGitHub()
                    buildkite = FakeBuildkite([[], []])

                    def check(
                        base_ref: str,
                        sha: str,
                        fault: str = fault,
                        github: FakeGitHub = github,
                    ) -> int:
                        if fault == "api":
                            raise ApiError(503, "Unavailable")
                        changed = make_pr()
                        if fault == "head":
                            changed["head"]["sha"] = "another-head"
                        elif fault == "closed":
                            changed["state"] = "closed"
                        else:
                            changed["base"]["ref"] = "another-base"
                        github.prs[42] = changed
                        return 0

                    with patch.object(
                        github, "get_commits_behind_base", side_effect=check
                    ):
                        run(make_event(command), github, buildkite)
                    self.assertEqual(buildkite.created_builds, [])
                    self.assertEqual(buildkite.retry_calls, [])
                    self.assertTrue(github.comments[0].startswith("❌ "))
                    self.assertNotIn("Triggered", github.comments[0])

    def test_amd_run_ignores_blocked_builds(self) -> None:
        for metadata in ({}, {"github-comment-id": "98"}):
            with self.subTest(metadata=metadata):
                github = FakeGitHub()
                blocked_build = {
                    "blocked": True,
                    "created_at": "2026-08-18T01:00:00Z",
                    "meta_data": metadata,
                    "number": 122,
                    "pull_request": {"id": 42},
                    "state": "blocked",
                    "web_url": "https://buildkite.example/amd-ci/builds/122",
                }
                buildkite = FakeBuildkite([[], [blocked_build]])

                run(make_event(COMMAND_RUN_AMD_CI_ALL), github, buildkite)

                self.assertEqual(len(buildkite.created_builds), 1)
                self.assertEqual(buildkite.created_builds[0]["env"]["RUN_ALL"], "1")

    def test_amd_run_deduplicates_comment_triggered_active_build(self) -> None:
        github = FakeGitHub()
        command_build = {
            "blocked": False,
            "created_at": "2026-08-18T01:00:00Z",
            "meta_data": {"github-comment-id": "98"},
            "number": 122,
            "pull_request": {"id": 42},
            "source": "api",
            "state": "running",
            "web_url": "https://buildkite.example/amd-ci/builds/122",
        }
        buildkite = FakeBuildkite([[], [command_build]])

        run(make_event(COMMAND_RUN_AMD_CI_ALL), github, buildkite)

        self.assertEqual(buildkite.created_builds, [])
        self.assertIn("AMD CI is already running", github.comments[0])

    def test_amd_run_ignores_finished_webhook_build_for_author(self) -> None:
        webhook_build = {
            "created_at": "2026-08-18T01:00:00Z",
            "meta_data": {},
            "number": 122,
            "pull_request": {"id": 42},
            "source": "webhook",
            "state": "passed",
            "web_url": "https://buildkite.example/amd-ci/builds/122",
        }
        ui_build = {**webhook_build, "source": "ui"}
        command_build = {
            **webhook_build,
            "meta_data": {"github-comment-id": "98"},
            "source": "api",
        }
        for build, allowed in (
            (webhook_build, True),
            (ui_build, False),
            (command_build, False),
        ):
            with self.subTest(source=build["source"]):
                github = FakeGitHub(
                    permission="read",
                    pr=make_pr(labels=[{"name": "ready"}]),
                )
                buildkite = FakeBuildkite([[], [build]])

                run(make_event(COMMAND_RUN_AMD_CI, "author"), github, buildkite)

                self.assertEqual(bool(buildkite.created_builds), allowed)
                if not allowed:
                    self.assertIn(
                        "AMD CI already ran on this commit", github.comments[0]
                    )

    def test_run_all_sets_buildkite_environment(self) -> None:
        github = FakeGitHub()
        buildkite = FakeBuildkite([[], []])

        run(make_event(COMMAND_RUN_CI_ALL), github, buildkite)

        payload = buildkite.created_builds[0]
        self.assertEqual(payload["message"], "PR #42 /ci run all by @reviewer")
        self.assertEqual(payload["env"]["RUN_ALL"], "1")
        self.assertNotIn("NIGHTLY", payload["env"])

    def test_run_nightly_sets_buildkite_environment(self) -> None:
        github = FakeGitHub()
        buildkite = FakeBuildkite([[], []])

        run(make_event(COMMAND_RUN_CI_NIGHTLY), github, buildkite)

        payload = buildkite.created_builds[0]
        self.assertEqual(payload["message"], "PR #42 /ci run nightly by @reviewer")
        self.assertEqual(payload["env"]["RUN_ALL"], "1")
        self.assertEqual(payload["env"]["NIGHTLY"], "1")

    def test_amd_run_variants_set_buildkite_environment(self) -> None:
        cases = (
            (COMMAND_RUN_AMD_CI, {}),
            (COMMAND_RUN_AMD_CI_ALL, {"RUN_ALL": "1"}),
            (
                COMMAND_RUN_AMD_CI_NIGHTLY,
                {"RUN_ALL": "1", "NIGHTLY": "1"},
            ),
        )
        for command, expected_env in cases:
            with self.subTest(command=command):
                github = FakeGitHub()
                buildkite = FakeBuildkite([[], []])

                run(make_event(command), github, buildkite)

                payload = buildkite.created_builds[0]
                self.assertEqual(payload["message"], f"PR #42 {command} by @reviewer")
                command_env = {
                    key: value
                    for key, value in payload["env"].items()
                    if key in {"RUN_ALL", "NIGHTLY"}
                }
                self.assertEqual(command_env, expected_env)
                self.assertIn("Buildkite AMD CI #123", github.comments[0])

    def test_unapproved_authors_are_denied_without_buildkite(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(),
            review_decision="REVIEW_REQUIRED",
        )
        buildkite = FakeBuildkite()
        run(make_event(COMMAND_RUN_CI, "author"), github, buildkite)

        self.assertEqual(buildkite.list_calls, [])
        self.assertEqual(github.lag_queries, [])
        self.assertEqual(github.reactions, ["eyes"])
        self.assertTrue(github.comments[0].startswith("❌ "))
        self.assertIn("approve the PR", github.comments[0])

        run(make_event(COMMAND_RUN_CI, "author"), github, buildkite)
        self.assertEqual(len(github.comments), 1)

    def test_unapproved_author_gets_amd_specific_guidance(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(),
            review_decision="REVIEW_REQUIRED",
        )
        buildkite = FakeBuildkite()

        run(make_event(COMMAND_RUN_AMD_CI, "author"), github, buildkite)

        self.assertEqual(buildkite.list_calls, [])
        self.assertIn("`/amd-ci run`", github.comments[0])
        self.assertNotIn("`/ci run`", github.comments[0])

    def test_untrusted_approval_cannot_launch_ci(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(),
            review_decision="APPROVED",
            reviews=[
                {
                    "state": "APPROVED",
                    "user": {"login": "untrusted-reviewer"},
                }
            ],
        )
        buildkite = FakeBuildkite()

        run(make_event(COMMAND_RUN_CI, "author"), github, buildkite)

        self.assertEqual(buildkite.list_calls, [])
        self.assertEqual(github.reactions, ["eyes"])
        self.assertTrue(github.comments[0].startswith("❌ "))

    def test_ready_label_notifies_author_once(self) -> None:
        pr = make_pr(labels=[{"name": "ready"}])
        event = {
            "action": "labeled",
            "label": {"name": "ready"},
            "pull_request": pr,
        }
        github = FakeGitHub(permission="read", pr=pr)

        notify_authorized(event, github)
        notify_authorized(event, github)

        self.assertEqual(len(github.comments), 1)
        comment = github.comments[0]
        self.assertTrue(comment.startswith("✅ @author"))
        self.assertIn("`/ci run` starts upstream CI", comment)
        self.assertIn("`/ci retry` retries failed jobs", comment)
        self.assertIn("`/ci cancel` cancels scheduled or running", comment)
        self.assertIn("`/amd-ci run` starts AMD CI only", comment)
        self.assertIn("`/amd-ci retry` retries failed jobs in AMD CI", comment)
        self.assertIn("`/amd-ci cancel` does the same for AMD CI only", comment)
        self.assertIn("CI build for the current PR head", comment)
        self.assertIn("only jobs that failed in the latest earlier CI build", comment)
        self.assertNotIn(COMMAND_RUN_CI_ALL, comment)
        self.assertNotIn(COMMAND_RUN_CI_NIGHTLY, comment)
        self.assertIn(CI_AUTHORIZED_COMMENT_MARKER, comment)

    def test_ready_label_notifies_after_missed_approval_notification(self) -> None:
        pr = make_pr(labels=[{"name": "ready"}])
        event = {
            "action": "labeled",
            "label": {"name": "ready"},
            "pull_request": pr,
        }
        github = FakeGitHub(
            permission="read",
            permissions={"reviewer": "write"},
            pr=pr,
            review_decision="APPROVED",
            reviews=[
                {
                    "state": "APPROVED",
                    "user": {"login": "reviewer"},
                }
            ],
        )

        notify_authorized(event, github)

        self.assertEqual(len(github.comments), 1)
        self.assertIn("@author", github.comments[0])

    def test_trusted_approval_notifies_author_once(self) -> None:
        event = {
            "action": "submitted",
            "pull_request": make_pr(),
            "review": {"state": "approved"},
        }
        github = FakeGitHub(
            permission="read",
            permissions={"reviewer": "write"},
            review_decision="APPROVED",
            reviews=[
                {
                    "state": "APPROVED",
                    "user": {"login": "reviewer"},
                }
            ],
        )

        notify_authorized(event, github)
        notify_authorized(event, github)

        self.assertEqual(len(github.comments), 1)
        self.assertIn("@author", github.comments[0])

    def test_workflow_run_resolves_pr_from_head_commit(self) -> None:
        pr = make_pr()
        github = FakeGitHub(
            pr=pr,
            pulls_for_commit=[{"number": pr["number"]}],
        )

        resolved = resolve_workflow_run_pr(
            {"head_sha": pr["head"]["sha"], "pull_requests": []},
            github,
        )

        self.assertEqual(resolved, pr)

    def test_workflow_run_ignores_unrelated_pr_association(self) -> None:
        pr = make_pr()
        github = FakeGitHub(
            pr=pr,
            pulls_for_commit=[{"number": pr["number"]}],
        )

        resolved = resolve_workflow_run_pr(
            {
                "head_sha": pr["head"]["sha"],
                "pull_requests": [{"number": 1}],
            },
            github,
        )

        self.assertEqual(resolved, pr)

    def test_approval_does_not_notify_when_ready_label_exists(self) -> None:
        pr = make_pr(labels=[{"name": "ready"}])
        event = {
            "action": "submitted",
            "pull_request": pr,
            "review": {"state": "approved"},
        }
        github = FakeGitHub(
            permission="read",
            permissions={"reviewer": "write"},
            pr=pr,
            review_decision="APPROVED",
            reviews=[
                {
                    "state": "APPROVED",
                    "user": {"login": "reviewer"},
                }
            ],
        )

        notify_authorized(event, github)

        self.assertEqual(github.comments, [])

    def test_untrusted_approval_does_not_notify_author(self) -> None:
        event = {
            "action": "submitted",
            "pull_request": make_pr(),
            "review": {"state": "approved"},
        }
        github = FakeGitHub(
            permission="read",
            review_decision="APPROVED",
            reviews=[
                {
                    "state": "APPROVED",
                    "user": {"login": "reviewer"},
                }
            ],
        )

        notify_authorized(event, github)

        self.assertEqual(github.comments, [])

    def test_notification_skips_authors_who_already_have_write(self) -> None:
        pr = make_pr(labels=[{"name": "ready"}])
        event = {
            "action": "labeled",
            "label": {"name": "ready"},
            "pull_request": pr,
        }
        github = FakeGitHub(permission="write", pr=pr)

        notify_authorized(event, github)

        self.assertEqual(github.comments, [])

    def test_notification_skips_draft_prs(self) -> None:
        pr = make_pr(draft=True, labels=[{"name": "ready"}])
        event = {
            "action": "labeled",
            "label": {"name": "ready"},
            "pull_request": pr,
        }
        github = FakeGitHub(permission="read", pr=pr)

        notify_authorized(event, github)

        self.assertEqual(github.comments, [])

    def test_ci_retry_retries_failed_jobs_while_build_is_running(self) -> None:
        github = FakeGitHub(
            behind=100,
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        buildkite = FakeBuildkite(
            [
                [
                    {
                        "created_at": "2026-07-28T01:00:00Z",
                        "number": 123,
                        "pull_request": {"id": 42},
                        "state": "failing",
                        "web_url": "https://buildkite.example/builds/123",
                    }
                ]
            ]
        )
        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.retry_calls, [(123, RETRY_STATES)])
        self.assertEqual(github.lag_queries, [])
        self.assertIn("Queued 3 failed job", github.comments[0])

    def test_amd_ci_retry_retries_only_the_current_head_build(self) -> None:
        github = FakeGitHub(
            behind=100,
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        buildkite = FakeBuildkite(
            [
                [
                    {
                        "created_at": "2026-07-28T01:00:00Z",
                        "number": 321,
                        "pull_request": {"id": 42},
                        "state": "failing",
                        "web_url": "https://buildkite.example/amd-ci/builds/321",
                    }
                ]
            ]
        )

        run(make_event(COMMAND_RETRY_AMD_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.retry_calls, [(321, RETRY_STATES)])
        self.assertEqual(github.lag_queries, [])
        self.assertIn("Buildkite AMD CI #321", github.comments[0])

    def test_amd_ci_retry_requires_a_build_for_the_current_head(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        buildkite = FakeBuildkite([[]])

        run(make_event(COMMAND_RETRY_AMD_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.list_calls, [("0123456789abcdef", None)])
        self.assertEqual(buildkite.retry_calls, [])
        self.assertEqual(buildkite.created_builds, [])
        self.assertIn(
            "No AMD CI build exists for the current PR head", github.comments[0]
        )
        self.assertIn("Use `/amd-ci run`", github.comments[0])

    def test_amd_ci_retry_ignores_webhook_build_for_the_current_head(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        webhook_build = {
            "created_at": "2026-08-18T01:00:00Z",
            "meta_data": {},
            "number": 122,
            "pull_request": {"id": 42},
            "source": "webhook",
            "state": "passed",
            "web_url": "https://buildkite.example/amd-ci/builds/122",
        }
        buildkite = FakeBuildkite([[webhook_build]])

        run(make_event(COMMAND_RETRY_AMD_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.retry_calls, [])
        self.assertIn(
            "No AMD CI build exists for the current PR head", github.comments[0]
        )

    def test_ci_retry_creates_filtered_build_for_new_head(self) -> None:
        github = FakeGitHub(
            behind=100,
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        source_build = {
            "commit": "old-commit",
            "created_at": "2026-07-28T01:00:00Z",
            "finished_at": "2026-07-28T02:00:00Z",
            "number": 122,
            "pull_request": {"id": 42},
            "state": "failed",
            "web_url": "https://buildkite.example/builds/122",
        }
        buildkite = FakeBuildkite(
            [[], [source_build]],
            [
                [
                    {
                        "state": "failed",
                        "step_key": "basic-models-test-other-cpu",
                        "type": "script",
                    },
                    {
                        "state": "failed",
                        "step_key": "basic-models-test-other-cpu",
                        "type": "script",
                    },
                    {
                        "state": "timed_out",
                        "step_key": "distributed-tests-2xh100-2xmi300",
                        "type": "script",
                    },
                ]
            ],
        )

        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.job_list_calls, [122])
        self.assertEqual(len(buildkite.created_builds), 1)
        payload = buildkite.created_builds[0]
        self.assertEqual(payload["commit"], "0123456789abcdef")
        self.assertEqual(github.lag_queries, [])
        self.assertEqual(payload["message"], "PR #42 /ci retry by @author")
        self.assertEqual(
            json.loads(payload["env"]["VLLM_CI_ONLY_STEP_KEYS"]),
            [
                "basic-models-test-other-cpu",
                "distributed-tests-2xh100-2xmi300",
            ],
        )
        self.assertEqual(
            payload["meta_data"]["github-retry-source-build"],
            "122",
        )
        self.assertEqual(
            payload["meta_data"]["github-retry-source-commit"],
            "old-commit",
        )
        self.assertIn("running 2 failed step", github.comments[0])
        self.assertIn("Buildkite CI #122", github.comments[0])

    def test_ci_retry_new_head_accepts_finished_build_left_blocked(self) -> None:
        # The "Run more tests" picker leaves a finished build in state blocked.
        github = FakeGitHub(permission="read", pr=make_pr(labels=[{"name": "ready"}]))
        source_build = {
            "blocked": True,
            "commit": "old-commit",
            "created_at": "2026-11-21T04:16:19Z",
            "finished_at": "2026-11-23T23:04:32Z",
            "number": 122,
            "pull_request": {"id": 42},
            "state": "blocked",
            "web_url": "https://buildkite.example/builds/122",
        }
        buildkite = FakeBuildkite(
            [[], [source_build]],
            [
                [
                    {
                        "state": "failed",
                        "step_key": "basic-models-test-other-cpu",
                        "type": "script",
                    }
                ]
            ],
        )

        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(len(buildkite.created_builds), 1)
        self.assertNotIn("still running", github.comments[0])

    def test_ci_retry_new_head_waits_for_blocked_build_with_unfinished_jobs(
        self,
    ) -> None:
        github = FakeGitHub(permission="read", pr=make_pr(labels=[{"name": "ready"}]))
        source_build = {
            "blocked": True,
            "commit": "old-commit",
            "created_at": "2026-11-21T04:16:19Z",
            "finished_at": None,
            "number": 122,
            "pull_request": {"id": 42},
            "state": "blocked",
            "web_url": "https://buildkite.example/builds/122",
        }
        buildkite = FakeBuildkite([[], [source_build]])

        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.created_builds, [])
        self.assertIn("still running", github.comments[0])

    def test_ci_retry_new_head_requires_stable_step_keys(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        buildkite = FakeBuildkite(
            [
                [],
                [
                    {
                        "commit": "old-commit",
                        "created_at": "2026-07-28T01:00:00Z",
                        "finished_at": "2026-07-28T02:00:00Z",
                        "number": 122,
                        "pull_request": {"id": 42},
                        "state": "failed",
                        "web_url": "https://buildkite.example/builds/122",
                    }
                ],
            ],
            [[{"state": "failed", "step_key": None, "type": "script"}]],
        )

        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.created_builds, [])
        self.assertIn("without stable step keys", github.comments[0])
        self.assertIn("Use `/ci run`", github.comments[0])

    def test_ci_retry_new_head_rejects_incomplete_setup_failure(self) -> None:
        github = FakeGitHub(
            permission="read",
            pr=make_pr(labels=[{"name": "ready"}]),
        )
        buildkite = FakeBuildkite(
            [
                [],
                [
                    {
                        "commit": "old-commit",
                        "created_at": "2026-07-28T01:00:00Z",
                        "finished_at": "2026-07-28T02:00:00Z",
                        "number": 122,
                        "pull_request": {"id": 42},
                        "state": "failed",
                        "web_url": "https://buildkite.example/builds/122",
                    }
                ],
            ],
            [
                [
                    {
                        "state": "failed",
                        "step_key": "image-build",
                        "type": "script",
                    }
                ]
            ],
        )

        run(make_event(COMMAND_RETRY_FAILED, "author"), github, buildkite)

        self.assertEqual(buildkite.created_builds, [])
        self.assertIn("failed during CI setup", github.comments[0])
        self.assertIn("Use `/ci run`", github.comments[0])

    def test_buildkite_retry_uses_retry_failed_jobs_endpoint(self) -> None:
        transport = FakeTransport({"retried_jobs_count": 2})
        client = BuildkiteClient(
            "secret",
            "vllm",
            "ci",
            transport=transport,
        )
        client.retry_failed_jobs(123, RETRY_STATES)

        call = transport.calls[0]
        self.assertEqual(call["method"], "PUT")
        self.assertTrue(call["url"].endswith("/123/retry_failed_jobs"))
        self.assertEqual(call["body"], {"states": RETRY_STATES})

    def test_ci_cancel_cancels_active_builds_for_pr_branch(self) -> None:
        github = FakeGitHub()
        buildkite = FakeBuildkite(
            [
                # Builds created under the head label (owner:branch).
                [
                    {
                        "branch": "contributor:feature",
                        "number": 123,
                        "pull_request": {"id": 42},
                        "state": "running",
                        "web_url": "https://buildkite.example/builds/123",
                    },
                    {
                        "branch": "contributor:feature",
                        "number": 124,
                        "meta_data": {"github-pr-number": "42"},
                        "state": "failing",
                        "web_url": "https://buildkite.example/builds/124",
                    },
                    {
                        "branch": "contributor:feature",
                        "number": 125,
                        "pull_request": {"id": 43},
                        "state": "running",
                        "web_url": "https://buildkite.example/builds/125",
                    },
                    {
                        "branch": "contributor:feature",
                        "number": 127,
                        "pull_request": {"id": 42},
                        "state": "passed",
                        "web_url": "https://buildkite.example/builds/127",
                    },
                ],
                # Legacy builds created under the bare ref before the label change.
                [
                    {
                        "branch": "feature",
                        "number": 126,
                        "pull_request": {"id": 42},
                        "state": "running",
                        "web_url": "https://buildkite.example/builds/126",
                    },
                ],
            ]
        )

        run(make_event(COMMAND_CANCEL_CI), github, buildkite)

        self.assertEqual(buildkite.cancel_calls, [123, 124, 126])
        self.assertEqual(github.lag_queries, [])
        self.assertEqual(
            [request["branch"] for request in buildkite.list_requests],
            ["contributor:feature", "feature"],
        )
        self.assertIn("Requested cancellation of 3 CI builds", github.comments[0])
        self.assertIn("#123", github.comments[0])
        self.assertIn("#124", github.comments[0])
        self.assertIn("#126", github.comments[0])

    def test_ci_cancel_is_a_noop_without_active_builds(self) -> None:
        github = FakeGitHub()
        buildkite = FakeBuildkite([[], []])

        run(make_event(COMMAND_CANCEL_CI), github, buildkite)

        self.assertEqual(buildkite.cancel_calls, [])
        self.assertIn("No cancelable CI build is running", github.comments[0])

    def test_amd_ci_cancel_handles_command_and_fork_webhook_branches(self) -> None:
        pr = make_pr()
        pr["head"]["label"] = "contributor:feature"
        github = FakeGitHub(pr=pr)
        buildkite = FakeBuildkite(
            [
                [
                    {
                        "branch": "contributor:feature",
                        "number": 322,
                        "pull_request": {"id": 42},
                        "state": "failing",
                        "web_url": "https://buildkite.example/amd-ci/builds/322",
                    }
                ],
                [
                    {
                        "branch": "feature",
                        "number": 321,
                        "pull_request": {"id": 42},
                        "state": "running",
                        "web_url": "https://buildkite.example/amd-ci/builds/321",
                    }
                ],
            ]
        )

        run(make_event(COMMAND_CANCEL_AMD_CI), github, buildkite)

        self.assertEqual(buildkite.cancel_calls, [322, 321])
        self.assertEqual(github.lag_queries, [])
        self.assertEqual(
            [request["branch"] for request in buildkite.list_requests],
            ["contributor:feature", "feature"],
        )
        self.assertTrue(
            all(
                request["states"] == CANCELABLE_BUILD_STATES
                for request in buildkite.list_requests
            )
        )
        self.assertIn("cancellation of 2 AMD CI builds", github.comments[0])

    def test_buildkite_create_build_ignores_pipeline_branch_filters(self) -> None:
        transport = FakeTransport({"number": 123})
        client = BuildkiteClient(
            "secret",
            "vllm",
            "amd-ci",
            transport=transport,
        )

        client.create_build({"branch": "main", "commit": "HEAD"})

        call = transport.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(
            call["body"],
            {
                "branch": "main",
                "commit": "HEAD",
                "ignore_pipeline_branch_filters": True,
            },
        )

    def test_buildkite_cancel_uses_cancel_build_endpoint(self) -> None:
        transport = FakeTransport({"number": 123, "state": "canceling"})
        client = BuildkiteClient(
            "secret",
            "vllm",
            "ci",
            transport=transport,
        )

        client.cancel_build(123)

        call = transport.calls[0]
        self.assertEqual(call["method"], "PUT")
        self.assertTrue(call["url"].endswith("/123/cancel"))
        self.assertIsNone(call["body"])

    def test_buildkite_list_builds_allows_query_on_builds_endpoint(self) -> None:
        transport = FakeTransport([])
        client = BuildkiteClient(
            "secret",
            "vllm",
            "ci",
            transport=transport,
        )

        builds = client.list_builds(
            "current-commit",
            branch="feature",
            metadata=("github-pr-number", "42"),
            states=CANCELABLE_BUILD_STATES,
        )
        client.list_builds(None, page=2, created_from="2026-01-01T00:00:00Z")

        self.assertEqual(builds, [])
        self.assertNotIn("&page=", transport.calls[0]["url"])
        self.assertIn("&page=2", transport.calls[1]["url"])
        self.assertIn(
            "created_from=2026-01-01T00%3A00%3A00Z", transport.calls[1]["url"]
        )
        url = transport.calls[0]["url"]
        self.assertIn("?exclude_jobs=true", url)
        self.assertIn("commit=current-commit", url)
        self.assertIn("branch=feature", url)
        self.assertIn("meta_data%5Bgithub-pr-number%5D=42", url)
        self.assertIn("state%5B%5D=scheduled", url)
        self.assertIn("state%5B%5D=running", url)
        self.assertIn("state%5B%5D=failing", url)

    def test_buildkite_failed_jobs_follow_cursor_pagination(self) -> None:
        next_url = (
            "https://api.buildkite.com/v2/organizations/vllm/pipelines/ci/"
            "builds/123/jobs?after=cursor"
        )
        transport = FakeTransport(
            (
                {
                    "items": [{"id": "first"}],
                    "links": {"next": next_url},
                },
                {
                    "items": [{"id": "second"}],
                    "links": {"next": None},
                },
            )
        )
        client = BuildkiteClient(
            "secret",
            "vllm",
            "ci",
            transport=transport,
        )

        jobs = client.list_failed_jobs(123)

        self.assertEqual(jobs, [{"id": "first"}, {"id": "second"}])
        self.assertIn("state%5B%5D=failed", transport.calls[0]["url"])
        self.assertIn("include_retried_jobs=false", transport.calls[0]["url"])
        self.assertEqual(transport.calls[1]["url"], next_url)


STACK_PR_FILES = ["docker/Dockerfile.rocm_base"]
STACK_BUILT = {"rocm-rocm-stack-built": "s@d1 w@d2"}
REFRESH_PASSED = {
    "finished_at": "1",
    "id": "r",
    "state": "passed",
    "step_key": "refresh-rocm-base-amd",
}
CODEOWNERS = (
    "/docker/ @docker-owner\n"
    "# ROCm\n"
    "/docker/Dockerfile.rocm* @tjtanaa @AndreasKaratzas\n"
)


def rocm_build(**overrides: Any) -> dict[str, Any]:
    """A CI build of PR #42 that a write user requested and that pushed a stack."""
    return {
        "branch": "contributor:feature",
        "commit": "fedcba9876543210",
        "created_at": "1",
        "env": {"VLLM_CI_TRIGGERED_BY": "reviewer"},
        "finished_at": "2",
        "meta_data": STACK_BUILT,
        "number": 10,
        "pull_request": {"id": "42"},
        "source": "api",
        "state": "failed",
        "web_url": "https://buildkite.example/ci/builds/10",
        **overrides,
    }


def gate_job(**overrides: Any) -> dict[str, Any]:
    return {
        "finished_at": "3",
        "id": "gate",
        "started_at": "3",
        "state": "failed",
        "step_key": ROCM_CACHE_CHECK_KEY,
        **overrides,
    }


class FakeRocmGitHub(FakeGitHub):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("files", STACK_PR_FILES)
        super().__init__(**kwargs)
        self.file_texts = {(".github/CODEOWNERS", "main"): CODEOWNERS}

    def get_file_text(self, path: str, ref: str) -> str:
        return self.file_texts.get((path, ref), "")

    def get_merge_base(self, base_ref: str, head_sha: str) -> str:
        return f"merge-base-of-{head_sha}"


class FakeRocmBuildkite:
    """Filters builds and jobs like the Buildkite API, and records writes."""

    def __init__(
        self,
        builds: list[dict[str, Any]] | None = None,
        jobs: dict[int, list[dict[str, Any]]] | None = None,
        retry_error: ApiError | None = None,
    ) -> None:
        self.builds = builds or []
        self.jobs = jobs or {}
        self.retry_error = retry_error
        self.created_builds: list[dict[str, Any]] = []
        self.retried_jobs: list[tuple[int, str]] = []

    def list_builds(
        self,
        commit: str | None,
        *,
        branch: str | None = None,
        metadata: tuple[str, str] | None = None,
        states: tuple[str, ...] = (),
        page: int = 1,
        created_from: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            build
            for build in self.builds
            if page == 1
            and commit in (None, build.get("commit"))
            and branch in (None, build.get("branch"))
            and (
                metadata is None
                or (build.get("meta_data") or {}).get(metadata[0]) == metadata[1]
            )
        ]

    def get_build(self, build_number: int) -> dict[str, Any]:
        for build in self.builds:
            if build["number"] == build_number:
                return build
        raise ApiError(404, "Not found")

    def create_build(self, body: dict[str, Any]) -> dict[str, Any]:
        self.created_builds.append(body)
        build = {
            "created_at": "4",
            "finished_at": None,
            "number": 500 + len(self.created_builds),
            "source": "api",
            "state": "scheduled",
            **body,
        }
        build["web_url"] = f"https://buildkite.example/builds/{build['number']}"
        self.builds.append(build)
        return build

    def list_jobs(
        self,
        build_number: int,
        *,
        states: tuple[str, ...] = (),
        step_key: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            job
            for job in self.jobs.get(build_number, [])
            if (not states or job["state"] in states)
            and step_key in (None, job.get("step_key"))
        ]

    def retry_job(self, build_number: int, job_id: str) -> dict[str, Any]:
        self.retried_jobs.append((build_number, job_id))
        if self.retry_error:
            raise self.retry_error
        return {}


def promote_build(**overrides: Any) -> dict[str, Any]:
    return {
        "branch": "main",
        "commit": "merge-base-of-fedcba9876543210",
        "created_at": "4",
        "env": {"ROCM_PROMOTE_PR": "42", "ROCM_PROMOTE_SHA": "fedcba9876543210"},
        "finished_at": "5",
        "meta_data": {"rocm-promote": "42@fedcba9876543210"},
        "number": 501,
        "source": "api",
        "state": "passed",
        "web_url": "https://buildkite.example/builds/501",
        **overrides,
    }


def head_ci(checks: list[dict[str, Any]], *builds: Any, **kwargs: Any) -> Any:
    """CI with source build 10, the given builds, and PR head build 20."""
    head = rocm_build(number=20, commit="0123456789abcdef", meta_data={})
    return FakeRocmBuildkite(
        [rocm_build(), *builds, head], {10: [REFRESH_PASSED], 20: checks}, **kwargs
    )


class RocmImageCacheTest(unittest.TestCase):
    def reconcile(
        self,
        github: FakeRocmGitHub,
        ci: FakeRocmBuildkite,
        amd: FakeRocmBuildkite | None = None,
        trusted_users: set[str] | None = None,
    ) -> str:
        return reconcile_rocm_cache(
            github.pr, github, ci, amd or FakeRocmBuildkite(), trusted_users or set()
        )

    def test_ci_run_by_a_trusted_user_starts_caching(self) -> None:
        """Write access and CI_TRUSTED_USERS both make a /ci run cacheable."""
        bake = [*STACK_PR_FILES, "docker/docker-bake-rocm.hcl"]
        cases = {
            "write": ("write", "", STACK_PR_FILES),
            "trusted users": ("read", "Reviewer", STACK_PR_FILES),
            # Like main's own, its images are keyed on their inputs, not the recipe.
            "recipe change": ("write", "", bake),
        }
        for name, (permission, trusted_users, files) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub(permission=permission, files=files)
                # An earlier build this reviewer requested already pushed the stack.
                ci = FakeRocmBuildkite([rocm_build()], {10: [REFRESH_PASSED]})

                run(
                    make_event(COMMAND_RUN_CI),
                    github,
                    ci,
                    trusted_users_value=trusted_users,
                    ci_buildkite=ci,
                )

                _, promote = ci.created_builds
                self.assertEqual(promote["branch"], "main")
                self.assertIn("cached into `main` automatically", github.comments[0])
                self.assertIn("`ROCm image cache` step", github.comments[0])

    def test_ci_run_after_an_approval_caches_the_build_ci_already_ran(self) -> None:
        head = "0123456789abcdef"
        approval = {"commit_id": head, "state": "APPROVED", "user": {"login": "x"}}
        github = FakeRocmGitHub(
            permission="read",
            permissions={"x": "write"},
            review_decision="APPROVED",
            reviews=[approval],
        )
        build = rocm_build(commit=head, env={"VLLM_CI_TRIGGERED_BY": "author"})
        ci = FakeRocmBuildkite([build], {10: [REFRESH_PASSED]})

        run(make_event(COMMAND_RUN_CI, "author"), github, ci, ci_buildkite=ci)

        # The repeat run is refused, but the approval vouches for the build.
        self.assertIn("already ran on this commit", github.comments[0])
        (promote,) = ci.created_builds
        self.assertEqual(promote["env"]["ROCM_PROMOTE_SHA"], head)

    def test_ci_run_note_is_about_the_head_the_run_built(self) -> None:
        # The approval vouches for the old head, not for main merged into it.
        approval = {
            "commit_id": "0123456789abcdef",
            "state": "APPROVED",
            "user": {"login": "x"},
        }
        github = FakeRocmGitHub(
            behind=1,
            permission="read",
            permissions={"x": "write"},
            review_decision="APPROVED",
            reviews=[approval],
        )
        ci = FakeRocmBuildkite()

        run(
            make_event(COMMAND_RUN_CI, "author"),
            github,
            ci,
            branch_updater=FakeBranchUpdater(github),
            ci_buildkite=ci,
        )

        self.assertIn("merge-commit-sha", json.dumps(ci.created_builds))
        self.assertIn("a trusted reviewer must approve this commit", github.comments[0])

    def test_ci_run_starts_even_when_the_rocm_note_fails(self) -> None:
        github = FakeRocmGitHub()
        ci = FakeRocmBuildkite()

        with patch.object(
            github, "get_file_text", side_effect=ApiError(502, "Bad gateway")
        ):
            run(make_event(COMMAND_RUN_CI), github, ci, ci_buildkite=ci)

        self.assertEqual(len(ci.created_builds), 1)
        self.assertTrue(github.comments[0].startswith("✅"))

    def test_ci_run_by_an_untrusted_author_asks_for_approval_or_a_run(self) -> None:
        github = FakeRocmGitHub(
            permission="read", pr=make_pr(labels=[{"name": "ready"}])
        )
        ci = FakeRocmBuildkite()

        run(make_event(COMMAND_RUN_CI, "author"), github, ci, ci_buildkite=ci)

        self.assertEqual(len(ci.created_builds), 1)
        self.assertIn(
            "@author @tjtanaa @AndreasKaratzas: this PR changes the ROCm",
            github.comments[0],
        )
        self.assertNotIn("@docker-owner", github.comments[0])
        self.assertIn(
            "must comment `/ci run`, or a trusted reviewer must approve this commit",
            github.comments[0],
        )

    def test_ci_run_of_an_approved_head_pings_nobody(self) -> None:
        """A trusted approval of the head already makes the new build cacheable."""
        review = {
            "commit_id": "0123456789abcdef",
            "state": "APPROVED",
            "user": {"login": "reviewer"},
        }
        github = FakeRocmGitHub(
            permission="read",
            permissions={"reviewer": "write"},
            review_decision="APPROVED",
            reviews=[review],
        )
        ci = FakeRocmBuildkite()

        run(make_event(COMMAND_RUN_CI, "author"), github, ci, ci_buildkite=ci)

        self.assertIn("cached into `main` automatically", github.comments[0])
        self.assertNotIn("@tjtanaa", github.comments[0])

    def test_ci_run_on_other_prs_leaves_the_rocm_cache_alone(self) -> None:
        for files, base in ((["vllm/foo.py"], "main"), (STACK_PR_FILES, "v0.20")):
            with self.subTest(base=base):
                pr = make_pr(base={"ref": base})
                github = FakeRocmGitHub(files=files, pr=pr)
                ci = FakeRocmBuildkite()

                run(make_event(COMMAND_RUN_CI), github, ci, ci_buildkite=ci)

                self.assertNotIn("ROCm", github.comments[0])

    def test_repo_codeowners_cover_every_rocm_stack_input(self) -> None:
        with open(CODEOWNERS_FILE, encoding="utf-8") as file:
            codeowners = file.read()

        for path in ROCM_STACK_INPUTS:
            with self.subTest(path=path):
                self.assertEqual(
                    codeowners_for(codeowners, [path]),
                    ["@tjtanaa", "@dllehr-amd", "@AndreasKaratzas"],
                )

    def test_reconcile_creates_one_promote_build_at_the_merge_base(self) -> None:
        github = FakeRocmGitHub()
        meta_data = {**STACK_BUILT, "rocm-rocm-runtime-built": "r@d3"}
        ci = FakeRocmBuildkite(
            [rocm_build(meta_data=meta_data)], {10: [REFRESH_PASSED]}
        )

        self.reconcile(github, ci)
        self.reconcile(github, ci)

        (build,) = ci.created_builds
        self.assertEqual(build["commit"], "merge-base-of-fedcba9876543210")
        self.assertEqual(build["branch"], "main")
        self.assertEqual(
            build["env"],
            {
                "ROCM_PROMOTE_PR": "42",
                "ROCM_PROMOTE_SHA": "fedcba9876543210",
                "ROCM_PROMOTE_IMAGES": "s@d1 w@d2 r@d3",
                "VLLM_CI_ONLY_STEP_KEYS": '["rocm-base-image-cache"]',
            },
        )
        self.assertEqual(ci.retried_jobs, [])

    def test_reconcile_promotes_a_stack_whose_base_image_step_failed_later(
        self,
    ) -> None:
        github = FakeRocmGitHub()
        refresh = {**REFRESH_PASSED, "state": "failed"}
        ci = FakeRocmBuildkite([rocm_build()], {10: [refresh]})

        self.reconcile(github, ci)

        self.assertEqual(len(ci.created_builds), 1)

    def test_reconcile_promotes_amd_ci_builds_too(self) -> None:
        github = FakeRocmGitHub()
        ci = FakeRocmBuildkite()
        amd = FakeRocmBuildkite([rocm_build(number=7)], {7: [REFRESH_PASSED]})

        self.reconcile(github, ci, amd)

        self.assertEqual(len(ci.created_builds), 1)
        self.assertEqual(amd.created_builds, [])

    def test_reconcile_only_caches_what_a_write_users_build_pushed(self) -> None:
        github = FakeRocmGitHub(permissions={"author": "read"})
        github.file_shas = {("docker/Dockerfile.rocm_base", "old"): "older"}
        jobs = {n: [REFRESH_PASSED] for n in range(10, 20)}
        jobs[16] = [{**REFRESH_PASSED, "state": "running", "finished_at": None}]
        newer = [
            rocm_build(
                number=11, commit="other", env={"VLLM_CI_TRIGGERED_BY": "author"}
            ),
            rocm_build(number=12, source="webhook"),
            rocm_build(
                number=13,
                pull_request=None,
                meta_data={**STACK_BUILT, "github-pr-number": "42"},
            ),
            rocm_build(number=14, pull_request={"id": "7"}),
            rocm_build(number=15, meta_data={"rocm-rocm-stack": "s@d3"}),
            rocm_build(number=16),
            rocm_build(number=17, commit="old"),
        ]
        cases = {"no source": [], "older source": [rocm_build(created_at="0")]}
        for name, older in cases.items():
            with self.subTest(name):
                builds = [{**build, "created_at": "1"} for build in newer] + older
                ci = FakeRocmBuildkite(builds, jobs)

                self.reconcile(github, ci)

                promotes = [b["env"]["ROCM_PROMOTE_SHA"] for b in ci.created_builds]
                self.assertEqual(promotes, ["fedcba9876543210"] if older else [])

    def test_reconcile_leaves_prs_it_cannot_cache_alone(self) -> None:
        cases = {
            "closed": FakeRocmGitHub(pr=make_pr(state="closed")),
            "stack unchanged": FakeRocmGitHub(files=["vllm/foo.py"]),
        }
        for name, github in cases.items():
            with self.subTest(name):
                ci = FakeRocmBuildkite([rocm_build()], {10: [REFRESH_PASSED]})

                self.reconcile(github, ci)

                self.assertEqual(ci.created_builds, [])

    def test_key_format_reads_the_lines_build_sh_check_compares(self) -> None:
        build_sh = (REPO_ROOT / ROCM_BUILD_SH).read_text()
        keyfmt = (
            r"sed -n '/^case \$STACK in/,/^esac/p;/^_trw=/,/^  Kr=/p'"
            r" | grep -v '^[[:space:]]*#'"
        )
        self.assertIn(keyfmt, build_sh)
        sed = subprocess.run(
            ["bash", "-c", keyfmt],
            input=build_sh,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            sorted(rocm_key_format(build_sh)), sorted(sed.stdout.split("\n")[:-1])
        )
        self.assertIn(
            '  Ks=$( { echo "layers: zstd-3, mtimes kept"',
            "\n".join(rocm_key_format(build_sh)),
        )

    def test_stack_inputs_are_the_files_build_sh_keys_the_stack_on(self) -> None:
        def paths(text: str) -> set[str]:
            for char in "=\"'();":
                text = text.replace(char, " ")
            return {w for w in text.split() if w.startswith(("docker/", "tools/"))}

        build_sh = (REPO_ROOT / ROCM_BUILD_SH).read_text()
        workflow = (REPO_ROOT / ".github/workflows/rocm-image-cache.yml").read_text()
        key_paths = paths("\n".join(rocm_key_format(build_sh)))
        self.assertEqual(key_paths, set(ROCM_STACK_INPUTS))
        self.assertEqual(paths(workflow), set(ROCM_STACK_INPUTS))

    def test_pr_that_changes_how_the_stack_is_keyed_is_not_cached(self) -> None:
        build_sh = (REPO_ROOT / ROCM_BUILD_SH).read_text()
        cases = {
            "stack key": (build_sh.replace("zstd-3", "zstd-5"), False),
            "runtime key only": (
                build_sh.replace("rust-toolchain.toml", "x.toml"),
                True,
            ),
        }
        for name, (head_build_sh, cached) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub(
                    files=[*STACK_PR_FILES, ROCM_BUILD_SH],
                    pr=make_pr(labels=[{"name": "ready"}]),
                )
                github.file_texts.update(
                    {
                        (ROCM_BUILD_SH, "merge-base-of-0123456789abcdef"): build_sh,
                        (ROCM_BUILD_SH, "0123456789abcdef"): head_build_sh,
                    }
                )
                ci = FakeRocmBuildkite([rocm_build()], {10: [REFRESH_PASSED]})

                self.reconcile(github, ci)
                run(make_event(COMMAND_RUN_CI, "author"), github, ci, ci_buildkite=ci)

                promotes = [b for b in ci.created_builds if b["branch"] == "main"]
                self.assertEqual(len(promotes), int(cached))
                promise = "cached into `main` automatically"
                self.assertEqual(promise in github.comments[0], cached)

    def test_main_stack_change_rechecks_open_prs_whose_check_passed(self) -> None:
        head = "0123456789abcdef"
        github = FakeRocmGitHub()
        github.prs.update(
            {
                7: make_pr(number=7, state="closed"),
                8: make_pr(number=8, head={"sha": "moved"}),
                9: make_pr(number=9, base={"ref": "release"}),
            }
        )
        cached = {"rocm-stack-cached": "true"}

        def marked(number: int, pr: int, **overrides: Any) -> dict[str, Any]:
            return rocm_build(
                number=number,
                commit=head,
                meta_data=cached,
                pull_request={"id": str(pr)},
                **overrides,
            )

        passed = [gate_job(state="passed")]
        ci = FakeRocmBuildkite(
            [
                marked(24, 42, source="webhook"),
                marked(20, 42),
                marked(19, 42),  # older build of the same PR
                marked(21, 7),
                marked(22, 8),
                marked(23, 9),
                rocm_build(number=25, commit=head),  # not marked
            ],
            {n: passed for n in range(19, 26)},
        )

        recheck_rocm_stack_prs(github, ci)

        self.assertEqual(ci.retried_jobs, [(20, "gate")])
        for jobs, error in (
            ([gate_job()], None),
            (passed, ApiError(422, "Job cannot be retried")),
        ):
            with self.subTest(jobs=jobs, error=error):
                ci = FakeRocmBuildkite([marked(20, 42)], {20: jobs}, error)

                self.assertEqual(recheck_rocm_stack_prs(github, ci), [])
        # A deleted PR does not stop the others; a server error fails the run
        # once the others are done.
        ci = FakeRocmBuildkite([marked(21, 404), marked(20, 42)], {20: passed})
        self.assertEqual(recheck_rocm_stack_prs(github, ci), [rocm_build()["web_url"]])
        github.prs[7] = make_pr(number=7)
        ci = FakeRocmBuildkite([marked(21, 7), marked(20, 42)], {20: passed})
        with (
            patch.object(ci, "list_jobs", side_effect=[ApiError(502, "x"), passed]),
            self.assertRaisesRegex(ApiError, "Could not recheck PRs"),
        ):
            recheck_rocm_stack_prs(github, ci)
        self.assertEqual(ci.retried_jobs, [(20, "gate")])

    def test_reconcile_retries_only_a_gate_that_failed_before_the_promote(
        self,
    ) -> None:
        github = FakeRocmGitHub()
        other = {**gate_job(), "id": "other", "step_key": "basic-correctness"}
        running = promote_build(state="running", finished_at=None)
        cases = {
            "before": ([gate_job(), other], [(20, "gate")]),
            # It read the registry before the promote pushed, but ended after.
            "across": ([gate_job(finished_at="6")], [(20, "gate")]),
            "after": ([gate_job(started_at="6", finished_at="7")], []),
            "passed": ([gate_job(state="passed")], []),
            "none": ([], []),
            # main would never use this stack: the merge gets another one.
            "main moved": ([gate_job(exit_status=4)], []),
            "promote running": ([gate_job()], [], [running]),
            "passed, no promote": ([gate_job(state="passed")], [], []),
        }
        for name, (jobs, retried, *promotes) in cases.items():
            with self.subTest(name):
                ci = head_ci(jobs, *(promotes[0] if promotes else [promote_build()]))

                self.reconcile(github, ci)

                self.assertEqual(ci.created_builds, [])
                self.assertEqual(ci.retried_jobs, retried)

    def test_reconcile_tolerates_an_already_retried_gate(self) -> None:
        github = FakeRocmGitHub()
        error = ApiError(422, "Job has already been retried")
        ci = head_ci([gate_job()], promote_build(), retry_error=error)

        self.assertIn("Could not retry", self.reconcile(github, ci))

        ci.retry_error = ApiError(500, "Server error")
        with self.assertRaises(ApiError):
            self.reconcile(github, ci)

    def test_ci_retry_on_a_new_head_reruns_the_gate_if_the_head_defines_it(
        self,
    ) -> None:
        source_build = {
            "commit": "old-commit",
            "finished_at": "2",
            "number": 122,
            "pull_request": {"id": 42},
            "state": "failed",
            "web_url": "https://buildkite.example/builds/122",
        }
        failed = [{"state": "failed", "step_key": "basic-test", "type": "script"}]
        head_yaml = (
            "steps:\n  - label: ROCm image cache\n    key: rocm-image-cache-check\n"
        )
        cases = {
            "defined": (
                STACK_PR_FILES,
                head_yaml,
                ["basic-test", ROCM_CACHE_CHECK_KEY],
            ),
            "old branch": (STACK_PR_FILES, "steps: []\n", ["basic-test"]),
            "stack unchanged": (["vllm/foo.py"], head_yaml, ["basic-test"]),
            "not to main": (STACK_PR_FILES, head_yaml, ["basic-test"], "v0.20"),
        }
        for name, (files, pipeline_yaml, step_keys, *base) in cases.items():
            with self.subTest(name):
                pr = make_pr(base={"ref": base[0] if base else "main"})
                github = FakeRocmGitHub(files=files, pr=pr)
                github.file_texts[(ROCM_CACHE_YAML, "0123456789abcdef")] = pipeline_yaml
                buildkite = FakeBuildkite([[], [source_build]], [failed])

                run(make_event(COMMAND_RETRY_FAILED), github, buildkite)

                (payload,) = buildkite.created_builds
                self.assertEqual(
                    json.loads(payload["env"]["VLLM_CI_ONLY_STEP_KEYS"]), step_keys
                )
                self.assertIn("running 1 failed step", github.comments[0])

    def test_cache_pipeline_defines_the_step_keys_the_bot_names(self) -> None:
        with open(REPO_ROOT / ROCM_CACHE_YAML, encoding="utf-8") as file:
            pipeline_yaml = file.read()

        for key in (ROCM_CACHE_STEP_KEY, ROCM_CACHE_CHECK_KEY):
            with self.subTest(key=key):
                self.assertTrue(defines_step_key(pipeline_yaml, key))
        self.assertFalse(defines_step_key(pipeline_yaml, "rocm-image-cache"))
        amd_yaml = (REPO_ROOT / ".buildkite/hardware_tests/amd.yaml").read_text()
        self.assertTrue(defines_step_key(amd_yaml, ROCM_REFRESH_STEP_KEY))

    def test_reconcile_ignores_promote_builds_the_bot_did_not_create(self) -> None:
        cases = {
            "trigger step": {"source": "trigger_job", "triggered_from": {"id": "x"}},
            "ui": {"source": "ui"},
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub()
                fake = promote_build(state="running", finished_at=None, **overrides)
                ci = FakeRocmBuildkite([rocm_build(), fake], {10: [REFRESH_PASSED]})

                self.reconcile(github, ci)

                (build,) = ci.created_builds
                self.assertEqual(build["commit"], "merge-base-of-fedcba9876543210")

    def test_reconcile_trusts_approvals_only_at_the_build_commit(self) -> None:
        """An untrusted user's build is cached once a trusted user approves it."""

        def review(login: str, state: str, commit: str = "fedcba9876543210"):
            return {"commit_id": commit, "state": state, "user": {"login": login}}

        cases = {
            "no approval": ([], 0),
            "write approval": ([review("reviewer", "APPROVED")], 1),
            "trusted user approval": ([review("helper", "APPROVED")], 1),
            "stale approval": ([review("reviewer", "APPROVED", "older")], 0),
            "untrusted approval": ([review("random", "APPROVED")], 0),
            "approval then changes requested": (
                [
                    review("reviewer", "APPROVED"),
                    review("Reviewer", "CHANGES_REQUESTED"),
                ],
                0,
            ),
            "comment after approval": (
                [review("reviewer", "APPROVED"), review("reviewer", "COMMENTED")],
                1,
            ),
            "CI_TRUSTED_USERS build": ([], 1, "Helper"),
            "read user's build": ([], 0, "Random"),
        }
        for name, (reviews, created, *actor) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub(
                    permission="read",
                    permissions={"reviewer": "write"},
                    reviews=reviews,
                )
                env = {"VLLM_CI_TRIGGERED_BY": (actor or ["author"])[0]}
                ci = FakeRocmBuildkite([rocm_build(env=env)], {10: [REFRESH_PASSED]})

                self.reconcile(github, ci, trusted_users={"helper"})

                self.assertEqual(len(ci.created_builds), created)

    def test_reconcile_trusts_every_build_of_a_trusted_commit(self) -> None:
        """A trusted run that reused the stack an earlier build pushed caches it."""
        trusted_run = rocm_build(number=11, created_at="2", meta_data={})
        cases = {
            "same commit": (trusted_run, 1),
            "other commit": ({**trusted_run, "commit": "0123456789abcdef"}, 0),
        }
        for name, (build, created) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub(permissions={"author": "read"})
                pushed = rocm_build(env={"VLLM_CI_TRIGGERED_BY": "author"})
                ci = FakeRocmBuildkite(
                    [pushed, build], {10: [REFRESH_PASSED], 11: [REFRESH_PASSED]}
                )

                self.reconcile(github, ci)

                promotes = [b["env"]["ROCM_PROMOTE_IMAGES"] for b in ci.created_builds]
                self.assertEqual(promotes, ["s@d1 w@d2"] * created)

    def test_status_of_a_pr_build_reconciles_its_pr(self) -> None:
        github = FakeRocmGitHub()
        ci = FakeRocmBuildkite([rocm_build()], {10: [REFRESH_PASSED]})
        event = {
            "sha": "fedcba9876543210",
            "target_url": "https://buildkite.com/vllm/ci/builds/10",
        }

        handle_buildkite_status(event, github, ci, FakeRocmBuildkite(), set())

        (promote,) = ci.created_builds
        self.assertEqual(promote["env"]["ROCM_PROMOTE_PR"], "42")

    def test_a_job_status_of_a_running_build_reconciles_its_pr(self) -> None:
        # The cache check's own status: its build is still running.
        github = FakeRocmGitHub()
        build = rocm_build(state="running", finished_at=None)
        ci = FakeRocmBuildkite([build], {10: [REFRESH_PASSED]})
        event = {
            "sha": "fedcba9876543210",
            "target_url": "https://buildkite.com/vllm/ci/builds/10#01a2106f-eff0",
        }

        handle_buildkite_status(event, github, ci, FakeRocmBuildkite(), set())

        (promote,) = ci.created_builds
        self.assertEqual(promote["env"]["ROCM_PROMOTE_PR"], "42")

    def test_a_promote_is_judged_by_its_cache_job(self) -> None:
        """The promote job's status wakes the bot before its build finishes."""
        running = promote_build(state="running", finished_at=None)
        cache_job = {"id": "c", "step_key": ROCM_CACHE_STEP_KEY, "created_at": "4"}
        cases = {
            "passed": ({**cache_job, "state": "passed", "finished_at": "5"}, 1),
            "retry pending": ({**cache_job, "state": "scheduled"}, 0),
        }
        for name, (job, retried) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub()
                ci = head_ci([gate_job()], running)
                ci.jobs[501] = [job]
                event = {
                    "sha": "merge-base-of-fedcba9876543210",
                    "target_url": "https://buildkite.com/vllm/ci/builds/501#c",
                }

                handle_buildkite_status(event, github, ci, FakeRocmBuildkite(), set())

                self.assertEqual(len(ci.retried_jobs), retried)
                self.assertEqual(ci.created_builds, [])

    def test_the_workflow_listens_to_the_cache_jobs_statuses(self) -> None:
        """ci-infra posts each job's status as `ci/step: <label>`, emoji
        dropped; renaming either step must update rocm-image-cache.yml."""
        steps, label = {}, ""
        for line in (REPO_ROOT / ROCM_CACHE_YAML).read_text().splitlines():
            name, _, value = line.strip().removeprefix("- ").partition(":")
            if name == "label":
                label = value.strip().strip("\"'")
            elif name == "key":
                steps[value.strip()] = label
        for key in (ROCM_CACHE_CHECK_KEY, ROCM_CACHE_STEP_KEY):
            words = steps[key].split()
            label = " ".join(w for w in words if not (w[0] == w[-1] == ":"))
            context = "'ci/step: " + label.replace("'", "''") + "'"
            for workflow in ("rocm-image-cache.yml", "_reusable-rocm-image-cache.yml"):
                with self.subTest(key=key, workflow=workflow):
                    text = (REPO_ROOT / ".github/workflows" / workflow).read_text()
                    self.assertIn(f"github.event.context == {context}", text)

    def test_status_of_a_promote_build_retries_its_prs_check(self) -> None:
        github = FakeRocmGitHub()
        ci = head_ci([gate_job()], promote_build())
        event = {
            "sha": "merge-base-of-fedcba9876543210",
            "target_url": "https://buildkite.com/vllm/ci/builds/501",
        }

        handle_buildkite_status(event, github, ci, FakeRocmBuildkite(), set())

        self.assertEqual(ci.retried_jobs, [(20, "gate")])

    def test_status_of_other_builds_changes_nothing(self) -> None:
        """Only the fetched build, not the status, names the PR to reconcile."""
        cases = {
            "not api": (rocm_build(number=30, source="webhook"), "fedcba9876543210"),
            "forged target": (rocm_build(number=30), "another-commit"),
            "main build": (
                {"branch": "main", "commit": "c", "number": 30, "source": "api"},
                "c",
            ),
            "promote env off main": (
                promote_build(number=30, branch="contributor:feature"),
                "merge-base-of-fedcba9876543210",
            ),
        }
        for name, (build, sha) in cases.items():
            with self.subTest(name):
                github = FakeRocmGitHub()
                # Reconciling PR #42 would promote build 10.
                ci = FakeRocmBuildkite([rocm_build(), build], {10: [REFRESH_PASSED]})
                event = {
                    "sha": sha,
                    "target_url": "https://buildkite.com/vllm/ci/builds/30",
                }

                handle_buildkite_status(event, github, ci, FakeRocmBuildkite(), set())

                self.assertEqual(ci.created_builds, [])

    def test_reconcile_restarts_a_failed_promote_then_asks_for_help(self) -> None:
        github = FakeRocmGitHub()
        ci = head_ci([gate_job()])

        for state in ("failed", "canceled", "failed", "failed"):
            self.reconcile(github, ci)
            ci.builds[-1].update(state=state, finished_at="5")
        self.reconcile(github, ci)

        self.assertEqual(len(ci.created_builds), 3)
        self.assertEqual(ci.retried_jobs, [])
        (comment,) = github.comments
        self.assertIn("failed 3 times, last in https://buildkite.example/", comment)

    def test_build_pagination_is_bounded(self) -> None:
        full = FakeBuildkite([[rocm_build()] * 100] * 101)

        with self.assertRaises(ApiError):
            list(iter_builds(full, branch="feature"))


if __name__ == "__main__":
    unittest.main()
