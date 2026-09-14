# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import unittest
import urllib.error
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
    ApiError,
    BuildkiteClient,
    HttpTransport,
    authorize,
    create_build_payload,
    has_trusted_approval,
    is_active_build,
    is_build_for_pr,
    notify_authorized,
    parse_command,
    parse_trusted_users,
    pipeline_for_command,
    resolve_workflow_run_pr,
    run,
    select_latest_build,
)


def make_pr(**overrides: Any) -> dict[str, Any]:
    pr = {
        "base": {"ref": "main"},
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
    ) -> None:
        self.comments = comments or []
        self.permission = permission
        self.permissions = permissions or {}
        self.pr = pr or make_pr()
        self.reactions: list[str] = []
        self.review_decision = review_decision
        self.reviews = reviews or []
        self.pulls_for_commit = pulls_for_commit or []
        self.prs = prs or {self.pr["number"]: self.pr}

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
    ) -> list[dict[str, Any]]:
        self.list_calls.append((commit, metadata))
        self.list_requests.append(
            {
                "branch": branch,
                "commit": commit,
                "metadata": metadata,
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

        self.assertIsNone(parse_command("/ci run please"))
        self.assertIsNone(parse_command("/ci run all please"))
        self.assertIsNone(parse_command("/ci cancel please"))
        self.assertIsNone(parse_command(" /ci run"))
        self.assertIsNone(parse_command("/amd-ci run please"))
        self.assertIsNone(parse_command("/amd-ci retry "))
        self.assertIsNone(parse_command("/AMD-CI run"))
        self.assertIsNone(parse_command("/amdci run"))

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
                "ignore_pipeline_branch_filters": True,
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
        self.assertIn("Queued 3 failed job", github.comments[0])

    def test_amd_ci_retry_retries_only_the_current_head_build(self) -> None:
        github = FakeGitHub(
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

    def test_ci_retry_creates_filtered_build_for_new_head(self) -> None:
        github = FakeGitHub(
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

        self.assertEqual(builds, [])
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


if __name__ == "__main__":
    unittest.main()
