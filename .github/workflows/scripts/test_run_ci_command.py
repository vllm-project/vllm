# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import unittest
import urllib.error
from typing import Any
from unittest.mock import patch

from run_ci_command import (
    CI_AUTHORIZED_COMMENT_MARKER,
    COMMAND_RETRY_FAILED,
    COMMAND_RUN_CI,
    COMMAND_RUN_CI_ALL,
    COMMAND_RUN_CI_NIGHTLY,
    RETRY_STATES,
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
    run,
    select_latest_build,
)


def make_pr(**overrides: Any) -> dict[str, Any]:
    pr = {
        "base": {"ref": "main"},
        "draft": False,
        "head": {
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
    ) -> None:
        self.comments = comments or []
        self.permission = permission
        self.permissions = permissions or {}
        self.pr = pr or make_pr()
        self.reactions: list[str] = []
        self.review_decision = review_decision
        self.reviews = reviews or []

    def get_pr(self, number: int) -> dict[str, Any]:
        return self.pr

    def get_permission(self, actor: str) -> str:
        return self.permissions.get(actor, self.permission)

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
        self.retry_calls: list[tuple[int, str]] = []

    def list_builds(
        self,
        commit: str | None,
        *,
        metadata: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        self.list_calls.append((commit, metadata))
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
        self.assertEqual(parse_command(COMMAND_RUN_CI), COMMAND_RUN_CI)
        self.assertEqual(parse_command(COMMAND_RUN_CI_ALL), COMMAND_RUN_CI_ALL)
        self.assertEqual(
            parse_command(COMMAND_RUN_CI_NIGHTLY),
            COMMAND_RUN_CI_NIGHTLY,
        )
        self.assertEqual(
            parse_command(COMMAND_RETRY_FAILED),
            COMMAND_RETRY_FAILED,
        )
        self.assertIsNone(parse_command("/ci run please"))
        self.assertIsNone(parse_command("/ci run all please"))
        self.assertIsNone(parse_command(" /ci run"))

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
                "branch": "feature",
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
        self.assertTrue(github.comments[0].startswith("✅ @author"))
        self.assertIn("`/ci run`", github.comments[0])
        self.assertIn("`/ci retry`", github.comments[0])
        self.assertNotIn(COMMAND_RUN_CI_ALL, github.comments[0])
        self.assertNotIn(COMMAND_RUN_CI_NIGHTLY, github.comments[0])
        self.assertIn(CI_AUTHORIZED_COMMENT_MARKER, github.comments[0])

    def test_ready_label_does_not_notify_after_trusted_approval(self) -> None:
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

        self.assertEqual(github.comments, [])

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
            metadata=("github-pr-number", "42"),
        )

        self.assertEqual(builds, [])
        url = transport.calls[0]["url"]
        self.assertIn("?exclude_jobs=true", url)
        self.assertIn("commit=current-commit", url)
        self.assertIn("meta_data%5Bgithub-pr-number%5D=42", url)

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
