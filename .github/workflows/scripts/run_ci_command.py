# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

COMMAND_RUN_CI = "/ci run"
COMMAND_RETRY_FAILED = "/ci retry"
CI_AUTHORIZED_COMMENT_MARKER = "<!-- vllm-ci-authorized -->"
READY_LABELS = {"ready", "ready-run-all-tests"}
TRUSTED_PERMISSIONS = {"admin", "maintain", "write"}
ACTIVE_BUILD_STATES = {
    "blocked",
    "creating",
    "scheduled",
    "running",
    "failing",
    "canceling",
    "waiting",
    "waiting_failed",
}
RETRY_STATES = "failed,timed_out,expired"
SETUP_STEP_KEYS = {
    "ensure-ci-base-amd",
    "pre-commit",
    "refresh-rocm-base-amd",
}


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


class HttpTransport:
    def request(
        self,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        method: str = "GET",
    ) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            url,
            data=data,
            headers=dict(headers or {}),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode()
        except urllib.error.HTTPError as error:
            response_body = error.read().decode()
            message = self._error_message(response_body, error.reason)
            raise ApiError(
                error.code,
                f"API returned {error.code}: {message}",
            ) from error
        except urllib.error.URLError as error:
            raise ApiError(None, f"API request failed: {error.reason}") from error

        if not response_body:
            return None
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as error:
            raise ApiError(None, "API returned a non-JSON response.") from error

    @staticmethod
    def _error_message(response_body: str, fallback: str) -> str:
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return fallback
        return str(parsed.get("message", fallback))


class GitHubClient:
    def __init__(
        self,
        token: str,
        repository: str,
        transport: HttpTransport | None = None,
    ) -> None:
        if not token:
            raise RuntimeError("GH_TOKEN is not set.")
        self.owner, self.repo = repository.split("/", maxsplit=1)
        self.transport = transport or HttpTransport()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "vllm-ci-command",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(
        self,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        method: str = "GET",
    ) -> Any:
        return self.transport.request(
            f"https://api.github.com{path}",
            body=body,
            headers=self.headers,
            method=method,
        )

    def _repo_path(self, suffix: str) -> str:
        owner = urllib.parse.quote(self.owner, safe="")
        repo = urllib.parse.quote(self.repo, safe="")
        return f"/repos/{owner}/{repo}{suffix}"

    def _paginate(self, path: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            response = self._request(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(response, list):
                raise ApiError(None, "GitHub API returned an invalid list response.")
            results.extend(response)
            if len(response) < 100:
                return results
        raise ApiError(None, "GitHub API pagination exceeded 10,000 results.")

    def get_pr(self, number: int) -> dict[str, Any]:
        return self._request(self._repo_path(f"/pulls/{number}"))

    def get_permission(self, actor: str) -> str:
        username = urllib.parse.quote(actor, safe="")
        try:
            response = self._request(
                self._repo_path(f"/collaborators/{username}/permission")
            )
        except ApiError as error:
            if error.status == 404:
                return "none"
            raise
        return str(response["permission"])

    def get_review_decision(self, number: int) -> str | None:
        query = """
          query($owner: String!, $repo: String!, $number: Int!) {
            repository(owner: $owner, name: $repo) {
              pullRequest(number: $number) {
                reviewDecision
              }
            }
          }
        """
        response = self._request(
            "/graphql",
            body={
                "query": query,
                "variables": {
                    "number": number,
                    "owner": self.owner,
                    "repo": self.repo,
                },
            },
            method="POST",
        )
        return response["data"]["repository"]["pullRequest"]["reviewDecision"]

    def list_reviews(self, number: int) -> list[dict[str, Any]]:
        return self._paginate(self._repo_path(f"/pulls/{number}/reviews"))

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        return self._paginate(self._repo_path(f"/issues/{number}/comments"))

    def list_reactions(self, comment_id: int) -> list[dict[str, Any]]:
        return self._paginate(
            self._repo_path(f"/issues/comments/{comment_id}/reactions")
        )

    def add_reaction(self, comment_id: int, content: str) -> None:
        self._request(
            self._repo_path(f"/issues/comments/{comment_id}/reactions"),
            body={"content": content},
            method="POST",
        )

    def add_comment(self, issue_number: int, body: str) -> None:
        self._request(
            self._repo_path(f"/issues/{issue_number}/comments"),
            body={"body": body},
            method="POST",
        )


class BuildkiteClient:
    def __init__(
        self,
        token: str,
        organization: str,
        pipeline: str,
        transport: HttpTransport | None = None,
    ) -> None:
        self.token = token
        self.transport = transport or HttpTransport()
        organization = urllib.parse.quote(organization, safe="")
        pipeline = urllib.parse.quote(pipeline, safe="")
        self.base_url = (
            "https://api.buildkite.com/v2/organizations/"
            f"{organization}/pipelines/{pipeline}/builds"
        )

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise RuntimeError("The BUILDKITE_API_TOKEN repository secret is not set.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "vllm-ci-command",
        }

    def _request_url(
        self,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        method: str = "GET",
    ) -> Any:
        if (
            url != self.base_url
            and not url.startswith(f"{self.base_url}?")
            and not url.startswith(f"{self.base_url}/")
        ):
            raise ApiError(None, "Buildkite API returned an invalid pagination URL.")
        return self.transport.request(
            url,
            body=body,
            headers=self._headers(),
            method=method,
        )

    def _request(
        self,
        *,
        body: Mapping[str, Any] | None = None,
        method: str = "GET",
        path: str = "",
        query: Sequence[tuple[str, str]] = (),
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        return self._request_url(url, body=body, method=method)

    def list_builds(
        self,
        commit: str | None,
        *,
        metadata: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        query = [
            ("exclude_jobs", "true"),
            ("exclude_pipeline", "true"),
            ("per_page", "100"),
        ]
        if commit:
            query.append(("commit", commit))
        if metadata:
            key, value = metadata
            query.append((f"meta_data[{key}]", value))
        response = self._request(query=query)
        if not isinstance(response, list):
            raise ApiError(None, "Buildkite API returned an invalid build list.")
        return response

    def create_build(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._request(body=body, method="POST")

    def retry_failed_jobs(
        self,
        build_number: int,
        states: str,
    ) -> dict[str, Any]:
        number = urllib.parse.quote(str(build_number), safe="")
        return self._request(
            body={"states": states},
            method="PUT",
            path=f"/{number}/retry_failed_jobs",
        )

    def list_failed_jobs(self, build_number: int) -> list[dict[str, Any]]:
        number = urllib.parse.quote(str(build_number), safe="")
        query = [
            ("state[]", "failed"),
            ("state[]", "timed_out"),
            ("state[]", "expired"),
            ("include_retried_jobs", "false"),
            ("per_page", "100"),
        ]
        url = f"{self.base_url}/{number}/jobs?{urllib.parse.urlencode(query)}"
        jobs: list[dict[str, Any]] = []
        while url:
            response = self._request_url(url)
            if not isinstance(response, Mapping):
                raise ApiError(None, "Buildkite API returned an invalid job list.")
            items = response.get("items")
            links = response.get("links")
            if not isinstance(items, list) or not isinstance(links, Mapping):
                raise ApiError(None, "Buildkite API returned an invalid job list.")
            jobs.extend(items)
            next_url = links.get("next")
            if next_url is not None and not isinstance(next_url, str):
                raise ApiError(
                    None, "Buildkite API returned an invalid pagination URL."
                )
            url = next_url
        return jobs


def parse_command(body: str) -> str | None:
    if body in {COMMAND_RUN_CI, COMMAND_RETRY_FAILED}:
        return body
    return None


def parse_trusted_users(value: str = "") -> set[str]:
    return {
        user.casefold() for item in value.split(",") for user in item.split() if user
    }


def has_ready_label(pr: Mapping[str, Any]) -> bool:
    return any(label["name"] in READY_LABELS for label in pr["labels"])


def is_trusted_permission(permission: str) -> bool:
    return permission in TRUSTED_PERMISSIONS


def authorize(
    *,
    actor: str,
    permission: str,
    pr: Mapping[str, Any],
    trusted_approval: bool = False,
    trusted_users: set[str] | None = None,
) -> tuple[bool, str]:
    trusted_users = trusted_users or set()
    if is_trusted_permission(permission):
        return True, f"repository {permission} permission"
    if actor.casefold() in trusted_users:
        return True, "configured trusted contributor"
    if actor.casefold() != pr["user"]["login"].casefold():
        return (
            False,
            "Only reviewers with write access can run CI before it is "
            "delegated to the PR author.",
        )
    if pr["draft"]:
        return False, "PR authors cannot run CI while the PR is a draft."
    if has_ready_label(pr):
        return True, "ready label"
    if trusted_approval:
        return True, "approval from a trusted reviewer"
    return (
        False,
        "A reviewer with write access must run `/ci run`, approve the PR, "
        "or add the `ready` label first.",
    )


def has_trusted_approval(
    github: GitHubClient,
    number: int,
    trusted_users: set[str],
) -> bool:
    if github.get_review_decision(number) != "APPROVED":
        return False

    latest_review_states: dict[str, tuple[str, str]] = {}
    for review in github.list_reviews(number):
        user = review.get("user") or {}
        login = user.get("login")
        state = review.get("state")
        if login and state in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest_review_states[login.casefold()] = (login, state)

    for login, state in latest_review_states.values():
        if state != "APPROVED":
            continue
        if login.casefold() in trusted_users:
            return True
        if is_trusted_permission(github.get_permission(login)):
            return True
    return False


def is_build_for_pr(build: Mapping[str, Any], pr_number: int) -> bool:
    pull_request = build.get("pull_request")
    if isinstance(pull_request, Mapping):
        build_pr_number = pull_request.get("id", pull_request.get("number"))
        if build_pr_number is not None:
            return str(build_pr_number) == str(pr_number)
    metadata = build.get("meta_data") or {}
    return str(metadata.get("github-pr-number")) == str(pr_number)


def is_active_build(build: Mapping[str, Any]) -> bool:
    return bool(build.get("blocked")) or build.get("state") in ACTIVE_BUILD_STATES


def select_latest_build(
    builds: Sequence[dict[str, Any]],
    pr_number: int,
) -> dict[str, Any] | None:
    matching = [build for build in builds if is_build_for_pr(build, pr_number)]
    return max(matching, key=lambda build: build.get("created_at", ""), default=None)


def create_build_payload(
    *,
    actor: str,
    comment_id: int,
    pr: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "commit": pr["head"]["sha"],
        "branch": pr["head"]["ref"],
        "message": f"PR #{pr['number']} {COMMAND_RUN_CI} by @{actor}",
        "pull_request_id": pr["number"],
        "pull_request_base_branch": pr["base"]["ref"],
        "pull_request_repository": pr["head"]["repo"]["clone_url"],
        "pull_request_labels": [label["name"] for label in pr["labels"]],
        "ignore_pipeline_branch_filters": True,
        "env": {
            "VLLM_CI_GITHUB_COMMENT_ID": str(comment_id),
            "VLLM_CI_TRIGGERED_BY": actor,
        },
        "meta_data": {
            "github-comment-id": str(comment_id),
            "github-pr-number": str(pr["number"]),
            "github-triggered-by": actor,
        },
    }


def create_retry_build_payload(
    *,
    actor: str,
    comment_id: int,
    pr: Mapping[str, Any],
    source_build: Mapping[str, Any],
    step_keys: Sequence[str],
) -> dict[str, Any]:
    payload = create_build_payload(
        actor=actor,
        comment_id=comment_id,
        pr=pr,
    )
    source_number = str(source_build["number"])
    payload["message"] = f"PR #{pr['number']} {COMMAND_RETRY_FAILED} by @{actor}"
    payload["env"]["VLLM_CI_ONLY_STEP_KEYS"] = json.dumps(
        step_keys, separators=(",", ":")
    )
    payload["meta_data"].update(
        {
            "github-retry-source-build": source_number,
            "github-retry-source-commit": str(source_build.get("commit", "")),
        }
    )
    return payload


def add_reaction_safely(
    github: GitHubClient,
    comment_id: int,
    content: str,
) -> None:
    try:
        github.add_reaction(comment_id, content)
    except Exception as error:
        print(f"Could not add {content} reaction: {error}", file=sys.stderr)


def command_comment_marker(comment_id: int) -> str:
    return f"<!-- vllm-ci-command:{comment_id} -->"


def has_bot_comment_marker(
    github: GitHubClient,
    issue_number: int,
    marker: str,
) -> bool:
    return any(
        marker in str(comment.get("body", ""))
        and (comment.get("user") or {}).get("login") == "github-actions[bot]"
        for comment in github.list_issue_comments(issue_number)
    )


def is_already_handled(
    github: GitHubClient,
    issue_number: int,
    comment_id: int,
) -> bool:
    terminal_reaction = any(
        reaction.get("content") in {"rocket", "-1"}
        and (reaction.get("user") or {}).get("login") == "github-actions[bot]"
        for reaction in github.list_reactions(comment_id)
    )
    if terminal_reaction:
        return True
    return has_bot_comment_marker(
        github,
        issue_number,
        command_comment_marker(comment_id),
    )


def notify_authorized(
    event: Mapping[str, Any],
    github: GitHubClient,
    trusted_users_value: str = "",
) -> None:
    pr = event["pull_request"]
    if pr["state"] != "open" or pr["draft"]:
        return

    trusted_users = parse_trusted_users(trusted_users_value)
    author = pr["user"]["login"]
    author_permission = github.get_permission(author)
    if (
        is_trusted_permission(author_permission)
        or author.casefold() in trusted_users
        or has_bot_comment_marker(
            github,
            pr["number"],
            CI_AUTHORIZED_COMMENT_MARKER,
        )
    ):
        return

    if "label" in event:
        if (
            event.get("action") != "labeled"
            or event["label"]["name"] not in READY_LABELS
            or has_trusted_approval(github, pr["number"], trusted_users)
        ):
            return
    elif "review" in event:
        if (
            event.get("action") != "submitted"
            or str(event["review"].get("state", "")).casefold() != "approved"
            or has_ready_label(pr)
            or not has_trusted_approval(github, pr["number"], trusted_users)
        ):
            return
    else:
        return

    github.add_comment(
        pr["number"],
        (
            f"✅ @{author}, CI is now available for this PR. Comment `/ci run` "
            "to run full CI or `/ci retry` to retry failed jobs.\n\n"
            f"{CI_AUTHORIZED_COMMENT_MARKER}"
        ),
    )


def handle_run_ci(
    *,
    actor: str,
    buildkite: BuildkiteClient,
    comment_id: int,
    github: GitHubClient,
    pr: Mapping[str, Any],
) -> str:
    duplicate_builds = buildkite.list_builds(
        pr["head"]["sha"],
        metadata=("github-comment-id", str(comment_id)),
    )
    duplicate = select_latest_build(duplicate_builds, pr["number"])
    if duplicate:
        return f"CI was already requested by this comment: {duplicate['web_url']}"

    current_builds = buildkite.list_builds(pr["head"]["sha"])
    active_build = next(
        (
            build
            for build in current_builds
            if is_build_for_pr(build, pr["number"]) and is_active_build(build)
        ),
        None,
    )
    if active_build:
        return f"CI is already running for this commit: {active_build['web_url']}"

    current_pr = github.get_pr(pr["number"])
    if current_pr["state"] != "open" or current_pr["head"]["sha"] != pr["head"]["sha"]:
        return (
            "The PR head changed while processing the command. Comment `/ci run` again."
        )

    build = buildkite.create_build(
        create_build_payload(
            actor=actor,
            comment_id=comment_id,
            pr=current_pr,
        )
    )
    return (
        f"Triggered [Buildkite CI #{build['number']}]({build['web_url']}) "
        f"for commit `{current_pr['head']['sha'][:12]}`."
    )


def handle_retry_failed(
    *,
    actor: str,
    buildkite: BuildkiteClient,
    comment_id: int,
    github: GitHubClient,
    pr: Mapping[str, Any],
) -> str:
    builds = buildkite.list_builds(pr["head"]["sha"])
    build = select_latest_build(builds, pr["number"])
    if build:
        metadata = build.get("meta_data") or {}
        if str(metadata.get("github-comment-id")) == str(comment_id):
            return f"CI was already requested by this comment: {build['web_url']}"

        retried = buildkite.retry_failed_jobs(build["number"], RETRY_STATES)
        if retried["retried_jobs_count"] == 0:
            return (
                "No failed, timed-out, or expired jobs need retrying: "
                f"{build['web_url']}"
            )
        return (
            f"Queued {retried['retried_jobs_count']} failed job(s) for retry in "
            f"[Buildkite CI #{build['number']}]({build['web_url']})."
        )

    previous_builds = buildkite.list_builds(
        None,
        metadata=("github-pr-number", str(pr["number"])),
    )
    previous_builds = [
        candidate
        for candidate in previous_builds
        if candidate.get("commit") != pr["head"]["sha"]
    ]
    source_build = select_latest_build(previous_builds, pr["number"])
    if not source_build:
        return "No earlier CI build exists for this PR. Use `/ci run` first."
    if not source_build.get("finished_at") or is_active_build(source_build):
        return f"The previous CI build is still running: {source_build['web_url']}"

    failed_jobs = buildkite.list_failed_jobs(source_build["number"])
    failed_script_jobs = [job for job in failed_jobs if job.get("type") == "script"]
    missing_step_keys = [job for job in failed_script_jobs if not job.get("step_key")]
    if missing_step_keys:
        return (
            f"[Buildkite CI #{source_build['number']}]"
            f"({source_build['web_url']}) has failed jobs without stable step "
            "keys, so they cannot be retried on a new commit. Use `/ci run`."
        )

    failed_step_keys = {str(job["step_key"]) for job in failed_script_jobs}
    setup_failures = sorted(
        step_key
        for step_key in failed_step_keys
        if step_key.startswith("image-build") or step_key in SETUP_STEP_KEYS
    )
    if setup_failures:
        return (
            f"[Buildkite CI #{source_build['number']}]"
            f"({source_build['web_url']}) failed during CI setup, so its test "
            "failure set is incomplete. Use `/ci run` for the new commit."
        )

    step_keys = sorted(failed_step_keys)
    if not step_keys:
        return (
            "No failed, timed-out, or expired jobs need retrying in "
            f"[Buildkite CI #{source_build['number']}]"
            f"({source_build['web_url']})."
        )

    current_pr = github.get_pr(pr["number"])
    if current_pr["state"] != "open" or current_pr["head"]["sha"] != pr["head"]["sha"]:
        return (
            "The PR head changed while processing the command. "
            "Comment `/ci retry` again."
        )

    retry_build = buildkite.create_build(
        create_retry_build_payload(
            actor=actor,
            comment_id=comment_id,
            pr=current_pr,
            source_build=source_build,
            step_keys=step_keys,
        )
    )
    return (
        f"Triggered [Buildkite CI #{retry_build['number']}]"
        f"({retry_build['web_url']}) for commit "
        f"`{current_pr['head']['sha'][:12]}`, running {len(step_keys)} failed "
        f"step(s) from [Buildkite CI #{source_build['number']}]"
        f"({source_build['web_url']})."
    )


def run(
    event: Mapping[str, Any],
    github: GitHubClient,
    buildkite: BuildkiteClient,
    trusted_users_value: str = "",
) -> None:
    command = parse_command(event["comment"]["body"])
    if not command or "pull_request" not in event["issue"]:
        return

    issue_number = event["issue"]["number"]
    comment_id = event["comment"]["id"]
    actor = event["comment"]["user"]["login"]

    if is_already_handled(github, issue_number, comment_id):
        print(f"Comment {comment_id} was already handled.")
        return
    add_reaction_safely(github, comment_id, "eyes")

    try:
        pr = github.get_pr(issue_number)
        permission = github.get_permission(actor)
        if pr["state"] != "open":
            github.add_comment(
                issue_number,
                "❌ CI commands require an open PR.\n\n"
                f"{command_comment_marker(comment_id)}",
            )
            return

        trusted_users = parse_trusted_users(trusted_users_value)
        should_check_approval = (
            not is_trusted_permission(permission)
            and actor.casefold() not in trusted_users
            and actor.casefold() == pr["user"]["login"].casefold()
            and not pr["draft"]
            and not has_ready_label(pr)
        )
        trusted_approval = should_check_approval and has_trusted_approval(
            github,
            issue_number,
            trusted_users,
        )
        allowed, reason = authorize(
            actor=actor,
            permission=permission,
            pr=pr,
            trusted_approval=trusted_approval,
            trusted_users=trusted_users,
        )
        if not allowed:
            github.add_comment(
                issue_number,
                f"❌ @{actor}, {reason}\n\n{command_comment_marker(comment_id)}",
            )
            return

        print(f"Authorized @{actor}: {reason}")
        if command == COMMAND_RUN_CI:
            message = handle_run_ci(
                actor=actor,
                buildkite=buildkite,
                comment_id=comment_id,
                github=github,
                pr=pr,
            )
        else:
            message = handle_retry_failed(
                actor=actor,
                buildkite=buildkite,
                comment_id=comment_id,
                github=github,
                pr=pr,
            )
        add_reaction_safely(github, comment_id, "rocket")
        github.add_comment(issue_number, f"✅ {message}")
    except Exception:
        add_reaction_safely(github, comment_id, "confused")
        raise


def main() -> None:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, encoding="utf-8") as event_file:
        event = json.load(event_file)

    github = GitHubClient(
        os.environ.get("GH_TOKEN", ""),
        os.environ["GITHUB_REPOSITORY"],
    )
    event_name = os.environ.get("GITHUB_EVENT_NAME", "issue_comment")
    if event_name == "pull_request_target":
        notify_authorized(
            event,
            github,
            os.environ.get("CI_TRUSTED_USERS", ""),
        )
        return
    if event_name == "workflow_run":
        pull_requests = event["workflow_run"].get("pull_requests") or []
        if not pull_requests:
            return
        pr = github.get_pr(int(pull_requests[0]["number"]))
        notify_authorized(
            {
                "action": "submitted",
                "pull_request": pr,
                "review": {"state": "approved"},
            },
            github,
            os.environ.get("CI_TRUSTED_USERS", ""),
        )
        return

    if not parse_command(event["comment"]["body"]):
        return

    buildkite = BuildkiteClient(
        os.environ.get("BUILDKITE_API_TOKEN", ""),
        os.environ.get("BUILDKITE_ORGANIZATION", "vllm"),
        os.environ.get("BUILDKITE_PIPELINE", "ci"),
    )
    run(
        event,
        github,
        buildkite,
        os.environ.get("CI_TRUSTED_USERS", ""),
    )


if __name__ == "__main__":
    main()
