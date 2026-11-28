# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import binascii
import http.client
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any

COMMAND_RUN_CI = "/ci run"
COMMAND_RUN_CI_ALL = "/ci run all"
COMMAND_RUN_CI_NIGHTLY = "/ci run nightly"
COMMAND_RETRY_FAILED = "/ci retry"
COMMAND_CANCEL_CI = "/ci cancel"
COMMAND_RUN_AMD_CI = "/amd-ci run"
COMMAND_RUN_AMD_CI_ALL = "/amd-ci run all"
COMMAND_RUN_AMD_CI_NIGHTLY = "/amd-ci run nightly"
COMMAND_RETRY_AMD_FAILED = "/amd-ci retry"
COMMAND_CANCEL_AMD_CI = "/amd-ci cancel"
ALLOW_STALE_SUFFIX = " --allow-stale"
RUN_CI_COMMAND_ENV = {
    COMMAND_RUN_CI: {},
    COMMAND_RUN_CI_ALL: {"RUN_ALL": "1"},
    COMMAND_RUN_CI_NIGHTLY: {"RUN_ALL": "1", "NIGHTLY": "1"},
    COMMAND_RUN_AMD_CI: {},
    COMMAND_RUN_AMD_CI_ALL: {"RUN_ALL": "1"},
    COMMAND_RUN_AMD_CI_NIGHTLY: {"RUN_ALL": "1", "NIGHTLY": "1"},
}
ALLOW_STALE_COMMANDS = frozenset(
    f"{command}{ALLOW_STALE_SUFFIX}" for command in RUN_CI_COMMAND_ENV
)
RUN_CI_COMMAND_ENV.update(
    {
        command: RUN_CI_COMMAND_ENV[command.removesuffix(ALLOW_STALE_SUFFIX)]
        for command in ALLOW_STALE_COMMANDS
    }
)
UPSTREAM_CI_COMMANDS = frozenset(
    {
        *(command for command in RUN_CI_COMMAND_ENV if command.startswith("/ci ")),
        COMMAND_RETRY_FAILED,
        COMMAND_CANCEL_CI,
    }
)
AMD_CI_COMMANDS = frozenset(
    {
        *(command for command in RUN_CI_COMMAND_ENV if command.startswith("/amd-ci ")),
        COMMAND_RETRY_AMD_FAILED,
        COMMAND_CANCEL_AMD_CI,
    }
)
RETRY_COMMANDS = frozenset({COMMAND_RETRY_FAILED, COMMAND_RETRY_AMD_FAILED})
CANCEL_COMMANDS = frozenset({COMMAND_CANCEL_CI, COMMAND_CANCEL_AMD_CI})
ALL_CI_COMMANDS = UPSTREAM_CI_COMMANDS | AMD_CI_COMMANDS
CI_COMMAND_PREFIXES = frozenset({"/ci", "/amd-ci"})
CI_AUTHORIZED_COMMENT_MARKER = "<!-- vllm-ci-authorized -->"
READY_LABELS = {"ready", "ready-run-all-tests"}
TRUSTED_PERMISSIONS = {"admin", "maintain", "write"}
BRANCH_UPDATE_POLL_SECONDS = 3
BRANCH_UPDATE_POLL_ATTEMPTS = 40
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
CANCELABLE_BUILD_STATES = ("scheduled", "running", "failing")
# ensure-ci-base-amd: the old ROCm build chain, which ci-infra still renders
# for branches that predate .buildkite/scripts/rocm/build.sh.
ROCM_REFRESH_STEP_KEY = "refresh-rocm-base-amd"
SETUP_STEP_KEYS = {"ensure-ci-base-amd", ROCM_REFRESH_STEP_KEY, "pre-commit"}
# ci-infra adds it for the sharded steps a build runs; it is not selectable.
SHARD_COVERAGE_STEP_KEY = "runtime-shard-coverage-check"
# A PR to main that changes these builds a new ROCm base image. Its
# rocm-image-cache-check step fails until reconcile_rocm_cache caches the image
# a trusted user's CI build pushed, so main reuses it after the merge.
ROCM_STACK_INPUTS = (
    "docker/Dockerfile.rocm_base",
    "tools/vllm-rocm/amdsmi/amdsmi_wrapper.patch",
    "tools/vllm-rocm/amdsmi/pyproject.toml",
    "tools/vllm-rocm/therock_wheels.py",
)
ROCM_CACHE_YAML = ".buildkite/image_build/amd_rocm_cache.yaml"
ROCM_CACHE_CHECK_KEY = "rocm-image-cache-check"
ROCM_PROMOTE_KEY = "rocm-promote"
CODEOWNERS_PATH = ".github/CODEOWNERS"
# build.sh check: main changed the stack since the PR's merge base.
ROCM_CHECK_MAIN_MOVED = 4
ROCM_BUILD_SH = ".buildkite/scripts/rocm/build.sh"
# build.sh check: the PR's stack was in main's cache when the check passed.
ROCM_CACHED_KEY = "rocm-stack-cached"
# The build.sh line ranges that make the stack key (build.sh check's keyfmt). A PR
# that changes them is not cached: promote keys the stack with main's build.sh.
ROCM_KEY_RANGES = (("case $STACK in", "esac"), ("_trw=", "  Kr="))
# build.sh base mode records the images it pushed itself under these keys, not
# ones it reused from a tag another PR's CI could have pushed, and a runtime or
# test-deps only with the parent it was built on ("none" otherwise). Only a build
# that pushed the stack is a source; build.sh promote copies the rest over it.
ROCM_STACK_BUILT_KEY = "rocm-rocm-stack-built"
ROCM_BUILT_KEYS = (
    ROCM_STACK_BUILT_KEY,
    "rocm-rocm-runtime-built",
    "rocm-rocm-test-deps-built",
)
# The one step a cache build runs (.buildkite/image_build/amd_rocm_cache.yaml).
ROCM_CACHE_STEP_KEY = "rocm-base-image-cache"
# Promote builds per source build before the bot asks a maintainer to look.
ROCM_PROMOTE_ATTEMPTS = 3
REVIEW_VERDICTS = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
BUILDS_PER_PAGE = 100
MAX_BUILD_PAGES = 100


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


class CiPreparationError(RuntimeError):
    pass


def rate_limit_jitter() -> float:
    return random.uniform(1, 5)


class HttpTransport:
    def __init__(
        self,
        *,
        max_retries: int = 3,
        jitter: Callable[[], float] = rate_limit_jitter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries = max_retries
        self.jitter = jitter
        self.sleep = sleep

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
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    response_body = response.read().decode()
                break
            except urllib.error.HTTPError as error:
                response_body = error.read().decode()
                if error.code == 429 and attempt < self.max_retries:
                    delay = self._rate_limit_delay(error, response_body)
                    print(
                        "API rate limit reached; "
                        f"retry {attempt + 1}/{self.max_retries} "
                        f"in {delay:g} seconds.",
                        file=sys.stderr,
                    )
                    self.sleep(delay)
                    continue
                message = self._error_message(response_body, error.reason)
                raise ApiError(
                    error.code,
                    f"API returned {error.code}: {message}",
                ) from error
            except urllib.error.URLError as error:
                raise ApiError(
                    None,
                    f"API request failed: {error.reason}",
                ) from error
            except (OSError, http.client.HTTPException) as error:
                raise ApiError(None, f"API request failed: {error}") from error

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

    def _rate_limit_delay(
        self,
        error: urllib.error.HTTPError,
        response_body: str,
    ) -> float:
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = {}

        scope = parsed.get("scope")
        reset_header = (
            "RateLimit-User-Reset" if scope == "rest_user" else "RateLimit-Reset"
        )
        candidates = [
            error.headers.get(reset_header),
            parsed.get("reset"),
        ]
        for candidate in candidates:
            try:
                delay = float(candidate)
            except (TypeError, ValueError):
                continue
            if delay >= 0:
                return delay + self.jitter()
        return 60 + self.jitter()


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

    def get_commits_behind_base(self, base_ref: str, head_sha: str) -> int:
        base_ref = urllib.parse.quote(f"refs/heads/{base_ref}", safe="")
        head_sha = urllib.parse.quote(head_sha, safe="")
        response = self._request(
            self._repo_path(f"/compare/{base_ref}...{head_sha}?per_page=1")
        )
        behind = response.get("behind_by") if isinstance(response, dict) else None
        if type(behind) is not int or behind < 0:
            raise ApiError(None, "GitHub returned an invalid commits-behind count.")
        return behind

    def update_branch(self, number: int, expected_head_sha: str) -> None:
        self._request(
            self._repo_path(f"/pulls/{number}/update-branch"),
            body={"expected_head_sha": expected_head_sha},
            method="PUT",
        )

    def list_pulls_for_commit(self, commit: str) -> list[dict[str, Any]]:
        commit = urllib.parse.quote(commit, safe="")
        response = self._request(self._repo_path(f"/commits/{commit}/pulls"))
        if not isinstance(response, list):
            raise ApiError(None, "GitHub API returned an invalid pull request list.")
        return response

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

    def list_pr_files(self, number: int) -> list[str]:
        files = self._paginate(self._repo_path(f"/pulls/{number}/files"))
        return [
            str(path)
            for file in files
            for path in (file["filename"], file.get("previous_filename"))
            if path
        ]

    def get_file_sha(self, path: str, ref: str) -> str:
        path, ref = urllib.parse.quote(path), urllib.parse.quote(ref, safe="")
        try:
            return self._request(self._repo_path(f"/contents/{path}?ref={ref}"))["sha"]
        except ApiError as error:
            if error.status == 404:
                return "none"
            raise

    def get_file_text(self, path: str, ref: str) -> str:
        path, ref = urllib.parse.quote(path), urllib.parse.quote(ref, safe="")
        try:
            response = self._request(self._repo_path(f"/contents/{path}?ref={ref}"))
        except ApiError as error:
            if error.status == 404:
                return ""
            raise
        return binascii.a2b_base64(response["content"]).decode()

    def get_merge_base(self, base_ref: str, head_sha: str) -> str:
        base_ref = urllib.parse.quote(f"refs/heads/{base_ref}", safe="")
        head_sha = urllib.parse.quote(head_sha, safe="")
        response = self._request(
            self._repo_path(f"/compare/{base_ref}...{head_sha}?per_page=1")
        )
        return str(response["merge_base_commit"]["sha"])


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
        branch: str | None = None,
        metadata: tuple[str, str] | None = None,
        states: Sequence[str] = (),
        page: int = 1,
        created_from: str | None = None,
    ) -> list[dict[str, Any]]:
        query = [
            ("exclude_jobs", "true"),
            ("exclude_pipeline", "true"),
            ("per_page", str(BUILDS_PER_PAGE)),
        ]
        if page > 1:
            query.append(("page", str(page)))
        if created_from:
            query.append(("created_from", created_from))
        if commit:
            query.append(("commit", commit))
        if branch:
            query.append(("branch", branch))
        if metadata:
            key, value = metadata
            query.append((f"meta_data[{key}]", value))
        query.extend(("state[]", state) for state in states)
        response = self._request(query=query)
        if not isinstance(response, list):
            raise ApiError(None, "Buildkite API returned an invalid build list.")
        return response

    def get_build(self, build_number: int) -> dict[str, Any]:
        number = urllib.parse.quote(str(build_number), safe="")
        return self._request(path=f"/{number}")

    def create_build(self, body: Mapping[str, Any]) -> dict[str, Any]:
        # Bot builds are explicit commands. Pipeline branch settings gate
        # webhooks and would otherwise reject them, e.g. amd-ci on main.
        body = {**body, "ignore_pipeline_branch_filters": True}
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

    def cancel_build(self, build_number: int) -> dict[str, Any]:
        number = urllib.parse.quote(str(build_number), safe="")
        return self._request(method="PUT", path=f"/{number}/cancel")

    def retry_job(self, build_number: int, job_id: str) -> dict[str, Any]:
        number = urllib.parse.quote(str(build_number), safe="")
        job = urllib.parse.quote(job_id, safe="")
        return self._request(method="PUT", path=f"/{number}/jobs/{job}/retry")

    def list_failed_jobs(self, build_number: int) -> list[dict[str, Any]]:
        return self.list_jobs(build_number, states=RETRY_STATES.split(","))

    def list_jobs(
        self,
        build_number: int,
        *,
        states: Sequence[str] = (),
        step_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """List a build's jobs, without the ones that were retried."""
        number = urllib.parse.quote(str(build_number), safe="")
        query = [("state[]", state) for state in states]
        if step_key:
            query.append(("step_key", step_key))
        query += [("include_retried_jobs", "false"), ("per_page", "100")]
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
    body = body.strip()
    if body in ALL_CI_COMMANDS:
        return body
    return None


def is_ci_command_attempt(body: str) -> bool:
    words = body.split(maxsplit=1)
    return bool(words) and words[0].casefold() in CI_COMMAND_PREFIXES


def pipeline_for_command(
    command: str,
    *,
    amd_ci_pipeline: str = "amd-ci",
    upstream_ci_pipeline: str = "ci",
) -> str:
    if command in AMD_CI_COMMANDS:
        return amd_ci_pipeline
    if command in UPSTREAM_CI_COMMANDS:
        return upstream_ci_pipeline
    raise ValueError(f"Unsupported CI command: {command}")


def ci_name_for_command(command: str) -> str:
    if command in AMD_CI_COMMANDS:
        return "AMD CI"
    if command in UPSTREAM_CI_COMMANDS:
        return "CI"
    raise ValueError(f"Unsupported CI command: {command}")


def run_command_for_command(command: str) -> str:
    if command in AMD_CI_COMMANDS:
        return COMMAND_RUN_AMD_CI
    if command in UPSTREAM_CI_COMMANDS:
        return COMMAND_RUN_CI
    raise ValueError(f"Unsupported CI command: {command}")


def retry_command_for_command(command: str) -> str:
    if command in AMD_CI_COMMANDS:
        return COMMAND_RETRY_AMD_FAILED
    if command in UPSTREAM_CI_COMMANDS:
        return COMMAND_RETRY_FAILED
    raise ValueError(f"Unsupported CI command: {command}")


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
    run_command: str = COMMAND_RUN_CI,
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
            (
                "Only reviewers with write access can use CI commands before CI is "
                "delegated to the PR author."
            ),
        )
    if pr["draft"]:
        return False, "PR authors cannot run CI while the PR is a draft."
    if has_ready_label(pr):
        return True, "ready label"
    if trusted_approval:
        return True, "approval from a trusted reviewer"
    return (
        False,
        (
            f"A reviewer with write access must run `{run_command}`, approve the PR, "
            "or add the `ready` label first."
        ),
    )


def is_trusted_user(github: GitHubClient, login: str, trusted_users: set[str]) -> bool:
    return login.casefold() in trusted_users or is_trusted_permission(
        github.get_permission(login)
    )


def trusted_approved_commits(
    github: GitHubClient, number: int, trusted_users: set[str]
) -> set[str]:
    """Return the commits that trusted reviewers' latest reviews approved."""
    latest_reviews: dict[str, Mapping[str, Any]] = {}
    for review in github.list_reviews(number):
        login = (review.get("user") or {}).get("login")
        if login and review.get("state") in REVIEW_VERDICTS:
            latest_reviews[login.casefold()] = review
    return {
        str(review.get("commit_id"))
        for review in latest_reviews.values()
        if review["state"] == "APPROVED"
        and is_trusted_user(github, review["user"]["login"], trusted_users)
    }


def has_trusted_approval(
    github: GitHubClient,
    number: int,
    trusted_users: set[str],
) -> bool:
    return github.get_review_decision(number) == "APPROVED" and bool(
        trusted_approved_commits(github, number, trusted_users)
    )


def iter_builds(
    buildkite: Any,
    *,
    branch: str | None = None,
    metadata: tuple[str, str] | None = None,
    created_from: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield builds across every page, newest first."""
    for page in range(1, MAX_BUILD_PAGES + 1):
        builds = buildkite.list_builds(
            None,
            branch=branch,
            metadata=metadata,
            page=page,
            created_from=created_from,
        )
        yield from builds
        if len(builds) < BUILDS_PER_PAGE:
            return
    raise ApiError(None, "Buildkite API pagination exceeded 10,000 builds.")


def is_build_for_pr(build: Mapping[str, Any], pr_number: int) -> bool:
    pull_request = build.get("pull_request")
    if isinstance(pull_request, Mapping):
        build_pr_number = pull_request.get("id", pull_request.get("number"))
        if build_pr_number is not None:
            return str(build_pr_number) == str(pr_number)
    metadata = build.get("meta_data") or {}
    return str(metadata.get("github-pr-number")) == str(pr_number)


def is_active_build(build: Mapping[str, Any]) -> bool:
    state = build.get("state")
    if state == "blocked" or build.get("blocked"):
        # A build stays blocked after every job has finished while an optional
        # block or input step (the "Run more tests" picker) waits for a click,
        # which may never come. Only a block that holds back unfinished work
        # keeps the build active.
        if build.get("finished_at"):
            return state in ACTIVE_BUILD_STATES - {"blocked"}
        return True
    return state in ACTIVE_BUILD_STATES


def is_comment_triggered_build(build: Mapping[str, Any]) -> bool:
    metadata = build.get("meta_data") or {}
    return bool(metadata.get("github-comment-id"))


def blocks_new_run(command: str, build: Mapping[str, Any]) -> bool:
    if command in AMD_CI_COMMANDS:
        return (
            is_comment_triggered_build(build)
            and build.get("state") in ACTIVE_BUILD_STATES
            and build.get("state") != "blocked"
        )
    return is_active_build(build)


def list_commit_builds(
    buildkite: BuildkiteClient, command: str, commit: str
) -> list[dict[str, Any]]:
    builds = buildkite.list_builds(commit)
    if command not in AMD_CI_COMMANDS:
        return builds
    # bootstrap-amd.sh exits before uploading any test steps for PR webhook builds.
    return [build for build in builds if build.get("source") != "webhook"]


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
    command: str = COMMAND_RUN_CI,
    pr: Mapping[str, Any],
) -> dict[str, Any]:
    if command not in RUN_CI_COMMAND_ENV:
        raise ValueError(f"Unsupported run command: {command}")

    env = {
        "VLLM_CI_GITHUB_COMMENT_ID": str(comment_id),
        "VLLM_CI_TRIGGERED_BY": actor,
        **RUN_CI_COMMAND_ENV[command],
    }
    return {
        "commit": pr["head"]["sha"],
        # Use the head label (owner:branch) rather than the bare ref so a PR
        # whose fork branch is named e.g. "main" does not collide with the
        # upstream branch of the same name.
        "branch": pr["head"].get("label") or pr["head"]["ref"],
        "message": f"PR #{pr['number']} {command} by @{actor}",
        "pull_request_id": pr["number"],
        "pull_request_base_branch": pr["base"]["ref"],
        "pull_request_repository": pr["head"]["repo"]["clone_url"],
        "pull_request_labels": [label["name"] for label in pr["labels"]],
        "env": env,
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
    command: str = COMMAND_RETRY_FAILED,
    pr: Mapping[str, Any],
    source_build: Mapping[str, Any],
    step_keys: Sequence[str],
) -> dict[str, Any]:
    if command not in RETRY_COMMANDS:
        raise ValueError(f"Unsupported retry command: {command}")
    payload = create_build_payload(
        actor=actor,
        comment_id=comment_id,
        command=run_command_for_command(command),
        pr=pr,
    )
    source_number = str(source_build["number"])
    payload["message"] = f"PR #{pr['number']} {command} by @{actor}"
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
            f"✅ @{author}, CI is now available for this PR.\n\n"
            "- `/ci run` starts upstream CI; `/amd-ci run` starts AMD CI only.\n"
            "- Run commands test your branch with every commit currently on its "
            "upstream target branch. If it is behind, the bot merges the target "
            "branch into it first (pull before pushing again). Append "
            "`--allow-stale` to a run command to test an outdated branch as is, "
            "at your own risk.\n"
            "- `/ci retry` retries failed jobs in the CI build for the current "
            "PR head. If the current head has no CI build, it starts a new CI "
            "build for the current head containing only jobs that failed in "
            "the latest earlier CI build for this PR.\n"
            "- `/amd-ci retry` retries failed jobs in AMD CI for the current PR "
            "head. Use `/amd-ci run` when the current head has no AMD CI build.\n"
            "- `/ci cancel` cancels scheduled or running CI builds for this PR "
            "branch; `/amd-ci cancel` does the same for AMD CI only.\n\n"
            f"{CI_AUTHORIZED_COMMENT_MARKER}"
        ),
    )


def resolve_workflow_run_pr(
    workflow_run: Mapping[str, Any],
    github: GitHubClient,
) -> dict[str, Any] | None:
    head_sha = str(workflow_run.get("head_sha", ""))
    if not head_sha:
        return None

    seen: set[int] = set()

    def find_matching_pr(
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        for candidate in candidates:
            candidate_number = candidate.get("number")
            if not isinstance(candidate_number, int) or candidate_number in seen:
                continue
            seen.add(candidate_number)
            try:
                pr = github.get_pr(candidate_number)
            except ApiError as error:
                if error.status == 404:
                    continue
                raise
            if pr["state"] == "open" and pr["head"]["sha"] == head_sha:
                return pr
        return None

    associated_pr = find_matching_pr(workflow_run.get("pull_requests") or [])
    if associated_pr is not None:
        return associated_pr
    return find_matching_pr(github.list_pulls_for_commit(head_sha))


def merge_base_into_pr(
    github: GitHubClient,
    branch_updater: GitHubClient,
    pr: Mapping[str, Any],
    command: str,
    lag: str,
) -> tuple[dict[str, Any], str]:
    base_ref = pr["base"]["ref"]

    def merge_failed(reason: str) -> CiPreparationError:
        return CiPreparationError(
            f"{lag} The bot could not merge upstream `{base_ref}` into this "
            f"branch ({reason}). No new CI build was started. Either merge or "
            f"rebase onto the latest `{base_ref}` yourself and rerun `{command}`, "
            f"or use `{command}{ALLOW_STALE_SUFFIX}` to test this branch as is."
        )

    try:
        branch_updater.update_branch(pr["number"], pr["head"]["sha"])
        for _ in range(BRANCH_UPDATE_POLL_ATTEMPTS):
            time.sleep(BRANCH_UPDATE_POLL_SECONDS)
            current_pr = github.get_pr(pr["number"])
            if current_pr["head"]["sha"] == pr["head"]["sha"]:
                continue
            try:
                behind = github.get_commits_behind_base(
                    base_ref, current_pr["head"]["sha"]
                )
            except ApiError as error:
                # A fork's new head is briefly unknown to the base repository.
                if error.status == 404:
                    continue
                raise
            break
        else:
            raise merge_failed("the merge commit did not appear in time")
        if (
            current_pr["state"] != "open"
            or current_pr["base"]["ref"] != base_ref
            or behind
        ):
            raise merge_failed("the PR changed during the merge")
    except ApiError as error:
        raise merge_failed(str(error)) from error

    note = (
        f"ℹ️ {lag} Merged upstream `{base_ref}` into this branch as "
        f"`{current_pr['head']['sha'][:12]}` before starting CI. "
        "Pull the branch before pushing more changes."
    )
    print(note)
    return current_pr, note


def prepare_pr_for_ci(
    github: GitHubClient,
    pr: Mapping[str, Any],
    command: str,
    branch_updater: GitHubClient | None = None,
) -> tuple[dict[str, Any], str]:
    base_ref = pr["base"]["ref"]
    no_action = "No new CI build was started."
    try:
        behind = github.get_commits_behind_base(base_ref, pr["head"]["sha"])
        current_pr = github.get_pr(pr["number"])
    except ApiError as error:
        raise CiPreparationError(
            f"Could not check the PR against upstream `{base_ref}`. "
            f"{no_action} Comment `{command}` again. {error}"
        ) from error
    if (
        current_pr["state"] != "open"
        or current_pr["head"]["sha"] != pr["head"]["sha"]
        or current_pr["base"]["ref"] != pr["base"]["ref"]
    ):
        raise CiPreparationError(
            f"The PR changed while checking its branch. {no_action} "
            f"Comment `{command}` again."
        )
    commits = "commit" if behind == 1 else "commits"
    lag = f"This PR is {behind} {commits} behind upstream `{base_ref}`."
    if command in ALLOW_STALE_COMMANDS:
        warning = ""
        if behind:
            run_command = command.removesuffix(ALLOW_STALE_SUFFIX)
            warning = (
                f"⚠️ {lag} Running CI at your own risk because "
                "`--allow-stale` was requested; outdated CI configuration may "
                "cause failures. Before merging, merge or rebase onto the latest "
                f"`{base_ref}`, then rerun `{run_command}` on the latest PR commit."
            )
            print(warning)
        return current_pr, warning

    if behind and branch_updater is not None:
        return merge_base_into_pr(github, branch_updater, current_pr, command, lag)
    if behind:
        raise CiPreparationError(
            f"{lag} Your branch must contain every commit currently on upstream "
            f"`{base_ref}`. {no_action} "
            f"Merge or rebase onto the latest `{base_ref}`, then rerun `{command}`."
            " To test this branch at your own risk, "
            f"use `{command}{ALLOW_STALE_SUFFIX}`."
        )
    return current_pr, ""


def handle_run_ci(
    *,
    actor: str,
    buildkite: BuildkiteClient,
    comment_id: int,
    command: str,
    github: GitHubClient,
    pr: Mapping[str, Any],
    branch_updater: GitHubClient | None = None,
    allow_repeat_run: bool = False,
) -> str:
    ci_name = ci_name_for_command(command)
    duplicate_builds = buildkite.list_builds(
        pr["head"]["sha"],
        metadata=("github-comment-id", str(comment_id)),
    )
    duplicate = select_latest_build(duplicate_builds, pr["number"])
    if duplicate:
        return (
            f"{ci_name} was already requested by this comment: {duplicate['web_url']}"
        )

    current_builds = list_commit_builds(buildkite, command, pr["head"]["sha"])
    active_build = next(
        (
            build
            for build in current_builds
            if is_build_for_pr(build, pr["number"]) and blocks_new_run(command, build)
        ),
        None,
    )
    if active_build:
        return (
            f"{ci_name} is already running for this commit: {active_build['web_url']}"
        )

    tested_build = select_latest_build(
        [build for build in current_builds if build.get("state") != "canceled"],
        pr["number"],
    )
    if tested_build and not allow_repeat_run:
        raise CiPreparationError(
            f"{ci_name} already ran on this commit: {tested_build['web_url']}. "
            "No new CI build was started. Push new changes to run CI again, use "
            f"`{retry_command_for_command(command)}` to rerun failed jobs, or ask "
            "a reviewer with write access to rerun it."
        )

    current_pr, warning = prepare_pr_for_ci(github, pr, command, branch_updater)

    build = buildkite.create_build(
        create_build_payload(
            actor=actor,
            comment_id=comment_id,
            command=command,
            pr=current_pr,
        )
    )
    return (
        f"Triggered [Buildkite {ci_name} #{build['number']}]({build['web_url']}) "
        f"for commit `{current_pr['head']['sha'][:12]}`."
        + (f"\n\n{warning}" if warning else "")
    )


def handle_retry_failed(
    *,
    actor: str,
    buildkite: BuildkiteClient,
    comment_id: int,
    command: str,
    github: GitHubClient,
    pr: Mapping[str, Any],
) -> str:
    ci_name = ci_name_for_command(command)
    run_command = run_command_for_command(command)
    retry_command = retry_command_for_command(command)
    builds = list_commit_builds(buildkite, command, pr["head"]["sha"])
    build = select_latest_build(builds, pr["number"])
    if build:
        metadata = build.get("meta_data") or {}
        if str(metadata.get("github-comment-id")) == str(comment_id):
            return (
                f"{ci_name} was already requested by this comment: {build['web_url']}"
            )

        retried = buildkite.retry_failed_jobs(build["number"], RETRY_STATES)
        if retried["retried_jobs_count"] == 0:
            return (
                "No failed, timed-out, or expired jobs need retrying: "
                f"{build['web_url']}"
            )
        return (
            f"Queued {retried['retried_jobs_count']} failed job(s) for retry in "
            f"[Buildkite {ci_name} #{build['number']}]({build['web_url']})."
        )

    if command == COMMAND_RETRY_AMD_FAILED:
        return (
            "No AMD CI build exists for the current PR head. "
            f"Use `{run_command}` first."
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
        return (
            f"No earlier {ci_name} build exists for this PR. Use `{run_command}` first."
        )
    if not source_build.get("finished_at") or is_active_build(source_build):
        return (
            f"The previous {ci_name} build is still running: {source_build['web_url']}"
        )

    failed_jobs = buildkite.list_failed_jobs(source_build["number"])
    # A soft-failed job never blocks the PR, so it is neither a setup failure
    # nor worth re-running.
    failed_script_jobs = [
        job
        for job in failed_jobs
        if job.get("type") == "script" and not job.get("soft_failed")
    ]
    missing_step_keys = [job for job in failed_script_jobs if not job.get("step_key")]
    if missing_step_keys:
        return (
            f"[Buildkite {ci_name} #{source_build['number']}]"
            f"({source_build['web_url']}) has failed jobs without stable step "
            "keys, so they cannot be retried on a new commit. "
            f"Use `{run_command}`."
        )

    failed_step_keys = {str(job["step_key"]) for job in failed_script_jobs}
    failed_step_keys.discard(SHARD_COVERAGE_STEP_KEY)
    setup_failures = sorted(
        step_key
        for step_key in failed_step_keys
        if step_key.startswith("image-build") or step_key in SETUP_STEP_KEYS
    )
    if setup_failures:
        return (
            f"[Buildkite {ci_name} #{source_build['number']}]"
            f"({source_build['web_url']}) failed during CI setup, so its test "
            f"failure set is incomplete. Use `{run_command}` for the new commit."
        )

    step_keys = sorted(failed_step_keys)
    if not step_keys:
        return (
            "No failed, timed-out, or expired jobs need retrying in "
            f"[Buildkite {ci_name} #{source_build['number']}]"
            f"({source_build['web_url']})."
        )

    current_pr = github.get_pr(pr["number"])
    if current_pr["state"] != "open" or current_pr["head"]["sha"] != pr["head"]["sha"]:
        return (
            "The PR head changed while processing the command. "
            f"Comment `{retry_command}` again."
        )

    failed_count = len(step_keys)
    # Branches from before the check existed fail on an unknown step key.
    try:
        if (
            pr["base"]["ref"] == "main"
            and set(ROCM_STACK_INPUTS) & set(github.list_pr_files(pr["number"]))
            and defines_step_key(
                github.get_file_text(ROCM_CACHE_YAML, pr["head"]["sha"]),
                ROCM_CACHE_CHECK_KEY,
            )
        ):
            step_keys = sorted({*step_keys, ROCM_CACHE_CHECK_KEY})
    except ApiError as error:
        print(f"Not adding {ROCM_CACHE_CHECK_KEY}: {error}")

    retry_build = buildkite.create_build(
        create_retry_build_payload(
            actor=actor,
            comment_id=comment_id,
            command=command,
            pr=current_pr,
            source_build=source_build,
            step_keys=step_keys,
        )
    )
    return (
        f"Triggered [Buildkite {ci_name} #{retry_build['number']}]"
        f"({retry_build['web_url']}) for commit "
        f"`{current_pr['head']['sha'][:12]}`, running {failed_count} failed "
        f"step(s) from [Buildkite {ci_name} #{source_build['number']}]"
        f"({source_build['web_url']})."
    )


def handle_cancel_ci(
    *,
    buildkite: BuildkiteClient,
    command: str = COMMAND_CANCEL_CI,
    pr: Mapping[str, Any],
) -> str:
    ci_name = ci_name_for_command(command)
    branch = pr["head"]["ref"]
    # Builds are created under the head label (owner:branch); also match the
    # bare ref so builds created before that change can still be cancelled.
    branches = [branch]
    head_label = pr["head"].get("label")
    if head_label and head_label not in branches:
        branches.insert(0, str(head_label))
    states = CANCELABLE_BUILD_STATES

    builds_by_number: dict[int, dict[str, Any]] = {}
    for candidate_branch in branches:
        for build in buildkite.list_builds(
            None,
            branch=candidate_branch,
            states=states,
        ):
            builds_by_number[build["number"]] = build

    cancelable_builds = [
        build
        for build in builds_by_number.values()
        if build.get("branch") in branches
        and is_build_for_pr(build, pr["number"])
        and build.get("state") in states
    ]
    if not cancelable_builds:
        return f"No cancelable {ci_name} build is running for branch `{branch}`."

    for build in cancelable_builds:
        buildkite.cancel_build(build["number"])

    links = ", ".join(
        f"[#{build['number']}]({build['web_url']})" for build in cancelable_builds
    )
    count = len(cancelable_builds)
    noun = "build" if count == 1 else "builds"
    return (
        f"Requested cancellation of {count} {ci_name} {noun} for `{branch}`: {links}."
    )


def codeowners_for(codeowners: str, paths: Sequence[str]) -> list[str]:
    """Return the owners of paths, where the last matching CODEOWNERS line wins."""
    rules = [
        line.split()
        for line in codeowners.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    found: dict[str, str] = {}
    for path in paths:
        matches = [
            owners
            for pattern, *owners in rules
            if fnmatchcase(path, pattern.lstrip("/"))
            or path.startswith(pattern.strip("/") + "/")
        ]
        for owner in matches[-1] if matches else []:
            found.setdefault(owner.casefold(), owner)
    return list(found.values())


def rocm_key_format(build_sh: str) -> list[str]:
    lines = []
    for start, end in ROCM_KEY_RANGES:
        inside = False
        for line in build_sh.splitlines():
            if inside or line.startswith(start):
                if not line.lstrip().startswith("#"):
                    lines.append(line)
                inside = not (inside and line.startswith(end))
    return lines


def rocm_changes_key_format(
    github: GitHubClient, pr: Mapping[str, Any], files: set[str]
) -> bool:
    if ROCM_BUILD_SH not in files:
        return False
    head = pr["head"]["sha"]
    base = github.get_merge_base(pr["base"]["ref"], head)
    return rocm_key_format(github.get_file_text(ROCM_BUILD_SH, base)) != (
        rocm_key_format(github.get_file_text(ROCM_BUILD_SH, head))
    )


def rocm_cache_run_note(
    github: GitHubClient,
    pr: Mapping[str, Any],
    trusted: bool,
    trusted_users: set[str],
) -> str:
    files = set(github.list_pr_files(pr["number"]))
    stack_files = [path for path in ROCM_STACK_INPUTS if path in files]
    if not stack_files:
        return ""
    if rocm_changes_key_format(github, pr, files):
        return (
            "This PR changes how `build.sh` keys the ROCm base image: `main` "
            "rebuilds it after the merge, so there is nothing to cache."
        )
    if trusted or pr["head"]["sha"] in trusted_approved_commits(
        github, pr["number"], trusted_users
    ):
        return (
            "This PR changes the ROCm base image: it is cached into `main` "
            "automatically when this build finishes. The `ROCm image cache` step "
            "must be green before merge."
        )
    owners = codeowners_for(github.get_file_text(CODEOWNERS_PATH, "main"), stack_files)
    mentions: dict[str, str] = {}
    for user in (pr["user"]["login"], *owners):
        user = user.removeprefix("@")
        mentions.setdefault(user.casefold(), user)
    return (
        f"{' '.join(f'@{user}' for user in mentions.values())}: this PR changes "
        f"the ROCm base image. A maintainer must comment `{COMMAND_RUN_CI}`, or a "
        f"trusted reviewer must approve this commit and comment `{COMMAND_RUN_CI}`, "
        "to cache it into `main`; until then the `ROCm image cache` step blocks "
        "the merge."
    )


def defines_step_key(pipeline_yaml: str, key: str) -> bool:
    for line in pipeline_yaml.splitlines():
        line = line.split("#", 1)[0].strip().removeprefix("- ")
        name, _, value = line.partition(":")
        if name.strip() == "key" and value.strip().strip("\"'") == key:
            return True
    return False


def find_rocm_cache_source(
    github: GitHubClient,
    pr: Mapping[str, Any],
    buildkites: Sequence[BuildkiteClient],
    trusted_users: set[str],
) -> dict[str, Any] | None:
    """Return the newest build of the PR that may be cached into main.

    A trusted user had the bot build its commit, or a trusted reviewer's latest
    review approved that commit: every build the bot made of the commit runs
    the same code. It pushed the ROCm stack itself, finished its ROCm base image
    step (a later failure leaves the stack whole), and built the PR head's stack
    inputs. Its pull request is checked, not
    is_build_for_pr's github-pr-number fallback, which is metadata that any job
    can write.
    """
    # Builds are created under the head label (owner:branch); older ones under
    # the bare ref, unless that is the base branch, whose builds are not this PR's.
    branches = dict.fromkeys(
        branch
        for branch in (pr["head"].get("label"), pr["head"]["ref"])
        if branch and branch != pr["base"]["ref"]
    )
    builds = sorted(
        (
            (build, client)
            for client in buildkites
            for branch in branches
            for build in iter_builds(
                client, branch=branch, created_from=pr.get("created_at")
            )
            if str((build.get("pull_request") or {}).get("id")) == str(pr["number"])
            and build.get("source") == "api"
        ),
        key=lambda candidate: candidate[0].get("created_at", ""),
        reverse=True,
    )
    actors = {
        str((build.get("env") or {}).get("VLLM_CI_TRIGGERED_BY", ""))
        for build, _ in builds
    }
    trusted_actors = {
        actor
        for actor in actors
        if actor and is_trusted_user(github, actor, trusted_users)
    }
    vouched = trusted_approved_commits(github, pr["number"], trusted_users) | {
        build["commit"]
        for build, _ in builds
        if (build.get("env") or {}).get("VLLM_CI_TRIGGERED_BY") in trusted_actors
    }
    head_stack = {
        p: github.get_file_sha(p, pr["head"]["sha"]) for p in ROCM_STACK_INPUTS
    }
    for build, client in builds:
        built = (build.get("meta_data") or {}).get(ROCM_STACK_BUILT_KEY)
        if not built or build["commit"] not in vouched:
            continue
        jobs = client.list_jobs(build["number"], step_key=ROCM_REFRESH_STEP_KEY)
        latest = max(jobs, key=lambda job: job.get("created_at") or "", default={})
        if not latest.get("finished_at"):
            continue
        if all(
            github.get_file_sha(path, build["commit"]) == sha
            for path, sha in head_stack.items()
        ):
            return build
    return None


def start_rocm_promote(
    github: GitHubClient,
    ci_buildkite: BuildkiteClient,
    pr: Mapping[str, Any],
    source: Mapping[str, Any],
    promote_id: str,
) -> str:
    sha = source["commit"]
    # A main commit, so build.sh and the pipeline are trusted.
    commit = github.get_merge_base("main", sha)
    meta_data = source["meta_data"]
    build = ci_buildkite.create_build(
        {
            "commit": commit,
            "branch": "main",
            "message": f"Cache the ROCm base image of PR #{pr['number']}",
            "env": {
                "ROCM_PROMOTE_PR": str(pr["number"]),
                "ROCM_PROMOTE_SHA": sha,
                "ROCM_PROMOTE_IMAGES": " ".join(
                    meta_data[key]
                    for key in ROCM_BUILT_KEYS
                    if meta_data.get(key, "none") != "none"
                ),
                "VLLM_CI_ONLY_STEP_KEYS": json.dumps([ROCM_CACHE_STEP_KEY]),
            },
            "meta_data": {ROCM_PROMOTE_KEY: promote_id},
        }
    )
    return f"Started {build['web_url']} to cache {source['web_url']}."


def reconcile_rocm_cache(
    pr: Mapping[str, Any],
    github: GitHubClient,
    ci_buildkite: BuildkiteClient,
    amd_buildkite: BuildkiteClient,
    trusted_users: set[str],
) -> str:
    """Take the next step to cache a PR's ROCm base image into main.

    Stateless and idempotent: it promotes the source build's images with a CI
    build on main, then retries the PR head's failed rocm-image-cache-check.
    Returns what it did, for the workflow log.
    """
    if pr["state"] != "open" or pr["base"]["ref"] != "main":
        return "Not an open PR to main."
    files = set(github.list_pr_files(pr["number"]))
    if not files & set(ROCM_STACK_INPUTS):
        return "The PR does not change the ROCm base image."
    if rocm_changes_key_format(github, pr, files):
        return "The PR changes how build.sh keys the stack; main rebuilds after merge."

    head_builds = [
        build
        for build in ci_buildkite.list_builds(pr["head"]["sha"])
        if str((build.get("pull_request") or {}).get("id")) == str(pr["number"])
    ]
    head_build = max(
        head_builds, key=lambda build: build.get("created_at", ""), default=None
    )
    checks = (
        ci_buildkite.list_jobs(head_build["number"], step_key=ROCM_CACHE_CHECK_KEY)
        if head_build
        else []
    )
    latest_check = max(checks, key=lambda job: job.get("created_at") or "", default={})
    if latest_check.get("state") == "passed":
        return "The PR head's cache check passed."
    # main would never use this stack: the merge gets another one.
    if latest_check.get("exit_status") == ROCM_CHECK_MAIN_MOVED:
        return "Main changed the ROCm stack; the PR must merge main."

    source = find_rocm_cache_source(
        github, pr, (ci_buildkite, amd_buildkite), trusted_users
    )
    if source is None:
        return "No trusted CI build has built the PR's ROCm stack yet."

    sha = source["commit"]
    promote_id = f"{pr['number']}@{sha}"
    # Only builds the bot created: a trigger step can set meta-data.
    promotes = [
        build
        for build in ci_buildkite.list_builds(
            None, branch="main", metadata=(ROCM_PROMOTE_KEY, promote_id)
        )
        if build.get("source") == "api"
    ]
    promote = max(promotes, key=lambda build: build.get("created_at", ""), default=None)
    if promote is None:
        return start_rocm_promote(github, ci_buildkite, pr, source, promote_id)
    promoted_at, promote_state = rocm_promote_result(ci_buildkite, promote)
    if not promoted_at:
        return f"Waiting for {promote['web_url']}."
    if promote_state != "passed":
        if len(promotes) < ROCM_PROMOTE_ATTEMPTS:
            return start_rocm_promote(github, ci_buildkite, pr, source, promote_id)
        marker = f"<!-- vllm-ci-rocm-promote-failed:{promote_id} -->"
        if not has_bot_comment_marker(github, pr["number"], marker):
            github.add_comment(
                pr["number"],
                f"❌ Caching this PR's ROCm base image into `main` failed "
                f"{len(promotes)} times, last in {promote['web_url']}. A "
                f"maintainer must look into it.\n\n{marker}",
            )
        return f"Gave up after {len(promotes)} failed promote builds."

    if head_build is None:
        return "No CI build for the PR head to check the cache."
    # A check that started after the promote build finished already saw its result.
    failed = [
        job
        for job in checks
        if job.get("state") in RETRY_STATES.split(",")
        and (job.get("started_at") or "") < promoted_at
    ]
    check = max(failed, key=lambda job: job.get("created_at") or "", default=None)
    if check is None:
        return "No failed cache check needs a retry."
    try:
        ci_buildkite.retry_job(head_build["number"], check["id"])
    except ApiError as error:
        if error.status is None or not 400 <= error.status < 500:
            raise
        return f"Could not retry the cache check: {error}"
    return f"Retried the cache check of {head_build['web_url']}."


def rocm_promote_result(
    ci_buildkite: BuildkiteClient, promote: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """When a promote build's cache job finished, and how; the build's own
    if it has none (its pipeline never uploaded). The job's status wakes the
    bot before the build itself has finished."""
    jobs = ci_buildkite.list_jobs(promote["number"], step_key=ROCM_CACHE_STEP_KEY)
    job = max(jobs, key=lambda job: job.get("created_at") or "", default=None)
    if job is None:
        return promote.get("finished_at"), promote.get("state")
    return job.get("finished_at"), job.get("state")


def handle_buildkite_status(
    event: Mapping[str, Any],
    github: GitHubClient,
    ci_buildkite: BuildkiteClient,
    amd_buildkite: BuildkiteClient,
    trusted_users: set[str],
) -> str:
    """Reconcile the PR whose ROCm image cache check or promote job just
    finished (the `ci/step:` statuses rocm-image-cache.yml listens to).

    The status's target URL only names the build (and the job, after `#`);
    the build itself, fetched from Buildkite, says which PR it belongs to.
    The build may still be running: these jobs finish before it does.
    """
    url = str(event.get("target_url") or "").partition("#")[0]
    number = url.rpartition("/builds/")[2]
    if not number.isdigit():
        return "The status does not link a Buildkite build."
    build = ci_buildkite.get_build(int(number))
    if build.get("source") != "api" or build.get("commit") != event.get("sha"):
        return "Not a build the bot created for this commit."
    pr_number = str((build.get("pull_request") or {}).get("id") or "")
    if not pr_number and build.get("branch") == "main":
        pr_number = str((build.get("env") or {}).get("ROCM_PROMOTE_PR", ""))
    if not pr_number.isdigit():
        return "Not a PR or ROCm promote build."
    pr = github.get_pr(int(pr_number))
    return reconcile_rocm_cache(pr, github, ci_buildkite, amd_buildkite, trusted_users)


def recheck_rocm_stack_prs(
    github: GitHubClient, ci_buildkite: BuildkiteClient
) -> list[str]:
    """Retry the passed cache check of each open PR, after main changed its stack.

    Each check compares the PR with main's stack as it was when it ran. Run again,
    it fails until the PR merges main if main changed the stack since its merge base.
    Returns the builds whose check was retried.
    """
    seen, retried, failed = set(), [], []
    for build in iter_builds(ci_buildkite, metadata=(ROCM_CACHED_KEY, "true")):
        number = str((build.get("pull_request") or {}).get("id") or "")
        if build.get("source") != "api" or not number.isdigit() or number in seen:
            continue
        seen.add(number)  # its newest build
        try:
            pr = github.get_pr(int(number))
            if (
                pr["state"] != "open"
                or pr["base"]["ref"] != "main"
                or pr["head"]["sha"] != build["commit"]
            ):
                continue
            checks = ci_buildkite.list_jobs(
                build["number"], step_key=ROCM_CACHE_CHECK_KEY
            )
            check = max(checks, key=lambda job: job.get("created_at") or "", default={})
            if check.get("state") != "passed":
                continue
            ci_buildkite.retry_job(build["number"], check["id"])
        except ApiError as error:
            print(f"Could not recheck PR #{number}: {error}")
            if error.status is None or not 400 <= error.status < 500:
                failed.append(number)
            continue
        retried.append(build["web_url"])
    if failed:
        raise ApiError(None, f"Could not recheck PRs {failed}; retried {retried}.")
    return retried


def reply_unrecognized_command(
    event: Mapping[str, Any],
    github: GitHubClient,
) -> None:
    if "pull_request" not in event["issue"]:
        return
    issue_number = event["issue"]["number"]
    comment_id = event["comment"]["id"]
    actor = event["comment"]["user"]["login"]
    if is_already_handled(github, issue_number, comment_id):
        print(f"Comment {comment_id} was already handled.")
        return
    add_reaction_safely(github, comment_id, "-1")
    commands = ", ".join(f"`{command}`" for command in sorted(ALL_CI_COMMANDS))
    github.add_comment(
        issue_number,
        f"❌ @{actor}, that is not a recognized CI command, so no CI was "
        "started. Post a new comment containing only one of: "
        f"{commands}.\n\n{command_comment_marker(comment_id)}",
    )


def run(
    event: Mapping[str, Any],
    github: GitHubClient,
    buildkite: BuildkiteClient,
    trusted_users_value: str = "",
    branch_updater: GitHubClient | None = None,
    ci_buildkite: BuildkiteClient | None = None,
    amd_buildkite: BuildkiteClient | None = None,
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

    rocm = False
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
        trusted = is_trusted_permission(permission) or actor.casefold() in trusted_users
        should_check_approval = (
            not trusted
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
            run_command=run_command_for_command(command),
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
        rocm = command.startswith(COMMAND_RUN_CI) and pr["base"]["ref"] == "main"
        if command in RUN_CI_COMMAND_ENV:
            message = handle_run_ci(
                actor=actor,
                buildkite=buildkite,
                comment_id=comment_id,
                command=command,
                github=github,
                pr=pr,
                branch_updater=branch_updater,
                allow_repeat_run=is_trusted_permission(permission),
            )
        elif command in RETRY_COMMANDS:
            message = handle_retry_failed(
                actor=actor,
                buildkite=buildkite,
                comment_id=comment_id,
                command=command,
                github=github,
                pr=pr,
            )
        elif command in CANCEL_COMMANDS:
            message = handle_cancel_ci(
                buildkite=buildkite,
                command=command,
                pr=pr,
            )
        else:
            raise ValueError(f"Unsupported CI command: {command}")
        add_reaction_safely(github, comment_id, "rocket")
        if rocm:
            # The head the run built: handle_run_ci may have merged main into it.
            try:
                note = rocm_cache_run_note(
                    github, github.get_pr(issue_number), trusted, trusted_users
                )
            except Exception as error:
                note = ""
                print(f"No ROCm image cache note: {error}")
            if note:
                message = f"{message}\n\n{note}"
        github.add_comment(issue_number, f"✅ {message}")
    except CiPreparationError as error:
        add_reaction_safely(github, comment_id, "-1")
        github.add_comment(
            issue_number,
            f"❌ {error}\n\n{command_comment_marker(comment_id)}",
        )
    except Exception:
        add_reaction_safely(github, comment_id, "confused")
        raise
    # Also when preparing the run failed: an approval can vouch for an earlier build.
    if rocm:
        try:
            result = reconcile_rocm_cache(
                github.get_pr(issue_number),
                github,
                ci_buildkite or buildkite,
                amd_buildkite or buildkite,
                parse_trusted_users(trusted_users_value),
            )
        except Exception as error:
            result = f"Error (comment /ci run again to retry): {error}"
        print(f"ROCm image cache: {result}")


def main() -> None:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, encoding="utf-8") as event_file:
        event = json.load(event_file)

    github = GitHubClient(
        os.environ.get("GH_TOKEN", ""),
        os.environ["GITHUB_REPOSITORY"],
    )
    event_name = os.environ.get("GITHUB_EVENT_NAME", "issue_comment")
    buildkite_token = os.environ.get("BUILDKITE_API_TOKEN", "")
    organization = os.environ.get("BUILDKITE_ORGANIZATION", "vllm")
    ci_pipeline = os.environ.get("BUILDKITE_PIPELINE", "ci")
    amd_pipeline = os.environ.get("BUILDKITE_AMD_PIPELINE", "amd-ci")
    ci_buildkite = BuildkiteClient(buildkite_token, organization, ci_pipeline)
    amd_buildkite = BuildkiteClient(buildkite_token, organization, amd_pipeline)
    if event_name == "status":
        print(
            handle_buildkite_status(
                event,
                github,
                ci_buildkite,
                amd_buildkite,
                parse_trusted_users(os.environ.get("CI_TRUSTED_USERS", "")),
            )
        )
        return
    if event_name == "push":
        print(recheck_rocm_stack_prs(github, ci_buildkite))
        return
    if event_name == "pull_request_target":
        notify_authorized(
            event,
            github,
            os.environ.get("CI_TRUSTED_USERS", ""),
        )
        return
    if event_name == "workflow_run":
        pr = resolve_workflow_run_pr(event["workflow_run"], github)
        if pr is None:
            print("Could not resolve an open PR for the approval workflow run.")
            return
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

    body = event["comment"]["body"]
    command = parse_command(body)
    if not command:
        if is_ci_command_attempt(body):
            reply_unrecognized_command(event, github)
        return

    pipeline = pipeline_for_command(
        command, amd_ci_pipeline=amd_pipeline, upstream_ci_pipeline=ci_pipeline
    )
    branch_update_token = os.environ.get("GH_BRANCH_UPDATE_TOKEN", "")
    run(
        event,
        github,
        amd_buildkite if pipeline == amd_pipeline else ci_buildkite,
        os.environ.get("CI_TRUSTED_USERS", ""),
        GitHubClient(branch_update_token, os.environ["GITHUB_REPOSITORY"])
        if branch_update_token
        else None,
        ci_buildkite,
        amd_buildkite,
    )


if __name__ == "__main__":
    main()
