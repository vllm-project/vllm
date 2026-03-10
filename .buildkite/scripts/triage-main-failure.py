#!/usr/bin/env python3
"""
Triage main branch CI failures using the Buildkite API.

Fetches the latest main branch build status, identifies failed builds,
and attempts to determine which PR may have caused the failure by
analyzing commit history (merge commits).

Requires: BUILDKITE_API_KEY environment variable
"""

import json
import os
import re
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Buildkite API configuration for vllm
ORG_SLUG = "vllm"
PIPELINE_SLUG = "ci"
API_BASE = "https://api.buildkite.com/v2"


def api_request(path: str, params: dict | None = None, allow_404: bool = False) -> dict | list | None:
    """Make authenticated request to Buildkite API."""
    token = os.environ.get("BUILDKITE_API_KEY")
    if not token:
        print("ERROR: BUILDKITE_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)

    url = f"{API_BASE}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    req = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        if allow_404 and e.code == 404:
            return None
        print(f"API error: {e.code} {e.reason}", file=sys.stderr)
        if e.code == 401:
            print("Check that BUILDKITE_API_KEY is valid and has read_builds scope", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        sys.exit(1)


def get_main_builds(limit: int = 20) -> list:
    """Fetch latest builds for main branch."""
    path = f"/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds"
    data = api_request(path, {"branch": "main", "per_page": str(limit)})
    return data if isinstance(data, list) else []


def get_build_with_jobs(build_number: str | int) -> dict | None:
    """Fetch full build details (includes jobs in response)."""
    path = f"/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds/{build_number}"
    data = api_request(path, allow_404=True)
    return data if isinstance(data, dict) else None


def get_failed_jobs_from_build(build: dict) -> list:
    """Extract failed jobs from build (jobs may be in build or need separate fetch)."""
    jobs = build.get("jobs") or []
    if not jobs and build.get("number"):
        # Jobs might not be in list response - fetch full build
        full = get_build_with_jobs(build["number"])
        if full:
            jobs = full.get("jobs") or []
    return [j for j in jobs if j.get("state") == "failed"]


def get_pr_from_message(message: str) -> str | None:
    """Extract PR number from build message like '... (#36475)' or 'Merge pull request #123'."""
    # Match (#12345) at end of message or Merge pull request #123
    for pattern in [r"\(#(\d+)\)", r"Merge pull request #(\d+)", r"#(\d+)"]:
        match = re.search(pattern, message)
        if match:
            return f"https://github.com/vllm-project/vllm/pull/{match.group(1)}"
    return None


def get_pr_from_commit(commit_sha: str) -> str | None:
    """
    Try to find the PR that introduced a commit by checking if it's a merge commit
    and looking at the merge source.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%P %s", commit_sha],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/workspace",
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip()
        if not line:
            return None
        parts = line.split(" ", 1)
        parents = parts[0].split() if parts[0] else []
        subject = parts[1] if len(parts) > 1 else ""

        # Merge commit: first parent is main, second is the PR branch
        if len(parents) >= 2 and "Merge pull request" in subject:
            # Extract PR number from "Merge pull request #123 from org/branch"
            match = re.search(r"#(\d+)", subject)
            if match:
                return f"https://github.com/vllm-project/vllm/pull/{match.group(1)}"

        # Check a few commits back for merge commits (in case this is a direct push)
        result = subprocess.run(
            ["git", "log", "-10", "--format=%H %s", commit_sha, "--ancestry-path", "--first-parent"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/workspace",
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                sha, subject = line.split(" ", 1)
                if "Merge pull request" in subject:
                    match = re.search(r"#(\d+)", subject)
                    if match:
                        return f"https://github.com/vllm-project/vllm/pull/{match.group(1)}"
    except Exception:
        pass
    return None


def main():
    print("=" * 60)
    print("Main Branch CI Failure Triage")
    print("=" * 60)

    builds = get_main_builds(limit=15)
    if not builds:
        print("No builds found for main branch.")
        return

    # Find failed builds
    failed_builds = [b for b in builds if b.get("state") == "failed"]
    failed_builds = failed_builds[:5]  # Look at most recent 5 failures

    if not failed_builds:
        print("\nNo failed builds found on main. Latest build status:")
        latest = builds[0]
        print(f"  Build #{latest.get('number')}: {latest.get('state')} - {latest.get('message', '')[:60]}...")
        print(f"  Commit: {latest.get('commit')}")
        print(f"  URL: {latest.get('web_url')}")
        return

    print(f"\nFound {len(failed_builds)} failed build(s) on main:\n")

    for build in failed_builds:
        build_id = build.get("id")
        build_num = build.get("number")
        commit = build.get("commit")
        msg = build.get("message", "")[:80]
        url = build.get("web_url")

        print(f"--- Build #{build_num} (FAILED) ---")
        print(f"  Commit: {commit}")
        print(f"  Message: {msg}")
        print(f"  URL: {url}")

        # Get failed jobs (from full build details)
        failed_jobs = get_failed_jobs_from_build(build)
        if failed_jobs:
            print(f"  Failed jobs ({len(failed_jobs)}):")
            for j in failed_jobs[:10]:
                name = j.get("name", "?")
                print(f"    - {name}")
            if len(failed_jobs) > 10:
                print(f"    ... and {len(failed_jobs) - 10} more")

        # Try to find PR (message first, then git history)
        pr_url = get_pr_from_message(msg) or get_pr_from_commit(commit)
        if pr_url:
            print(f"  Likely PR: {pr_url}")
        else:
            print(f"  Could not determine PR (run 'git fetch origin main' and retry)")

        print()

    # Summary: most recent failure
    latest_failed = failed_builds[0]
    commit = latest_failed.get("commit")
    msg = latest_failed.get("message", "")
    pr_url = get_pr_from_message(msg) or get_pr_from_commit(commit)

    print("=" * 60)
    print("TRIAGE SUMMARY")
    print("=" * 60)
    print(f"Latest failed build: #{latest_failed.get('number')}")
    print(f"Commit: {commit}")
    print(f"Build URL: {latest_failed.get('web_url')}")
    if pr_url:
        print(f"Most likely PR: {pr_url}")
    else:
        print("PR: Could not determine from commit history")
        print("  Tip: Run 'git fetch origin main' and retry, or check the build URL for PR links")
    print()


if __name__ == "__main__":
    main()
