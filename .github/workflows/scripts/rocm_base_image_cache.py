# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy a ROCm stack PR's base images into rocm/vllm-ci once a maintainer labels it.

Run by rocm-base-image-cache.yml when the label is added. It waits for this PR's
CI to build the images if needed, then starts a main build of AMD CI that runs
build.sh promote with the digests that CI build recorded, so main reuses them
after the merge. pre-commit.yml fails a stack PR into main until it has the label.
"""

import json
import os
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from run_ci_command import ApiError, BuildkiteClient, GitHubClient, is_build_for_pr

CACHE_LABEL = "cache-rocm-base-image"
STACK_INPUTS = ("docker/Dockerfile.rocm_base", "tools/vllm-rocm/therock_wheels.py")
# build.sh promote copies only images built with main's versions of these.
RECIPE = (".buildkite/scripts/rocm/build.sh", "docker/docker-bake-rocm.hcl")
# build.sh base mode records the stack images it built under these keys.
IMAGE_KEYS = ("rocm-rocm-stack-wheels", "rocm-rocm-stack")
POLL_SECONDS = 300  # a stack build takes hours
WAIT_SECONDS = 5 * 3600  # well under the job's 6 h timeout


class CacheGitHub(GitHubClient):
    def list_pr_files(self, number: int) -> list[str]:
        files = self._paginate(self._repo_path(f"/pulls/{number}/files"))
        return [
            str(path)
            for file in files
            for path in (file["filename"], file.get("previous_filename"))
            if path
        ]

    def remove_label(self, issue_number: int, label: str) -> None:
        label = urllib.parse.quote(label, safe="")
        try:
            self._request(
                self._repo_path(f"/issues/{issue_number}/labels/{label}"),
                method="DELETE",
            )
        except ApiError as error:
            if error.status != 404:  # already removed
                raise

    def get_file_sha(self, path: str, ref: str) -> str:
        path, ref = urllib.parse.quote(path), urllib.parse.quote(ref, safe="")
        try:
            return self._request(self._repo_path(f"/contents/{path}?ref={ref}"))["sha"]
        except ApiError as error:
            if error.status == 404:
                return "none"
            raise


def recorded_images(
    sources: Sequence[BuildkiteClient], pr: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The latest CI build of the PR head that recorded its stack images."""
    builds = [
        build
        for client in sources
        for build in client.list_builds(pr["head"]["sha"])
        if is_build_for_pr(build, pr["number"])
        and all((build.get("meta_data") or {}).get(key) for key in IMAGE_KEYS)
    ]
    return max(builds, key=lambda build: build.get("created_at", ""), default=None)


def cache(
    github: CacheGitHub,
    amd: BuildkiteClient,
    sources: Sequence[BuildkiteClient],
    number: int,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wait_seconds: float = WAIT_SECONDS,
) -> tuple[str | None, bool]:
    """Start the copy once the PR's CI recorded its images.

    Returns the comment to post, and whether the label should come off: it only
    stays on a PR whose image is being cached.
    """
    deadline = clock() + wait_seconds
    recipes: dict[str, bool] = {}  # head SHA -> built with main's recipe
    while True:
        try:
            pr = github.get_pr(number)
            if (
                pr["state"] != "open"
                or pr["base"]["ref"] != "main"
                or CACHE_LABEL not in {label["name"] for label in pr["labels"]}
                or not set(STACK_INPUTS) & set(github.list_pr_files(number))
            ):
                return None, False
            sha = pr["head"]["sha"]
            if sha not in recipes:
                recipes[sha] = all(
                    github.get_file_sha(p, "main") == github.get_file_sha(p, sha)
                    for p in RECIPE
                )
            if not recipes[sha]:
                return (
                    f"`{sha[:12]}` has another `build.sh` or `docker-bake-rocm.hcl` "
                    "than main, so its ROCm base image cannot be cached. If main "
                    "changed them, merge main (`/ci run`) and add "
                    f"`{CACHE_LABEL}` again; if this PR changes them, main builds "
                    "the image after the merge."
                ), True
            source = recorded_images(sources, pr)
        except ApiError as error:  # a missed poll, not a reason to give up
            print(f"Poll failed: {error}")
            source = None
        if source:
            break
        if clock() >= deadline:
            return (
                "No CI build of this PR's head built the ROCm base image in time, so "
                f"it was not cached. Once CI has built it, add `{CACHE_LABEL}` again."
            ), True
        sleep(POLL_SECONDS)

    build = amd.create_build(
        {
            "commit": "HEAD",
            "branch": "main",
            "message": f"Cache the ROCm base image of PR #{number}",
            "env": {
                "ROCM_PROMOTE_PR": str(number),
                "ROCM_PROMOTE_SHA": sha,
                "ROCM_PROMOTE_IMAGES": " ".join(
                    source["meta_data"][key] for key in IMAGE_KEYS
                ),
            },
        }
    )
    return (
        f"Caching the ROCm base image of `{sha[:12]}` for main: "
        f"[Buildkite AMD CI #{build['number']}]({build['web_url']}) copies what "
        f"[this PR's CI]({source['web_url']}) built into `rocm/vllm-ci`."
    ), False


def main() -> None:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        number = json.load(event_file)["pull_request"]["number"]
    github = CacheGitHub(
        os.environ.get("GH_TOKEN", ""), os.environ["GITHUB_REPOSITORY"]
    )
    token = os.environ.get("BUILDKITE_API_TOKEN", "")
    org = os.environ.get("BUILDKITE_ORGANIZATION", "vllm")
    amd = BuildkiteClient(
        token, org, os.environ.get("BUILDKITE_AMD_PIPELINE", "amd-ci")
    )
    ci = BuildkiteClient(token, org, os.environ.get("BUILDKITE_PIPELINE", "ci"))
    try:
        message, unlabel = cache(github, amd, (amd, ci), number)
    except Exception:
        message, unlabel = (
            f"Caching the ROCm base image failed (see the `{CACHE_LABEL}` workflow "
            f"run); add `{CACHE_LABEL}` again to retry.",
            True,
        )
        raise
    finally:
        # The label only stays on a PR whose image is being cached, so the
        # pre-commit check fails again until a maintainer adds it back.
        if unlabel:
            github.remove_label(number, CACHE_LABEL)
        if message:
            github.add_comment(number, message)


if __name__ == "__main__":
    main()
