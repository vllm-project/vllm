# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from typing import Any

from rocm_base_image_cache import CACHE_LABEL, POLL_SECONDS, cache
from run_ci_command import ApiError

IMAGES = {"rocm-rocm-stack": "s@d1", "rocm-rocm-stack-wheels": "w@d2"}


def make_pr(**overrides: Any) -> dict[str, Any]:
    return {
        "base": {"ref": "main"},
        "head": {"sha": "0123456789abcdef"},
        "labels": [{"name": CACHE_LABEL}],
        "number": 42,
        "state": "open",
        **overrides,
    }


def ci_build(**overrides: Any) -> dict[str, Any]:
    return {
        "created_at": "1",
        "meta_data": IMAGES,
        "pull_request": {"id": 42},
        "web_url": "https://buildkite.example/ci/builds/2",
        **overrides,
    }


class FakeGitHub:
    def __init__(self, prs: list[dict[str, Any]], files: list[str] | None = None):
        self.prs = prs  # get_pr returns the next one, then repeats the last
        self.files = ["docker/Dockerfile.rocm_base"] if files is None else files
        self.file_shas: dict[tuple[str, str], str] = {}

    def get_pr(self, number: int) -> dict[str, Any]:
        pr = self.prs.pop(0) if len(self.prs) > 1 else self.prs[0]
        if isinstance(pr, Exception):
            raise pr
        return pr

    def list_pr_files(self, number: int) -> list[str]:
        return self.files

    def get_file_sha(self, path: str, ref: str) -> str:
        return self.file_shas.get((path, ref), "blob")


class FakeBuildkite:
    def __init__(self, builds: list[list[dict[str, Any]]] | None = None) -> None:
        self.builds = builds or []  # one list per list_builds call, then empty
        self.listed: list[str] = []
        self.created: list[dict[str, Any]] = []

    def list_builds(self, commit: str) -> list[dict[str, Any]]:
        self.listed.append(commit)
        return self.builds.pop(0) if self.builds else []

    def create_build(self, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append(body)
        return {"number": 7, "web_url": "https://buildkite.example/amd-ci/builds/7"}


def run(github: FakeGitHub, ci: FakeBuildkite, wait: float = 0) -> tuple:
    """(comment, remove the label, builds created, sleeps) on a clock that only
    moves while sleeping."""
    amd, slept = FakeBuildkite(), []
    message, unlabel = cache(
        github,
        amd,
        (ci,),
        42,
        sleep=slept.append,
        clock=lambda: sum(slept),
        wait_seconds=wait,
    )
    return message, unlabel, amd.created, slept


class RocmBaseImageCacheTest(unittest.TestCase):
    def test_copies_the_images_the_prs_ci_recorded(self) -> None:
        builds = [
            ci_build(meta_data={key: "old" for key in IMAGES}, created_at="0"),
            ci_build(),
            ci_build(created_at="2", pull_request={"id": 7}),  # another PR
            ci_build(created_at="3", meta_data={"rocm-rocm-stack": "s"}),  # partial
        ]

        message, unlabel, created, _ = run(
            FakeGitHub([make_pr()]), FakeBuildkite([builds])
        )

        (build,) = created
        self.assertEqual((build["branch"], build["commit"]), ("main", "HEAD"))
        self.assertEqual(
            build["env"],
            {
                "ROCM_PROMOTE_PR": "42",
                "ROCM_PROMOTE_SHA": "0123456789abcdef",
                "ROCM_PROMOTE_IMAGES": "w@d2 s@d1",
            },
        )
        self.assertIn("ci/builds/2", message)
        self.assertFalse(unlabel)

    def test_waits_for_ci_to_build_the_images_of_the_current_head(self) -> None:
        # A push (or /ci run merging main) moves the head while it waits, and a
        # failed poll is retried.
        moved = make_pr(head={"sha": "fedcba9876543210"})
        github = FakeGitHub([make_pr(), ApiError(502, "bad gateway"), moved])
        ci = FakeBuildkite([[], [ci_build()]])

        message, _, created, slept = run(github, ci, wait=10 * POLL_SECONDS)

        self.assertEqual(slept, [POLL_SECONDS, POLL_SECONDS])
        self.assertEqual(ci.listed[-1], "fedcba9876543210")
        self.assertEqual(created[0]["env"]["ROCM_PROMOTE_SHA"], "fedcba9876543210")
        self.assertIn("Caching", message)

    def test_gives_up_after_the_wait_and_takes_the_label_off(self) -> None:
        message, unlabel, created, slept = run(
            FakeGitHub([make_pr()]), FakeBuildkite(), wait=2 * POLL_SECONDS
        )

        self.assertEqual((created, len(slept)), ([], 2))
        # Off, so the pre-commit check fails again until it is added back.
        self.assertTrue(unlabel)
        self.assertIn(f"add `{CACHE_LABEL}` again", message)

    def test_refuses_a_pr_built_with_another_recipe(self) -> None:
        github = FakeGitHub([make_pr()])
        github.file_shas = {("docker/docker-bake-rocm.hcl", "main"): "newer"}

        message, unlabel, created, _ = run(github, FakeBuildkite([[ci_build()]]))

        self.assertEqual((created, unlabel), ([], True))
        self.assertIn("another `build.sh` or `docker-bake-rocm.hcl`", message)

    def test_does_nothing_unless_a_labelled_stack_pr_into_main(self) -> None:
        cases = [
            (make_pr(labels=[]), None),  # label removed while waiting
            (make_pr(state="closed"), None),
            (make_pr(base={"ref": "releases/v1"}), None),
            (make_pr(), ["vllm/foo.py"]),
        ]
        for pr, files in cases:
            with self.subTest(pr=pr, files=files):
                message, unlabel, created, _ = run(
                    FakeGitHub([pr], files), FakeBuildkite([[ci_build()]])
                )
                self.assertEqual((message, unlabel, created), (None, False, []))


if __name__ == "__main__":
    unittest.main()
