#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find published ancestor caches without waiting for in-flight main builds."""

import json
import os
import subprocess
import sys


def main() -> None:
    if os.environ.get("BUILDKITE_BRANCH") != "main":
        return
    try:
        # HEAD and its parent are already imported by ci.hcl.
        commits = subprocess.check_output(
            ["git", "rev-list", "--first-parent", "--max-count=22", "HEAD"],
            text=True,
            timeout=3,
        ).splitlines()[2:]
        if not commits:
            return
        published = json.loads(
            subprocess.check_output(
                [
                    "aws",
                    "ecr",
                    "batch-get-image",
                    "--region",
                    "us-east-1",
                    "--registry-id",
                    "936637512419",
                    "--repository-name",
                    "vllm-ci-test-cache",
                    "--image-ids",
                    *(f"imageTag={commit}" for commit in commits),
                    "--query",
                    "images[].imageId.imageTag",
                    "--output",
                    "json",
                    "--cli-connect-timeout",
                    "3",
                    "--cli-read-timeout",
                    "5",
                    "--no-cli-pager",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        )
        if not isinstance(published, list):
            return
        # Registry response order is arbitrary; prefer the nearest ancestors.
        available = [commit for commit in commits if commit in published][:3]
        for commit in available:
            print(
                "type=registry,ref=936637512419.dkr.ecr.us-east-1.amazonaws.com/"
                f"vllm-ci-test-cache:{commit}"
            )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Ancestor cache lookup skipped: {type(exc).__name__}", file=sys.stderr)


if __name__ == "__main__":
    main()
