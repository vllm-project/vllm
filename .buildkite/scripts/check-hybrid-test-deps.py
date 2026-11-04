# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks the test image has the prebuilt mamba-ssm and causal-conv1d.

docker/Dockerfile's hybrid-test-wheels stage builds them from the commits that
tools/build_hybrid_test_wheels.sh pins, and records those commits in
COMMITS_FILE.
"""

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

COMMITS_FILE = Path("/opt/hybrid-test-deps-commits.txt")
EXPECTED_VERSIONS = {"mamba_ssm": "2.3.0", "causal_conv1d": "1.6.0"}
EXPECTED_COMMITS = {
    "mamba_ssm": "f1493ff6e9335160eb134eb67e59f8e4d9adefd6",
    "causal_conv1d": "da6dbaa9fd5a919967f14d3fd031da1288ad5025",
}


def find_problems(
    installed_versions: dict[str, str | None], commits_text: str | None
) -> list[str]:
    """Lists every way the image differs from the pinned builds.

    Args:
        installed_versions: Installed version of each package, None if missing.
        commits_text: Contents of COMMITS_FILE, None if the file is missing.

    Returns:
        One message per problem; empty when the image matches the pins.

    """
    problems = []
    for package, expected_version in EXPECTED_VERSIONS.items():
        installed_version = installed_versions[package]
        if installed_version != expected_version:
            problems.append(
                f"{package}: installed {installed_version}, expected {expected_version}"
            )
    if commits_text is None:
        problems.append(f"{COMMITS_FILE} is missing")
        return problems
    recorded_commits = {}
    for line in commits_text.splitlines():
        package, commit = line.split()
        recorded_commits[package] = commit
    if recorded_commits != EXPECTED_COMMITS:
        problems.append(f"built from {recorded_commits}, expected {EXPECTED_COMMITS}")
    return problems


def main() -> None:
    installed_versions = {}
    for package in EXPECTED_VERSIONS:
        try:
            installed_versions[package] = version(package)
        except PackageNotFoundError:
            installed_versions[package] = None
    commits_text = COMMITS_FILE.read_text() if COMMITS_FILE.exists() else None
    problems = find_problems(installed_versions, commits_text)
    if problems:
        sys.exit(
            "The test image's hybrid test dependencies do not match the pins in "
            "tools/build_hybrid_test_wheels.sh (arm64 and Rubin images skip "
            "them):\n" + "\n".join(problems)
        )

    # Importing loads each package's CUDA extension, so a build against a
    # different torch fails here rather than in the tests.
    import causal_conv1d
    import mamba_ssm

    print(
        "Verified hybrid test dependencies:",
        f"mamba_ssm=={mamba_ssm.__version__}@{EXPECTED_COMMITS['mamba_ssm']}",
        f"causal_conv1d=={causal_conv1d.__version__}@{EXPECTED_COMMITS['causal_conv1d']}",
    )


if __name__ == "__main__":
    main()
