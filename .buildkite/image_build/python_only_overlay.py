# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage a PR as a thin layer over the main CI image of BASE.

Usage: python_only_overlay.py BASE CONTEXT

If no file changed between BASE and HEAD is a build input, copy the changed
files into CONTEXT/vllm-workspace at their paths in the test image (vllm/ is
at src/vllm) and exit 0. Otherwise print why and exit 1.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Outside vllm/, these reach the test image only via `ADD . /vllm-workspace/`.
WORKSPACE_ONLY = ("tests/", "examples/", "benchmarks/", "docs/", ".buildkite/")
BUILD_INPUTS = (
    ".buildkite/image_build/",
    ".buildkite/check-wheel-size.py",
    "tests/vllm_test_utils/setup.py",  # installed editable at build time
    "vllm/envs.py",  # read by setup.py
    "vllm/__init__.py",  # copied into the csrc build stage
    "vllm/collect_env.py",  # also copied to /vllm-workspace/
)
# Files under vllm/ that the wheel ships unchanged from the source tree.
WHEEL_SUFFIXES = (".py", ".pyi", ".json", ".jinja")


def blocking_reason(status: str, path: str) -> str | None:
    if status not in ("A", "M"):
        return f"status {status} (a thin layer can only add or replace)"
    if path.startswith(BUILD_INPUTS):
        return "build input"
    if path.startswith("vllm/"):
        if status == "A" and not path.endswith(".py"):
            return "new data file, may be outside package_data"
        if not path.endswith(WHEEL_SUFFIXES):
            return "not shipped as-is in the wheel"
        return None
    if path.startswith(WORKSPACE_ONLY) or ("/" not in path and path.endswith(".md")):
        return None
    return "build input"


def image_path(path: str) -> str:
    return f"src/{path}" if path.startswith("vllm/") else path


def main(base: str, context: Path) -> int:
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", base, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changes = [line.split("\t", 1) for line in diff.splitlines()]
    blocked = [
        f"{path}: {reason}"
        for status, path in changes
        if (reason := blocking_reason(status, path))
    ]
    if blocked:
        print("Full build needed:", *blocked, sep="\n  ")
        return 1
    workspace = context / "vllm-workspace"
    (workspace / "src/vllm").mkdir(parents=True)
    for _, path in changes:
        dest = workspace / image_path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f"Thin build: {len(changes)} files over {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], Path(sys.argv[2])))
