# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <ci> <python_version> <changed_files...>

Args:
    ci: "1" if running in CI, "0" otherwise. In CI, follow_imports is set to
        "silent" for the main group of files.
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import subprocess
import sys
from typing import Optional

import regex as re

FILES = [
    "vllm/*.py",
    "vllm/assets",
    "vllm/entrypoints",
    "vllm/inputs",
    "vllm/logging_utils",
    "vllm/multimodal",
    "vllm/platforms",
    "vllm/transformers_utils",
    "vllm/triton_utils",
    "vllm/usage",
]

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", move the following directories to FILES
SEPARATE_GROUPS = [
    "tests",
    "vllm/attention",
    "vllm/compilation",
    "vllm/distributed",
    "vllm/engine",
    "vllm/executor",
    "vllm/inputs",
    "vllm/lora",
    "vllm/model_executor",
    "vllm/plugins",
    "vllm/worker",
    "vllm/v1",
]

# TODO(woosuk): Include the code from Megatron and HuggingFace.
EXCLUDE = [
    "vllm/model_executor/parallel_utils",
    "vllm/model_executor/models",
    "vllm/model_executor/layers/fla/ops",
    # Ignore triton kernels in ops.
    "vllm/attention/ops",
]


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """
    Group changed files into different mypy calls.

    Args:
        changed_files: List of changed files.

    Returns:
        A dictionary mapping file group names to lists of changed files.
    """
    exclude_pattern = re.compile(f"^{'|'.join(EXCLUDE)}.*")
    files_pattern = re.compile(f"^({'|'.join(FILES)}).*")
    file_groups = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    for changed_file in changed_files:
        # Skip files which should be ignored completely
        if exclude_pattern.match(changed_file):
            continue
        # Group files by mypy call
        if files_pattern.match(changed_file):
            file_groups[""].append(changed_file)
            continue
        else:
            for directory in SEPARATE_GROUPS:
                if re.match(f"^{directory}.*", changed_file):
                    file_groups[directory].append(changed_file)
                    break
    return file_groups


def mypy(
    targets: list[str],
    python_version: Optional[str],
    follow_imports: Optional[str],
    file_group: str,
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        file_group: The file group name for logging purposes.

    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {file_group}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    ci = sys.argv[1] == "1"
    python_version = sys.argv[2]
    file_groups = group_files(sys.argv[3:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if ci and file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(
                changed_files, python_version, follow_imports, file_group
            )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
