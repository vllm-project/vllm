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

import regex as re

FILES = [
    "vllm/*.py",
    "vllm/assets",
    "vllm/compilation",
    "vllm/distributed",
    "vllm/engine",
    "vllm/entrypoints",
    "vllm/executor",
    "vllm/inputs",
    "vllm/logging_utils",
    "vllm/multimodal",
    "vllm/platforms",
    "vllm/plugins",
    "vllm/renderers",
    "vllm/tokenizers",
    "vllm/transformers_utils",
    "vllm/triton_utils",
    "vllm/usage",
    "vllm/utils",
    "vllm/worker",
    "vllm/v1/attention",
    "vllm/v1/core",
    "vllm/v1/engine",
    "vllm/v1/executor",
    "vllm/v1/metrics",
    "vllm/v1/pool",
    "vllm/v1/sample",
    "vllm/v1/structured_output",
    "vllm/v1/worker",
]

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", remove its directory from SEPARATE_GROUPS.
SEPARATE_GROUPS = [
    "tests",
    # v0 related
    "vllm/lora",
    "vllm/model_executor",
    # v1 related
    "vllm/v1/kv_offload",
    "vllm/v1/spec_decode",
]

# TODO(woosuk): Include the code from Megatron and HuggingFace.
EXCLUDE = [
    "vllm/engine/arg_utils.py",
    "vllm/model_executor/parallel_utils",
    "vllm/model_executor/models",
    "vllm/model_executor/layers/fla/ops",
    # Ignore triton kernels in ops.
    "vllm/v1/attention/ops",
    # TODO: Remove these entries after fixing mypy errors.
    "vllm/benchmarks",
    "vllm/config",
    "vllm/device_allocator",
    "vllm/profiler",
    "vllm/reasoning",
    "vllm/tool_parser",
    "vllm/v1/cudagraph_dispatcher.py",
    "vllm/outputs.py",
    "vllm/logger.py",
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
    file_groups = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    for changed_file in changed_files:
        # Skip files which should be ignored completely
        if exclude_pattern.match(changed_file):
            continue
        # Group files by mypy call
        for directory in SEPARATE_GROUPS:
            if re.match(f"^{directory}.*", changed_file):
                file_groups[directory].append(changed_file)
                break
        else:
            if changed_file.startswith("vllm/"):
                file_groups[""].append(changed_file)
    return file_groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
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
