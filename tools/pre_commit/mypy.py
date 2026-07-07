# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import subprocess
import sys

import regex as re

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", remove its directory from SEPARATE_GROUPS.
SEPARATE_GROUPS = [
    "tests",
    "tests/benchmarks",
    "tests/compile/correctness_e2e",
    "tests/config",
    "tests/compile",
    "tests/compile/fullgraph",
    "tests/compile/fusions_e2e",
    "tests/compile/passes",
    "tests/distributed",
    "tests/entrypoints/anthropic",
    "tests/entrypoints/generate",
    "tests/entrypoints/llm",
    "tests/entrypoints/multimodal",
    "tests/entrypoints/openai",
    "tests/entrypoints/pooling",
    "tests/entrypoints/serve",
    "tests/entrypoints/speech_to_text",
    "tests/entrypoints/tool_parsers",
    "tests/entrypoints/unit_tests",
    "tests/entrypoints/weight_transfer",
    "tests/kernels",
    "tests/kernels/attention",
    "tests/kernels/core",
    "tests/kernels/helion",
    "tests/kernels/mamba",
    "tests/kernels/moe",
    "tests/kernels/quantization",
    "tests/lora",
    "tests/model_executor",
    "tests/model_executor/layers",
    "tests/model_executor/model_loader",
    "tests/models",
    "tests/models/test_initialization.py",
    "tests/models/language",
    "tests/models/multimodal",
    "tests/models/quantization",
    "tests/multimodal",
    "tests/parser",
    "tests/plugins_tests/gguf",
    "tests/plugins_tests/lora_resolvers",
    "tests/plugins/bge_m3_sparse_plugin",
    "tests/plugins/prithvi_io_processor_plugin",
    "tests/plugins/vllm_add_dummy_platform",
    "tests/plugins/vllm_add_dummy_stat_logger",
    "tests/plugins_tests",
    "tests/quantization",
    "tests/reasoning",
    "tests/renderers",
    "tests/samplers",
    "tests/spec_decode",
    "tests/tokenizers_",
    "tests/tool_parsers",
    "tests/tool_use",
    "tests/transformers_utils",
    "tests/utils_",
    "tests/v1",
    "tests/v1/attention",
    "tests/v1/core",
    "tests/v1/cudagraph",
    "tests/v1/determinism",
    "tests/v1/distributed",
    "tests/v1/e2e",
    "tests/v1/ec_connector",
    "tests/v1/engine",
    "tests/v1/executor",
    "tests/v1/kv_connector",
    "tests/v1/kv_offload",
    "tests/v1/logits_processors",
    "tests/v1/metrics",
    "tests/v1/sample",
    "tests/v1/shutdown",
    "tests/v1/simple_kv_offload",
    "tests/v1/spec_decode",
    "tests/v1/streaming_input",
    "tests/v1/structured_output",
    "tests/v1/worker",
]

# TODO(woosuk): Include the code from Megatron and HuggingFace.
EXCLUDE = [
    "vllm/model_executor/models",
    "vllm/model_executor/layers/fla/ops",
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
            if changed_file.startswith(("vllm/", "tests/")):
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
    python_version = sys.argv[1]
    file_groups = group_files(sys.argv[2:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(
                changed_files, python_version, follow_imports, file_group
            )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
