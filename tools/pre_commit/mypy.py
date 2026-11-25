# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g., "3.11") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.

"""

import subprocess
import sys

import regex as re

# Paths verified clean under follow_imports="silent". Matched before
# SEPARATE_GROUPS, so these files join the default group and are checked at the
# stricter setting even while a parent directory remains in SEPARATE_GROUPS.
#
# Fixing a directory means moving it from SEPARATE_GROUPS to here. Without that
# move the fixes are not enforced because "tests" claims every file below it.
SILENT_GROUPS = [
    "tests/benchmarks",
    "tests/compile",
    "tests/config",
    "tests/distributed",
    "tests/entrypoints/anthropic",
    "tests/entrypoints/generate",
    "tests/entrypoints/llm",
    "tests/entrypoints/multimodal",
    "tests/entrypoints/openai",
    "tests/entrypoints/pooling",
    "tests/entrypoints/rl",
    "tests/entrypoints/serve",
    "tests/entrypoints/speech_to_text",
    "tests/entrypoints/tool_parsers",
    "tests/entrypoints/unit_tests",
    "tests/entrypoints/common",
    "tests/kernels",
    "tests/lora",
    "tests/model_executor",
    "tests/models",
    "tests/multimodal",
    "tests/parser",
    "tests/plugins/bge_m3_sparse_plugin",
    "tests/plugins/prithvi_io_processor_plugin",
    "tests/plugins/vllm_add_dummy_platform",
    "tests/plugins/vllm_add_dummy_stat_logger",
    "tests/plugins_tests",
    "tests/quantization",
    "tests/reasoning",
    "tests/renderers",
    "tests/samplers",
    "tests/tokenizers_",
    "tests/tool_parsers",
    "tests/tool_use",
    "tests/transformers_utils",
    "tests/utils_",
    "tests/v1",
    "tests/__init__.py",
    "tests/ci_envs.py",
    "tests/conftest.py",
    "tests/single_rank_group.py",
    "tests/test_attention_backend_registry.py",
    "tests/test_audio_media_size_precheck.py",
    "tests/test_cmake_utils.py",
    "tests/test_config.py",
    "tests/test_connections.py",
    "tests/test_embedded_commit.py",
    "tests/test_envs.py",
    "tests/test_force_first_config.py",
    "tests/test_logger.py",
    "tests/test_logprobs.py",
    "tests/test_outputs.py",
    "tests/test_pcp_dp.py",
    "tests/test_pooling_params.py",
    "tests/test_quantization_revision_pin.py",
    "tests/test_ray_env.py",
    "tests/test_ray_env_utils.py",
    "tests/test_regression.py",
    "tests/test_request_input_bounds.py",
    "tests/test_sampling_params.py",
    "tests/test_scalartype.py",
    "tests/test_sequence.py",
    "tests/test_tensorizer_revision_pin.py",
    "tests/test_triton_utils.py",
    "tests/test_uvicorn_logging.py",
    "tests/test_version.py",
    "tests/test_vllm_port.py",
    "tests/test_zen_cpu_platform_detection.py",
    "tests/test_zentorch_version_pin.py",
    "tests/utils.py",
]

# After fixing errors resulting from changing follow_imports
# from "skip" to "silent", move its directory to SILENT_GROUPS.
SEPARATE_GROUPS = [
    "tests",
]


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """Group changed files into different mypy calls.

    Args:
        changed_files: List of changed files.

    Returns:
        A dictionary mapping file group names to lists of changed files.

    """
    silent_pattern = re.compile(f"^({'|'.join(SILENT_GROUPS)}).*")
    file_groups: dict[str, list[str]] = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    # Longest path first so a sub-directory is not shadowed by its parent
    separate_groups = sorted(SEPARATE_GROUPS, key=len, reverse=True)
    for changed_file in changed_files:
        # Already-fixed paths go in the default group, which runs at the
        # stricter follow_imports setting from pyproject.toml
        if silent_pattern.match(changed_file):
            file_groups[""].append(changed_file)
            continue
        # Group files by mypy call
        for directory in separate_groups:
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
    """Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.11") or None to use
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
