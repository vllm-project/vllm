# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility to calculate checksum of vLLM source code for detecting changes.
"""

import hashlib
import os
from pathlib import Path

INCLUDE_EXTENSIONS: set[str] = {
    ".py",
    ".pyx",
    ".pxd",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".hxx",
    ".cmake",
    ".txt",
    ".toml",
}


TOP_LEVEL_INCLUDE_DIRS: set[str] = {
    "cmake",
    "csrc",
    "requirements",
    "tools",
    "vllm",
}

EXCLUDE_DIRS: set[str] = {
    "__pycache__",
}


def should_exclude_dir(dir_name: str) -> bool:
    if dir_name.startswith("."):
        return True
    return dir_name in EXCLUDE_DIRS


def calculate_file_checksum(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def calculate_vllm_source_factors() -> dict[str, str]:
    current = Path(__file__).resolve()
    root_path = None
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "vllm").is_dir():
            root_path = parent
            break

    if root_path is None:
        raise RuntimeError("Could not find vLLM root directory")

    checksums: dict[str, str] = {}

    root_str = str(root_path) + os.sep
    root_str_len = len(root_str)

    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue

        file_str = str(file_path)
        assert file_str.startswith(root_str)
        relative_str = file_str[root_str_len:]
        relative_str = relative_str.replace(os.sep, "/")

        _, ext = os.path.splitext(relative_str)
        if ext not in INCLUDE_EXTENSIONS:
            continue

        parts = relative_str.split(os.sep)

        if len(parts) > 0:
            top_level_dir = parts[0]
            if top_level_dir not in TOP_LEVEL_INCLUDE_DIRS:
                continue

        if any(should_exclude_dir(part) for part in parts):
            continue

        checksums[relative_str] = calculate_file_checksum(file_path)

    return dict(sorted(checksums.items()))


def calculate_vllm_source_checksum(factors: dict[str, str]) -> str:
    factors_str = "".join(f"{path}:{checksum}" for path, checksum in factors.items())
    return hashlib.sha256(factors_str.encode()).hexdigest()


def print_vllm_source_checksum() -> None:
    factors = calculate_vllm_source_factors()
    checksum = calculate_vllm_source_checksum(factors)
    print(f"vLLM source checksum: {checksum}")


if __name__ == "__main__":
    print_vllm_source_checksum()
