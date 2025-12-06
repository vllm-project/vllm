# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility to calculate checksum of vLLM source code for detecting changes.
"""

import hashlib
import os
from pathlib import Path

# File extensions to include (meaningful source code)
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


# Directories to include at top level only
TOP_LEVEL_INCLUDE_DIRS: set[str] = {
    "cmake",
    "csrc",
    "requirements",
    "tools",
    "vllm",
}

# Directories to exclude at any level
EXCLUDE_DIRS: set[str] = {
    "__pycache__",
}


def should_exclude_dir(dir_name: str) -> bool:
    """Check if a directory should be excluded at any level."""
    # Exclude directories starting with dot
    if dir_name.startswith("."):
        return True
    # Exclude cache directories at any level
    return dir_name in EXCLUDE_DIRS


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum for a single file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def calculate_vllm_source_factors() -> dict[str, str]:
    """
    Calculate checksum of vLLM source code.
    """
    # Get the vLLM root directory by looking for setup.py
    current = Path(__file__).resolve()
    root_path = None
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "vllm").is_dir():
            root_path = parent
            break

    if root_path is None:
        raise RuntimeError("Could not find vLLM root directory")

    checksums: dict[str, str] = {}

    # Pre-compute root path as string for faster operations
    root_str = str(root_path) + os.sep
    root_str_len = len(root_str)

    # Scan all files in root directory
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue

        file_str = str(file_path)
        assert file_str.startswith(root_str)
        relative_str = file_str[root_str_len:]
        relative_str = relative_str.replace(os.sep, "/")

        # Quick extension check
        _, ext = os.path.splitext(relative_str)
        if ext not in INCLUDE_EXTENSIONS:
            continue

        parts = relative_str.split(os.sep)

        # Check if the top-level directory should be excluded
        if len(parts) > 0:
            top_level_dir = parts[0]
            if top_level_dir not in TOP_LEVEL_INCLUDE_DIRS:
                continue

        # Check if any part of the path contains excluded directories
        if any(should_exclude_dir(part) for part in parts):
            continue

        checksums[relative_str] = calculate_file_checksum(file_path)

    return dict(sorted(checksums.items()))


def calculate_vllm_source_checksum(factors: dict[str, str]) -> str:
    factors_str = "".join(f"{path}:{checksum}" for path, checksum in factors.items())
    return hashlib.sha256(factors_str.encode()).hexdigest()


def print_vllm_source_checksum() -> None:
    """Calculate and print the vLLM source code checksum."""
    factors = calculate_vllm_source_factors()
    checksum = calculate_vllm_source_checksum(factors)
    print(f"vLLM source checksum: {checksum}")


if __name__ == "__main__":
    print_vllm_source_checksum()
