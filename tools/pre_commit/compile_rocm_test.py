#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrapper around `uv pip compile` for ROCm test requirements.

Runs the compile, then strips CUDA/NVIDIA packages from the output using
pattern matching so the list doesn't need manual updating every time a new
CUDA dependency appears.  Emits a warning for every package that was
dynamically filtered so maintainers stay aware of what's being excluded.
"""

import subprocess
import sys

import regex as re

# Packages that are explicitly excluded because they are ROCm-incompatible
# for reasons *other* than being CUDA/NVIDIA vendored libraries.
EXPLICIT_EXCLUDES = [
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
]

# Patterns that match CUDA/NVIDIA vendored packages.
# Any resolved package whose name matches one of these is stripped from the
# output file automatically.
CUDA_NVIDIA_PATTERNS = [
    re.compile(r"^cuda-"),
    re.compile(r"^cupy-cuda"),
    re.compile(r"^nvidia-"),
]


def _matches_cuda_nvidia(package_name: str) -> bool:
    return any(p.search(package_name) for p in CUDA_NVIDIA_PATTERNS)


def _package_name_from_line(line: str) -> str | None:
    """Extract the package name from a requirements-txt line like 'foo==1.2'."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = re.match(r"^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)", stripped)
    return match.group(1).lower() if match else None


def main() -> int:
    uv_args = [
        "uv",
        "pip",
        "compile",
        "requirements/rocm-test.in",
        "-o",
        "requirements/rocm-test.txt",
        "--index-strategy",
        "unsafe-best-match",
        "-c",
        "requirements/rocm.txt",
        "--python-platform",
        "x86_64-manylinux_2_28",
        "--python-version",
        "3.12",
    ]

    # Add the explicit (non-pattern) excludes.
    for pkg in EXPLICIT_EXCLUDES:
        uv_args += ["--no-emit-package", pkg]

    result = subprocess.run(uv_args, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stdout.write(result.stdout)
        return result.returncode

    # Post-process: filter CUDA/NVIDIA packages from the output file.
    output_path = "requirements/rocm-test.txt"
    with open(output_path) as f:
        lines = f.readlines()

    filtered_lines: list[str] = []
    filtered_packages: list[str] = []

    for line in lines:
        pkg_name = _package_name_from_line(line)
        if pkg_name and _matches_cuda_nvidia(pkg_name):
            filtered_packages.append(pkg_name)
        else:
            filtered_lines.append(line)

    if filtered_packages:
        print(
            f"WARNING: Dynamically filtered {len(filtered_packages)} "
            "CUDA/NVIDIA package(s) from rocm-test.txt:",
            file=sys.stderr,
        )
        for pkg in sorted(filtered_packages):
            print(f"  - {pkg}", file=sys.stderr)

    with open(output_path, "w") as f:
        f.writelines(filtered_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
