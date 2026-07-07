#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lint vLLM C++/CUDA sources to maintain libtorch stable ABI usage.

Runs on all files under csrc/. Sources that still use the unstable libtorch
C++ ABI are listed in the temporary ignore lists below — remove entries as
ROCm, CPU, and legacy HIP builds are migrated (see CMakeLists.txt).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import regex as re

CSRC_ROOT = "csrc/"

# PyTorch headers allowed in stable-ABI code.
ALLOWED_TORCH_INCLUDE_PREFIXES = (
    "torch/csrc/stable/",
    "torch/headeronly/",
    "torch/csrc/inductor/aoti_torch/c/",
)

UNSTABLE_INCLUDE_PREFIXES = (
    "ATen/",
    "c10/",
    "torch/csrc/api/",
    "torch/csrc/autograd/",
    "torch/csrc/jit/",
)

UNSTABLE_INCLUDE_EXACT = frozenset(
    {
        "torch/all.h",
        "torch/extension.h",
        "torch/torch.h",
        "torch/cuda.h",
        "torch/library.h",
    }
)

# Temporary exemptions until ROCm / CPU / legacy HIP builds use stable libtorch ABI.
# Remove entries as each area is migrated (see CMakeLists.txt).

IGNORE_ROCM_CPU_PREFIXES = {
    "csrc/cpu/",
    "csrc/rocm/",
    "csrc/quantization/machete/generated/",
}

IGNORE_ROCM_CPU_FILES = {
    # Legacy HIP _C extension (CMakeLists.txt: VLLM_EXT_SRC when HIP).
    "csrc/torch_bindings.cpp",
    "csrc/custom_quickreduce.cu",
    "csrc/cuda_view.cu",
    "csrc/moe/dynamic_4bit_int_moe_cpu.cpp",
    # Pre-stable shared headers; stable copies live under csrc/libtorch_stable/.
    "csrc/cache.h",
    "csrc/core/registration.h",
    "csrc/dispatch_utils.h",
    "csrc/ops.h",
    "csrc/torch_utils.h",
    "csrc/quantization/w8a8/fp8/common.cuh",
}

INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*[<"]([^">]+)[">]',
    re.MULTILINE,
)

# Patterns applied to code with comments stripped.
COMMON_FORBIDDEN_CODE_PATTERNS: list[tuple[str, str]] = [
    (
        r"(?<!STABLE_)TORCH_LIBRARY\b",
        "Use STABLE_TORCH_LIBRARY / STABLE_TORCH_LIBRARY_FRAGMENT instead.",
    ),
    (
        r"(?<!STABLE_)TORCH_LIBRARY_IMPL\b",
        "Use STABLE_TORCH_LIBRARY_IMPL instead.",
    ),
    (
        r"\bPYBIND11_MODULE\b",
        "pybind11 bindings are not stable-ABI compatible. Use "
        "STABLE_TORCH_LIBRARY registration.",
    ),
    (
        r"\bAT_DISPATCH_\w+",
        "Use THO_DISPATCH_* macros from libtorch_stable/dispatch_utils.h.",
    ),
    (
        r"(?<!STD_)TORCH_CHECK\b",
        "Use STD_TORCH_CHECK from libtorch_stable/torch_utils.h.",
    ),
    (
        r"\bat::Tensor\b",
        "Use torch::stable::Tensor from torch/csrc/stable/tensor.h.",
    ),
    (
        r"\bat::ScalarType\b",
        "Use torch::headeronly::ScalarType.",
    ),
]

COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE_RE = re.compile(r"//.*")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    message: str


def _line_number(content: str, index: int) -> int:
    return content[: index + 1].count("\n") + 1


def _strip_comments(content: str) -> str:
    without_blocks = COMMENT_BLOCK_RE.sub("", content)
    return COMMENT_LINE_RE.sub("", without_blocks)


def _repo_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    idx = normalized.find(CSRC_ROOT)
    if idx != -1:
        return normalized[idx:]
    return normalized


def _is_ignored_file(path: str) -> bool:
    rel_path = _repo_relative_path(path)
    if not rel_path.startswith(CSRC_ROOT):
        return True

    for prefix in IGNORE_ROCM_CPU_PREFIXES:
        if rel_path.startswith(prefix):
            return True
    return rel_path in IGNORE_ROCM_CPU_FILES


def _is_unstable_include(header: str) -> bool:
    if header.startswith(UNSTABLE_INCLUDE_PREFIXES):
        return True
    if header in UNSTABLE_INCLUDE_EXACT:
        return True
    return header.startswith("torch/") and not header.startswith(
        ALLOWED_TORCH_INCLUDE_PREFIXES
    )


def _unstable_include_message(header: str) -> str:
    return f"Cannot include unstable header '{header}'; use stable equivalents instead."


def _check_includes(path: str, content: str) -> list[Violation]:
    violations: list[Violation] = []
    for match in INCLUDE_RE.finditer(content):
        header = match.group(1)
        if not _is_unstable_include(header):
            continue
        line = _line_number(content, match.start())
        violations.append(
            Violation(
                path=path,
                line=line,
                message=_unstable_include_message(header),
            )
        )
    return violations


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("*")


def check_file(path: str) -> list[Violation]:
    if _is_ignored_file(path):
        return []

    rel_path = _repo_relative_path(path)
    if not rel_path.startswith(CSRC_ROOT):
        return []

    with open(path, encoding="utf-8") as f:
        content = f.read()

    violations: list[Violation] = []
    violations.extend(_check_includes(path, content))

    lines = content.splitlines()
    for line_no, line in enumerate(lines, start=1):
        if _is_comment_line(line):
            continue
        for pattern, tip in COMMON_FORBIDDEN_CODE_PATTERNS:
            if re.search(pattern, line):
                violations.append(Violation(path=path, line=line_no, message=tip))

    return violations


def _print_violation(violation: Violation) -> None:
    print(
        f"{violation.path}:{violation.line}: \033[91merror:\033[0m {violation.message}"
    )


def main() -> int:
    returncode = 0
    for path in sys.argv[1:]:
        for violation in check_file(path):
            _print_violation(violation)
            returncode = 1
    return returncode


def test_patterns() -> None:
    sample = """
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/stable/tensor.h>
STABLE_TORCH_LIBRARY(_C, m) {}
TORCH_LIBRARY(_C, m) {}
// TORCH_LIBRARY comment
STD_TORCH_CHECK(true, "ok");
"""
    assert _is_unstable_include("ATen/cuda/CUDAContext.h")
    assert not _is_unstable_include("torch/csrc/stable/tensor.h")
    assert _is_ignored_file("csrc/cpu/torch_bindings.cpp")
    assert _is_ignored_file("csrc/rocm/attention.cu")
    assert not _is_ignored_file("csrc/libtorch_stable/torch_utils.h")
    stripped = _strip_comments(sample)
    assert re.search(r"(?<!STABLE_)TORCH_LIBRARY\b", stripped)
    assert not re.search(r"(?<!STABLE_)TORCH_LIBRARY\b", "STABLE_TORCH_LIBRARY(_C, m)")
    print("All stable ABI regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_patterns()
    else:
        sys.exit(main())
