# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile AMD architecture checks without importing HIP or using a GPU."""

import os
import shlex
import subprocess
from pathlib import Path

import regex as re

ROOT = Path(__file__).parents[3]


def compile_cpp(source: Path) -> None:
    result = subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic-errors",
            "-fsyntax-only",
            "-I",
            str(ROOT / "csrc"),
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_architecture_mapping_and_parser_contract():
    compile_cpp(Path(__file__).with_suffix(".cpp"))


def test_all_cmake_hip_targets_have_an_architecture(tmp_path: Path):
    match = re.search(
        r'set\(HIP_SUPPORTED_ARCHS "([^"]+)"\)',
        (ROOT / "CMakeLists.txt").read_text(),
    )
    assert match is not None
    source = tmp_path / "cmake_hip_targets.cpp"
    source.write_text(
        '#include "core/amd_arch.h"\nusing namespace vllm::amd;\n'
        + "\n".join(
            f'static_assert(parse_gcn_arch("{target}").architecture != '
            "GPUArchitecture::UNKNOWN);"
            for target in match[1].split(";")
        )
    )
    compile_cpp(source)
