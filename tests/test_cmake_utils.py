# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _get_cmake_bin() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    venv_cmake = Path(sys.executable).parent / "cmake"
    if venv_cmake.is_file():
        return str(venv_cmake)
    return "cmake"


def test_exact_family_arch_precedes_generic_family_fallback(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_cuda_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(
  actual "10.0f;10.7f" "10.7")
if(NOT "${{actual}}" STREQUAL "10.7f")
  message(FATAL_ERROR "Expected 10.7f, got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def _check_loose_intersection(
    tmp_path: Path, src_archs: str, tgt_archs: str, expected: str
):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_cuda_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(actual "{src_archs}" "{tgt_archs}")
if(NOT "${{actual}}" STREQUAL "{expected}")
  message(FATAL_ERROR "Expected '{expected}', got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


@pytest.mark.parametrize(
    ("src_archs", "tgt_archs", "expected"),
    [
        pytest.param(
            "10.7f", "7.5;8.0;8.6;8.9;9.0;10.0;12.0", "", id="cu134-release-list"
        ),
        pytest.param("10.7f", "10.0a", "", id="lower-minor-arch-specific"),
        pytest.param("12.1f", "12.0", "", id="lower-minor-plain"),
        pytest.param("10.3f", "10.0a;10.7a", "10.7a", id="skips-lower-minor"),
        pytest.param("12.0f", "12.1a", "12.1a", id="higher-minor"),
    ],
)
def test_family_arch_fallback_never_picks_lower_minor(
    tmp_path: Path, src_archs: str, tgt_archs: str, expected: str
):
    _check_loose_intersection(tmp_path, src_archs, tgt_archs, expected)


@pytest.mark.parametrize("src_archs", ["10.0f;10.7f", "10.7f;10.0f"])
def test_family_arch_fallback_runs_after_name_matches(tmp_path: Path, src_archs: str):
    _check_loose_intersection(tmp_path, src_archs, "10.7a;10.3a", "10.7a;10.3a")


def test_extract_archs_prefers_sass_target_over_corrupted_virtual_arch(
    tmp_path: Path,
):
    """Torch's autodetection can emit a bogus arch=compute_* half (e.g.
    capability 12.1 corrupted to arch=compute_20,code=sm_121); the SASS
    target must win, while PTX-only entries keep the virtual arch."""
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_extract_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
extract_unique_cuda_archs_ascending(actual
  "-gencode arch=compute_20,code=sm_121;\
-gencode arch=compute_80,code=sm_80;\
-gencode arch=compute_80,code=compute_80")
if(NOT "${{actual}}" STREQUAL "8.0;12.1")
  message(FATAL_ERROR "Expected '8.0;12.1', got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_inkling2_kernels_include_hopper(tmp_path: Path):
    """Regression test for https://github.com/vllm-project/vllm/issues/69046.

    The vllm-openai:inkling-2-mini image shipped the Inkling2 kernels
    (csrc/inkling2/conv_mix.cu, allreduce_conv_mix.cu, seq_parallel.cu) only
    for sm_100: the CMakeLists.txt arch list had no 9.0a entry, so on H100
    (sm_90) launches failed with "no kernel image is available for execution
    on the device". Hopper support was added in #67880; pin the arch lists in
    CMakeLists.txt so that every standard Docker/release build (whose
    TORCH_CUDA_ARCH_LIST includes 9.0) still yields a 9.0a cubin.
    """
    repo_root = Path(__file__).parents[1]
    cmake_lists = (repo_root / "CMakeLists.txt").read_text()
    marker = "cuda_archs_loose_intersection(INKLING2_KERNEL_ARCHS"
    src_lists = []
    start = 0
    while True:
        i = cmake_lists.find(marker, start)
        if i < 0:
            break
        q1 = cmake_lists.index('"', i)
        q2 = cmake_lists.index('"', q1 + 1)
        src_lists.append(cmake_lists[q1 + 1 : q2])
        start = q2 + 1
    assert src_lists, "INKLING2_KERNEL_ARCHS block not found in CMakeLists.txt"
    # Standard build arch lists (docker/Dockerfile, docker/versions.json,
    # .buildkite/release-pipeline.yaml) all include 9.0.
    tgt_lists = (
        "7.5;8.0;8.6;8.9;9.0;10.0;11.0;12.0",  # Dockerfile default
        "7.5;8.0;8.6;8.9;9.0;10.0;12.0",  # release pipeline x86_64
        "8.0;8.7;8.9;9.0;10.0;11.0;12.0",  # release pipeline aarch64
    )
    for src_archs in src_lists:
        for tgt_archs in tgt_lists:
            script = tmp_path / "test_inkling2_archs.cmake"
            script.write_text(
                f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(actual "{src_archs}" "{tgt_archs}")
if(NOT "9.0a" IN_LIST actual)
  message(FATAL_ERROR "Inkling2 kernels miss Hopper (9.0a) for target list '{tgt_archs}': got '${{actual}}'")
endif()
"""
            )
            subprocess.run([_get_cmake_bin(), "-P", script], check=True)

def test_clear_cuda_gencode_flags(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_clear_flags.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
set(CMAKE_CUDA_FLAGS "-Wall -gencode arch=compute_80,code=sm_80")
clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
if(NOT "${{CMAKE_CUDA_FLAGS}}" STREQUAL "-Wall ")
  message(FATAL_ERROR "Expected '-Wall ', got '${{CMAKE_CUDA_FLAGS}}'")
endif()
if(NOT "${{CUDA_ARCH_FLAGS}}" STREQUAL "-gencode arch=compute_80,code=sm_80")
  message(FATAL_ERROR "Expected '-gencode arch=compute_80,code=sm_80', "
    "got '${{CUDA_ARCH_FLAGS}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_hipify_records_original_sources_and_headers_without_modifying_them(tmp_path):
    project = tmp_path / "source"
    output = tmp_path / "build" / "csrc"
    project.mkdir()
    source = project / "probe.cu"
    header = project / "cuda_shared.h"
    source.write_text('#include "cuda_shared.h"\n__global__ void probe() {}\n')
    header.write_text("#include <cuda_runtime.h>\n")
    original = {path.name: path.read_bytes() for path in project.iterdir()}
    script = Path(__file__).parents[1] / "cmake" / "hipify.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "-p",
            str(project),
            "-o",
            str(output),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert {path.name: path.read_bytes() for path in project.iterdir()} == original
    provenance = json.loads((output / "hipify-source-map.json").read_text())
    assert provenance[str(output / "probe.hip")] == str(source)
    assert provenance[str(output / "cuda_shared.h")] == str(header)
    assert provenance[str(output / "hip_shared.h")] == str(header)
    assert '#include "hip_shared.h"' in (output / "probe.hip").read_text()
