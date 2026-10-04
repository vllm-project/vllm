# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Direct-source regression for the ARM SHM copy tail boundary.

Apple Silicon does not build `csrc/cpu/shm.cpp` into the normal CPU extension,
so this test compiles the exact source file under ASan instead of exercising an
installed operator path. That preserves the vulnerable native boundary while
keeping the regression runnable on the host that reproduced the bug.
"""

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import include_paths, library_paths

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARNESS = Path(__file__).with_name("shm_tail_guard_harness.cpp")
_DARWIN_OPENMP_INSTALL_NAME = "/opt/llvm-openmp/lib/libomp.dylib"

if not (_REPO_ROOT / "csrc/cpu/shm.cpp").exists():
    pytest.skip("C/C++ source tree not available", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    platform.machine() not in {"arm64", "aarch64"},
    reason="The direct SHM tail harness currently targets the ARM vector backend.",
)


def _compiler_command() -> list[str]:
    command = shlex.split(os.environ.get("CXX", "clang++"))
    if not command or shutil.which(command[0]) is None:
        pytest.skip("A C++ compiler is required for the SHM tail harness")
    return command


def _run_checked(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"command failed: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _rewrite_darwin_openmp_install_name(binary: Path) -> None:
    if sys.platform != "darwin":
        return
    otool = shutil.which("otool")
    install_name_tool = shutil.which("install_name_tool")
    if otool is None or install_name_tool is None:
        pytest.skip("otool and install_name_tool are required on Darwin")
    assert otool is not None
    assert install_name_tool is not None

    linked_libraries = _run_checked([otool, "-L", str(binary)], _REPO_ROOT)
    if _DARWIN_OPENMP_INSTALL_NAME not in linked_libraries.stdout:
        return
    _run_checked(
        [
            install_name_tool,
            "-change",
            _DARWIN_OPENMP_INSTALL_NAME,
            "@rpath/libomp.dylib",
            str(binary),
        ],
        _REPO_ROOT,
    )


@pytest.fixture(scope="module")
def shm_tail_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary = tmp_path_factory.mktemp("shm-tail") / "shm_tail_guard_harness"
    command = [
        *_compiler_command(),
        "-std=c++17",
        "-O0",
        "-g",
        "-fsanitize=address",
        "-fno-omit-frame-pointer",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
        "-Icsrc",
        "-I.",
        *[f"-I{path}" for path in include_paths(device_type="cpu")],
        str(_HARNESS),
        *[f"-L{path}" for path in library_paths(device_type="cpu")],
        *[f"-Wl,-rpath,{path}" for path in library_paths(device_type="cpu")],
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(binary),
    ]
    if sys.platform == "darwin":
        command.append("-lomp")
    _run_checked(command, _REPO_ROOT)
    _rewrite_darwin_openmp_install_name(binary)
    return binary


@pytest.mark.parametrize(
    ("bytes_to_copy", "guard_mode"),
    [
        (1, "guard-src"),
        (1, "guard-dst"),
        (64, "guard-src"),
        (65, "guard-src"),
        (65, "guard-dst"),
        (100, "guard-src"),
        (100, "guard-dst"),
    ],
)
def test_memcpy_to_shm_handles_tail_without_page_fault(
    shm_tail_harness: Path, bytes_to_copy: int, guard_mode: str
) -> None:
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "halt_on_error=1:detect_leaks=0"
    result = subprocess.run(
        [str(shm_tail_harness), str(bytes_to_copy), guard_mode],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"SHM tail harness failed for {bytes_to_copy} bytes in {guard_mode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
