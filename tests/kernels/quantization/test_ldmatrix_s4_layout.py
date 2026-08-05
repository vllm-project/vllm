# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
import torch


def _ldmatrix_s4_target() -> str | None:
    if not torch.cuda.is_available():
        return None

    capability = torch.cuda.get_device_capability()
    return {
        (9, 0): "90a",
        (10, 0): "100f",
        (10, 3): "100f",
        (10, 7): "107f",
        (11, 0): "110f",
        (12, 0): "120f",
        (12, 1): "120f",
    }.get(capability)


def _nvcc_version(nvcc: str) -> tuple[int, int] | None:
    result = subprocess.run(
        [nvcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@pytest.mark.skipif(
    _ldmatrix_s4_target() is None,
    reason="Requires a GPU with CUDA 13.4 ldmatrix.s8.s4 support.",
)
def test_ldmatrix_s4_layout_and_mma_mapping(tmp_path: Path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    assert nvcc is not None
    if (_nvcc_version(nvcc) or (0, 0)) < (13, 4):
        pytest.skip("CUDA 13.4 or newer is required")

    source = Path(__file__).with_name("ldmatrix_s4_layout.cu")
    executable = tmp_path / "ldmatrix_s4_layout"
    target = _ldmatrix_s4_target()
    assert target is not None
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            f"-gencode=arch=compute_{target},code=sm_{target}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run(
        [executable],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "layout and MMA operand-B mapping passed" in result.stdout
