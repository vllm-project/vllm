# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
import torch


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


@pytest.mark.marlin_ldmatrix_s4
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires a Hopper GPU.",
)
def test_ldmatrix_s4_layout_and_mma_mapping(tmp_path: Path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not available")
    assert nvcc is not None
    if (_nvcc_version(nvcc) or (0, 0)) < (13, 4):
        pytest.skip("CUDA 13.4 or newer is required")

    source = Path(__file__).with_name("marlin_ldmatrix_s4_layout.cu")
    executable = tmp_path / "marlin_ldmatrix_s4_layout"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_90a,code=sm_90a",
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
