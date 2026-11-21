# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "tools/pre_commit/check_torch_cuda.py"


def _run_check(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "source.py"
    path.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECK), str(path)],
        capture_output=True,
        text=True,
    )


def test_non_importable_cuda_apis_are_reported(tmp_path):
    result = _run_check(
        tmp_path,
        "free, total = current_platform.mem_get_info()\n"
        "count = cuda_device_count_stateless()\n",
    )

    assert result.returncode == 1
    assert "current_platform.mem_get_info()" in result.stdout
    assert "cuda_device_count_stateless()" in result.stdout


def test_non_code_text_is_ignored(tmp_path):
    result = _run_check(
        tmp_path,
        '"""current_platform.mem_get_info()"""\n'
        'message = "cuda_device_count_stateless()"\n'
        "# current_platform.mem_get_info()\n",
    )

    assert result.returncode == 0
    assert result.stdout == ""


def test_replacements_are_accepted(tmp_path):
    result = _run_check(
        tmp_path,
        "free, total = torch.accelerator.get_memory_info()\n"
        "count = current_platform.device_count()\n",
    )

    assert result.returncode == 0
    assert result.stdout == ""
