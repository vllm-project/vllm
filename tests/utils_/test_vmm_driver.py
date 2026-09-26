# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest
import torch

from vllm.utils.vmm_driver import _find_loaded_library

_CUDART = "/usr/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
_MAPS = [
    f"7f00-7f01 r-xp 00000000 08:01 1 {_CUDART}",
    "7f02-7f03 r-xp 00000000 08:01 2 [heap]",
    "7f04-7f05 r-xp 00000000 08:01 3 /usr/lib64/libcuda.so.580.65.06",
    "7f06-7f07 r-xp 00000000 08:01 4 /usr/lib64/libcuda.so.1",
]


def test_find_loaded_library_matches_the_file_name_not_a_substring():
    """Torch maps libcudart before the driver; the driver lookup must not pick
    it (its symbols differ), nor match a path component."""
    assert _find_loaded_library("libcuda", _MAPS) == "/usr/lib64/libcuda.so.580.65.06"
    assert _find_loaded_library("libcudart", _MAPS) == _CUDART
    assert _find_loaded_library("libamdhip64", _MAPS) is None
    assert _find_loaded_library("libcuda", ["garbage"]) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA driver context handling",
)
def test_cuda_driver_context_usable_before_torch_initializes_the_device():
    """The VMM probe may run before torch touches the device; the driver must
    initialize itself rather than fail with a bare 'not initialized' error."""
    script = """
from vllm.utils.vmm_driver import get_vmm_driver
driver = get_vmm_driver()
driver.ensure_context(0)
driver.ensure_context(0)
print(driver.granularity(0))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.strip().splitlines()[-1]) > 0
