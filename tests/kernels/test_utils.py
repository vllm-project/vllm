"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform


def test_convert_fp8_opcheck():
    data = torch.randn((10, 15), dtype=torch.float32, device="cuda")
    result = torch.empty_like(data, dtype=torch.float8_e4m3fn)
    opcheck(torch.ops._C_cache_ops.convert_fp8, (result, data, 1.0, "fp8"))


def test_cuda_utils_opcheck():
    if current_platform.is_cuda():
        opcheck(torch.ops._C_cuda_utils.get_device_attribute, (0, 0))
        opcheck(
            torch.ops._C_cuda_utils.
            get_max_shared_memory_per_block_device_attribute, (0, ))
