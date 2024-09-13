"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform


@pytest.mark.smoke
def test_utils_opcheck():
    if current_platform.is_cuda():
        opcheck(torch.ops._C_cuda_utils.get_device_attribute, (0, 0))
        opcheck(
            torch.ops._C_cuda_utils.
            get_max_shared_memory_per_block_device_attribute, (0, ))
