# SPDX-License-Identifier: Apache-2.0
"""
Tests for miscellaneous utilities
"""

import torch

from tests.kernels.utils import opcheck


def test_convert_fp8_opcheck():
    data = torch.randn((256, 256), dtype=torch.float32, device="cuda")
    result = torch.empty_like(data, dtype=torch.float8_e4m3fn)
    opcheck(torch.ops._C_cache_ops.convert_fp8, (result, data, 1.0, "fp8"))


# TODO: Add this back, currently fails with
# csrc/cuda_utils_kernels.cu:15 'invalid argument'
# @pytest.mark.skipif(not current_platform.is_cuda(),
#                     reason="Only supported for CUDA")
# def test_cuda_utils_opcheck():
#     opcheck(torch.ops._C_cuda_utils.get_device_attribute, (0, 0))
#     opcheck(
#         torch.ops._C_cuda_utils.
#         get_max_shared_memory_per_block_device_attribute, (0, ))
