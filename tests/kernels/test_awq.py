# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


@pytest.mark.skipif(not hasattr(torch.ops._C, "awq_dequantize"),
                    reason="AWQ is not supported on this GPU type.")
def test_awq_dequantize_opcheck():
    os.environ["VLLM_USE_TRITON_AWQ"] = "0"
    qweight = torch.randint(-2000000000,
                            2000000000, (8192, 256),
                            device='cuda',
                            dtype=torch.int32)
    scales = torch.rand((64, 2048), device='cuda', dtype=torch.float16)
    zeros = torch.empty((64, 256), device='cuda', dtype=torch.int32)
    split_k_iters = 0
    thx = 0
    thy = 0
    opcheck(torch.ops._C.awq_dequantize,
            (qweight, scales, zeros, split_k_iters, thx, thy))


@pytest.mark.skipif(not hasattr(torch.ops._C, "awq_gemm"),
                    reason="AWQ is not supported on this GPU type.")
def test_awq_gemm_opcheck():
    os.environ["VLLM_USE_TRITON_AWQ"] = "0"
    input = torch.rand((2, 8192), device='cuda', dtype=torch.float16)
    qweight = torch.randint(-2000000000,
                            2000000000, (8192, 256),
                            device='cuda',
                            dtype=torch.int32)
    scales = torch.randint(-2000000000,
                           2000000000, (64, 256),
                           device='cuda',
                           dtype=torch.int32)
    qzeros = torch.empty((64, 2048), device='cuda', dtype=torch.float16)
    split_k_iters = 8
    opcheck(torch.ops._C.awq_gemm,
            (input, qweight, qzeros, scales, split_k_iters))
