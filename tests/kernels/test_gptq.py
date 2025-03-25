# SPDX-License-Identifier: Apache-2.0

import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


def test_gptq_shuffle_opcheck():
    weight = torch.randint(-2000000,
                           2000000, (1792, 4096),
                           device='cuda',
                           dtype=torch.int32)
    perm = torch.empty((0, ), device='cuda', dtype=torch.int32)
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, perm, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device='cuda', dtype=torch.float16)
    weight = torch.randint(-2000000,
                           2000000, (512, 6144),
                           device='cuda',
                           dtype=torch.int32)
    zeros = torch.zeros((32, 768), device='cuda', dtype=torch.int32)
    scales = torch.rand((32, 6144), device='cuda', dtype=torch.float16)
    idx = torch.empty((0, ), device='cuda', dtype=torch.int32)
    use_exllama = True
    bit = 4
    opcheck(torch.ops._C.gptq_gemm,
            (a, weight, zeros, scales, idx, use_exllama, bit))
