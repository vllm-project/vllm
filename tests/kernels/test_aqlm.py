#import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


def test_aqlm_dequant_opcheck():
    #codes=torch.Size([22016, 512, 1]), torch.int16
    #codebooks=torch.Size([2, 65536, 1, 8]), torch.float16
    codes = torch.randint(-32768,
                          32767, (22016, 512, 1),
                          device='cuda',
                          dtype=torch.int16)
    codebooks = torch.rand((2, 65536, 1, 8),
                           device='cuda',
                           dtype=torch.float16)
    codebook_partition_sizes = [11008, 11008]

    #codes=torch.Size([4096, 1376, 1]), torch.int16
    #codebooks=torch.Size([1, 65536, 1, 8]), torch.float16
    #codebook_partition_sizes=[4096]

    opcheck(torch.ops._C.aqlm_dequant,
            (codes, codebooks, codebook_partition_sizes))


def test_aqlm_gemm_opcheck():
    #input=torch.Size([4, 4096]), torch.float16
    #codes=torch.Size([12288, 512, 1]), torch.int16
    #codebooks=torch.Size([3, 65536, 1, 8]), torch.float16
    #scales=torch.Size([12288, 1, 1, 1]), torch.float16
    #codebook_partition_sizes=[4096, 4096, 4096]

    input = torch.rand((4, 4096), device='cuda', dtype=torch.float16)
    codes = torch.randint(-32768,
                          32767, (12288, 512, 1),
                          device='cuda',
                          dtype=torch.int16)
    codebooks = torch.rand((3, 65536, 1, 8),
                           device='cuda',
                           dtype=torch.float16)
    scales = torch.rand((12288, 1, 1, 1), device='cuda', dtype=torch.float16)
    codebook_partition_sizes = [4096, 4096, 4096]
    #bias = torch.rand((1,12288,4), device='cuda', dtype=torch.float16)
    bias = None

    opcheck(torch.ops._C.aqlm_gemm,
            (input, codes, codebooks, scales, codebook_partition_sizes, None))
    opcheck(torch.ops._C.aqlm_gemm,
            (input, codes, codebooks, scales, codebook_partition_sizes, bias))
