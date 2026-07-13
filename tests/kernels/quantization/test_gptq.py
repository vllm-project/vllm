# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    perm = torch.empty((0,), device="cuda", dtype=torch.int32)
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, perm, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    idx = torch.empty((0,), device="cuda", dtype=torch.int32)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, True, bit)
    )
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, False, bit)
    )


# --- Validation tests for GHSA-4hhp-h66f-j5j7 ---


@pytest.mark.parametrize("invalid_bit", [0, 1, 5, 6, 7, 16])
def test_gptq_gemm_invalid_bit(invalid_bit):
    """bit must be in {2, 3, 4, 8}; other values must raise."""
    a = torch.rand((8, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(0, 2**31 - 1, (512, 6144), device="cuda", dtype=torch.int32)
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    idx = torch.empty((0,), device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="gptq_gemm"):
        torch.ops._C.gptq_gemm(a, weight, zeros, scales, idx, True, True, invalid_bit)


def test_gptq_gemm_zero_groups():
    """groups == 0 (b_gptq_qzeros.size(0) == 0) must raise."""
    a = torch.rand((8, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(0, 2**31 - 1, (512, 6144), device="cuda", dtype=torch.int32)
    zeros = torch.zeros((0, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((0, 6144), device="cuda", dtype=torch.float16)
    idx = torch.empty((0,), device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="gptq_gemm"):
        torch.ops._C.gptq_gemm(a, weight, zeros, scales, idx, True, True, 4)


def test_gptq_gemm_g_idx_oob_group_index():
    """g_idx with value >= groups when use_exllama=False must raise."""
    size_k = 4096
    size_n = 6144
    groups = 32
    bit = 4

    a = torch.rand((8, size_k), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        0,
        2**31 - 1,
        (size_k * bit // 32, size_n),
        device="cuda",
        dtype=torch.int32,
    )
    zeros = torch.zeros((groups, size_n // 8), device="cuda", dtype=torch.int32)
    scales = torch.rand((groups, size_n), device="cuda", dtype=torch.float16)

    g_idx = torch.zeros(size_k, device="cuda", dtype=torch.int32)
    g_idx[0] = groups  # OOB: equals groups (valid range is [0, groups))

    with pytest.raises(RuntimeError, match="gptq_gemm"):
        torch.ops._C.gptq_gemm(a, weight, zeros, scales, g_idx, False, True, bit)


def test_gptq_gemm_g_idx_negative():
    """g_idx with negative value must raise."""
    size_k = 4096
    size_n = 6144
    groups = 32
    bit = 4

    a = torch.rand((8, size_k), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        0,
        2**31 - 1,
        (size_k * bit // 32, size_n),
        device="cuda",
        dtype=torch.int32,
    )
    zeros = torch.zeros((groups, size_n // 8), device="cuda", dtype=torch.int32)
    scales = torch.rand((groups, size_n), device="cuda", dtype=torch.float16)

    g_idx = torch.zeros(size_k, device="cuda", dtype=torch.int32)
    g_idx[100] = -1

    with pytest.raises(RuntimeError, match="gptq_gemm"):
        torch.ops._C.gptq_gemm(a, weight, zeros, scales, g_idx, False, True, bit)


def test_gptq_gemm_g_idx_oob_permutation():
    """g_idx with value >= size_k when use_exllama=True must raise."""
    size_k = 4096
    size_n = 6144
    groups = 32
    bit = 4

    a = torch.rand((8, size_k), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        0,
        2**31 - 1,
        (size_k * bit // 32, size_n),
        device="cuda",
        dtype=torch.int32,
    )
    zeros = torch.zeros((groups, size_n // 8), device="cuda", dtype=torch.int32)
    scales = torch.rand((groups, size_n), device="cuda", dtype=torch.float16)

    g_idx = torch.arange(size_k, device="cuda", dtype=torch.int32)
    g_idx[0] = size_k  # OOB: equals size_k (valid range is [0, size_k))

    with pytest.raises(RuntimeError, match="gptq_gemm"):
        torch.ops._C.gptq_gemm(a, weight, zeros, scales, g_idx, True, True, bit)
