"""Test HQQ with Marlin kernel.

Run `pytest tests/kernels/marlin/test_marlin_hqq.py`.
"""
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    marlin_make_empty_g_idx, marlin_permute_scales)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from vllm.scalar_type import scalar_types

USE_FP32_REDUCE_OPTS = [False, True]

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]
HQQ_SUPPORTED_GROUP_SIZES = [64]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(not is_quant_method_supported("gptq_marlin"),
                    reason="Marlin is not supported on this GPU type.")
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("group_size", HQQ_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("use_fp32_reduce", USE_FP32_REDUCE_OPTS)
def test_hqq_marlin_gemm(
    k_chunk,
    n_chunk,
    group_size,
    mnk_factors,
    use_fp32_reduce,
):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    quant_type = scalar_types.uint4

    a_input = rand_data((size_m, size_k))
    dev = a_input.device

    b_weight = torch.randint(0,
                             10, (size_n, size_k),
                             dtype=torch.uint8,
                             device=dev)
    scale = rand_data((size_n, size_k // group_size))
    zero = rand_data((size_n, size_k // group_size))

    gptq_w_q = gptq_pack(b_weight.transpose(1, 0), 4, size_k, size_n)

    sort_indices = torch.empty(0, dtype=torch.int, device=dev)
    marlin_w_q = ops.gptq_marlin_repack(gptq_w_q, sort_indices, size_k, size_n,
                                        4).to(dev)
    marlin_s = marlin_permute_scales(scale.transpose(1, 0), size_k, size_n,
                                     group_size).to(dev)
    marlin_zp = marlin_permute_scales(zero.transpose(1, 0), size_k, size_n,
                                      group_size).to(dev)

    g_idx = marlin_make_empty_g_idx(dev)
    g_idx_sort_indices = marlin_make_empty_g_idx(dev)

    workspace = MarlinWorkspace(size_n, GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    output = ops.gptq_marlin_gemm(
        a_input,
        marlin_w_q,
        marlin_s,
        marlin_zp,
        g_idx,
        g_idx_sort_indices,
        workspace.scratch,
        quant_type,
        a_input.shape[0],
        b_weight.shape[0],
        a_input.shape[1],
        is_k_full=True,
        has_zp=True,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=True,
    )

    b_flat = b_weight.reshape(-1, group_size)
    zp_flat = zero.reshape(-1, 1)
    s_flat = scale.reshape(-1, 1)
    dequant = (b_flat - zp_flat) * s_flat

    output_ref = torch.matmul(a_input,
                              dequant.reshape(b_weight.shape).transpose(1, 0))

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04
