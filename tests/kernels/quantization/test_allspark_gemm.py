# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_K_ALIGN, ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
    ALLSPARK_AMPERE_N_ALIGN)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def is_gptq_allspark_supported(min_capability: int,
                               max_capability: int) -> bool:
    if not current_platform.is_cuda():
        return False

    capability = current_platform.get_device_capability()
    assert capability is not None

    return capability.to_int() >= min_capability \
        and capability.to_int() <= max_capability


MNK_FACTORS = [
    (1, 4, 8),
    (13, 17, 67),
    (26, 37, 13),
    (48, 16, 24),
    (67, 13, 88),
    (257, 13, 11),
    (658, 13, 11),
    (1033, 9, 17),
]

DTYPES = [torch.float16, torch.bfloat16]
HAS_ZP_OPTS = [False, True]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(
    not is_gptq_allspark_supported(80, 89),
    reason="AllSpark Ampere kernel is not supported on this GPU type.")
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("has_zp", HAS_ZP_OPTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gptq_allspark_gemm_ampere(mnk_factors, group_size, has_zp, dtype):
    m_factor, n_factor, k_factor = mnk_factors
    m = m_factor
    n = n_factor * ALLSPARK_AMPERE_N_ALIGN
    k = k_factor * ALLSPARK_AMPERE_K_ALIGN

    input = rand_data((m, k), dtype=dtype)
    weight = rand_data((k, n), dtype=dtype)

    # Quantize (and apply act_order if provided)
    w_ref, qw, s, zp = quantize_weights(weight, scalar_types.uint8b128,
                                        group_size, has_zp)

    qw = qw.to(torch.uint8)
    if has_zp:
        zp = zp.to(dtype)
    properties = torch.cuda.get_device_properties(qw.device.index)
    sm_count = properties.multi_processor_count
    sm_version = properties.major * 10 + properties.minor

    n_32align = (n + 32 - 1) // 32 * 32

    qw_reorder, s_reorder, zp_reorder = ops.allspark_repack_weight(
        qw, s, zp, has_zp)
    opcheck(torch.ops._C.rearrange_kn_weight_as_n32k16_order,
            (qw, s, zp, has_zp, qw_reorder, s_reorder, zp_reorder, k, n,
             n_32align))

    opcheck(torch.ops._C.allspark_w8a16_gemm,
            (input, qw_reorder, s_reorder, zp_reorder, n, group_size, sm_count,
             sm_version, ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD, has_zp, True),
            test_utils=DEFAULT_OPCHECK_TEST_UTILS)
    output = ops.allspark_w8a16_gemm(input, qw_reorder, s_reorder, zp_reorder,
                                     n, group_size, sm_count, sm_version,
                                     ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
                                     has_zp, True)

    output_ref = torch.matmul(input, w_ref)
    torch.cuda.synchronize()
    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04
