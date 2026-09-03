# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the marlin kernel.

Run `pytest tests/kernels/quantization/test_marlin_gemm.py`.
"""

import itertools
from unittest import mock

import pytest
import torch

from tests.kernels.utils import opcheck
from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    query_marlin_supported_quant_types,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
    rand_marlin_weight_mxfp4_like,
    rand_marlin_weight_nvfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    marlin_quant_fp8_torch,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    awq_marlin_quantize,
    get_weight_perm,
    marlin_quantize,
    marlin_weights,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    awq_pack,
    gptq_pack,
    gptq_quantize_weights,
    quantize_weights,
    sort_weights,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if current_platform.is_rocm():
    pytest.skip(
        "These tests require marlin, which is not supported on ROCm.",
        allow_module_level=True,
    )

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]
USE_ATOMIC_ADD_OPTS = [False, True]
USE_FP32_REDUCE_OPTS = [True]

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MARLIN_REPACK_NK_FACTORS = [
    (4, 8),
    (7, 5),
    (13, 11),
]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (26, 37, 13),
    (257, 13, 11),
]

DTYPES = [torch.float16, torch.bfloat16]

DENSE_MARLIN_QUANT_TEST_CONFIGS = [
    # AWQ-INT4
    {"b_type": scalar_types.uint4, "group_blocks": [-1, 2, 4, 8]},
    # GPTQ-INT4
    {
        "b_type": scalar_types.uint4b8,
        "support_act_order": True,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT8
    {
        "b_type": scalar_types.uint8b128,
        "support_act_order": True,
        "group_blocks": [-1, 2, 4, 8],
    },
    # FP8
    {"b_type": scalar_types.float8_e4m3fn, "group_blocks": [-1, 8]},
    # NVFP4
    {"b_type": scalar_types.float4_e2m1f, "group_blocks": [1]},
    # MXFP4
    {
        "a_type": [scalar_types.bfloat16],
        "b_type": scalar_types.float4_e2m1f,
        "group_blocks": [2],
    },
    # AWQ-INT4 with INT8 activation
    {
        "a_type": [scalar_types.int8],
        "b_type": scalar_types.uint4,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT4 with INT8 activation
    {
        "a_type": [scalar_types.int8],
        "b_type": scalar_types.uint4b8,
        "group_blocks": [-1, 2, 4, 8],
    },
    # GPTQ-INT4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.uint4b8,
        "group_blocks": [-1, 2, 4, 8],
    },
    # AWQ-INT4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.uint4,
        "group_blocks": [-1, 2, 4, 8],
    },
    # MXFP4 with FP8 activation
    {
        "a_type": [scalar_types.float8_e4m3fn],
        "b_type": scalar_types.float4_e2m1f,
        "c_type": [scalar_types.bfloat16],
        "group_blocks": [2],
    },
]


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


def rand_data(shape, dtype=torch.float16):
    return torch.randn(shape, dtype=dtype, device="cuda")


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_int4_fp8_preprocess_without_zp():
    qweight_unpacked = torch.randint(
        0, 16, size=(2048, 2048), dtype=torch.int32, device="cuda"
    )
    qweight_packed = qweight_unpacked[:, ::2] * 16 + qweight_unpacked[:, 1::2]
    qweight_packed = qweight_packed.to(torch.int8).view(torch.int32)

    cuda_res = ops.marlin_int4_fp8_preprocess(qweight_packed)

    torch_res = torch.where(
        qweight_unpacked >= 8, qweight_unpacked - 8, 15 - qweight_unpacked
    )
    torch_res = torch_res[:, ::2] * 16 + torch_res[:, 1::2]
    torch_res = torch_res.to(torch.int8).view(torch.int32)

    assert (cuda_res == torch_res).all()


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_int4_fp8_preprocess_awq():
    group_size = 128

    qweight_unpacked = torch.randint(
        0, 16, size=(2048, 2048), dtype=torch.int32, device="cuda"
    )
    qzeros_unpacked = torch.randint(
        0, 16, size=(2048 // group_size, 2048), dtype=torch.int32, device="cuda"
    )

    qweight_packed = qweight_unpacked[:, ::2] * 16 + qweight_unpacked[:, 1::2]
    qweight_packed = qweight_packed.to(torch.int8).view(torch.int32)
    qzeros_packed = qzeros_unpacked[:, ::2] * 16 + qzeros_unpacked[:, 1::2]
    qzeros_packed = qzeros_packed.to(torch.int8).view(torch.int32)

    cuda_res = ops.marlin_int4_fp8_preprocess(qweight_packed, qzeros_packed)

    repeated_zp = qzeros_unpacked.repeat_interleave(group_size, 0)
    torch_res = qweight_unpacked - repeated_zp
    torch_res[torch_res < 0] = 15 - qweight_unpacked[torch_res < 0]
    torch_res = torch_res[:, ::2] * 16 + torch_res[:, 1::2]
    torch_res = torch_res.to(torch.int8).view(torch.int32)

    assert (cuda_res == torch_res).all()


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", query_marlin_supported_quant_types(False, False))
@pytest.mark.parametrize("act_order", ACT_ORDER_OPTS)
@pytest.mark.parametrize("is_a_8bit", [True, False])
@pytest.mark.parametrize("nk_factors", MARLIN_REPACK_NK_FACTORS)
def test_gptq_marlin_repack(
    k_chunk, n_chunk, quant_type, act_order, is_a_8bit, nk_factors
):
    n_factor, k_factor = nk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor
    group_size = 128

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return
        if is_a_8bit:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order
    )

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits, is_a_8bit)
    marlin_q_w_1 = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, weight_perm, is_a_8bit
    )

    opcheck(
        torch.ops._C.gptq_marlin_repack,
        (q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits, is_a_8bit),
    )

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.gptq_marlin_repack(
        q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits, is_a_8bit
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(marlin_q_w_1, marlin_q_w_2)


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", query_marlin_supported_quant_types(True))
@pytest.mark.parametrize("is_a_8bit", [True, False])
@pytest.mark.parametrize("nk_factors", MARLIN_REPACK_NK_FACTORS)
def test_awq_marlin_repack(k_chunk, n_chunk, quant_type, is_a_8bit, nk_factors):
    n_factor, k_factor = nk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    group_size = 128

    # Create input
    b_weight = rand_data((size_k, size_n))

    # Quantize
    w_ref, q_w, s, zp = quantize_weights(
        b_weight, quant_type, group_size, zero_points=True
    )

    # Pack to AWQ format
    q_w_awq = awq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits, is_a_8bit)
    marlin_q_w_1 = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, weight_perm, is_a_8bit
    )

    opcheck(
        torch.ops._C.awq_marlin_repack,
        (q_w_awq, size_k, size_n, quant_type.size_bits, is_a_8bit),
    )

    # Run Marlin repack GPU kernel
    marlin_q_w_2 = ops.awq_marlin_repack(
        q_w_awq, size_k, size_n, quant_type.size_bits, is_a_8bit
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(marlin_q_w_1, marlin_q_w_2)


def marlin_generate_valid_test_cases():
    all_combinations = itertools.product(
        DENSE_MARLIN_QUANT_TEST_CONFIGS,
        MNK_FACTORS,
        MARLIN_N_CHUNKS,
        MARLIN_K_CHUNKS,
        ACT_ORDER_OPTS,
        K_FULL_OPTS,
        USE_ATOMIC_ADD_OPTS,
        USE_FP32_REDUCE_OPTS,
    )

    def is_invalid(
        a_type,
        b_type,
        c_type,
        group_blocks,
        size_m,
        size_n,
        size_k,
        act_order,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
    ):
        if use_atomic_add:
            if use_fp32_reduce:
                return False
            if (
                c_type == scalar_types.bfloat16
                and torch.cuda.get_device_capability()[0] < 9
            ):
                return False

        group_size = group_blocks if group_blocks <= 0 else group_blocks * 16
        if group_size > 0 and size_k % group_size != 0:
            return False

        if act_order and group_size in [-1, size_k]:
            return False
        if group_size == size_k:
            return False
        if not act_order and is_k_full:
            return False

        return a_type.size_bits < 16 or a_type is c_type

    cases = []
    for case in all_combinations:
        quant_test_config, mnk_factors, n_chunk, k_chunk, act_order, *_ = case
        size_m = mnk_factors[0]
        size_n = mnk_factors[1] * n_chunk
        size_k = mnk_factors[2] * k_chunk

        if act_order and not quant_test_config.get("support_act_order", False):
            continue

        f16_types = [scalar_types.float16, scalar_types.bfloat16]
        inner_combinations = itertools.product(
            quant_test_config.get("a_type", f16_types),
            [quant_test_config["b_type"]],
            quant_test_config.get("c_type", f16_types),
            quant_test_config["group_blocks"],
        )

        for sub_case in inner_combinations:
            if (
                sub_case[0] == scalar_types.float8_e4m3fn
                and not current_platform.is_device_capability(89)
                and not current_platform.is_device_capability_family(120)
            ):
                continue
            args = sub_case + (size_m, size_n, size_k) + case[4:]
            if is_invalid(*args):
                cases.append(args)
    return cases


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    (
        "a_type, b_type, c_type, group_blocks,"
        "size_m, size_n, size_k, act_order, is_k_full,"
        "use_atomic_add, use_fp32_reduce"
    ),
    marlin_generate_valid_test_cases(),
)
def test_marlin_gemm(
    a_type,
    b_type,
    c_type,
    group_blocks,
    size_m,
    size_n,
    size_k,
    act_order,
    is_k_full,
    use_atomic_add,
    use_fp32_reduce,
):
    has_zp = b_type in [scalar_types.uint4, scalar_types.uint8]

    group_size = group_blocks if group_blocks <= 0 else group_blocks * 16

    if c_type == scalar_types.float16:
        dtype = torch.float16
    elif c_type == scalar_types.bfloat16:
        dtype = torch.bfloat16
    else:
        raise RuntimeError("unsupported c_type")

    if a_type == scalar_types.int8:
        a_dtype = torch.int8
    elif a_type == scalar_types.float8_e4m3fn:
        a_dtype = torch.float8_e4m3fn
    else:
        a_dtype = dtype

    a_input = rand_data((size_m, size_k), dtype=dtype)
    b_weight = rand_data((size_k, size_n), dtype=dtype)

    if b_type == scalar_types.float4_e2m1f:
        if group_size == 16:
            w_ref, marlin_q_w, marlin_s, marlin_s2 = rand_marlin_weight_nvfp4_like(
                b_weight.T, group_size, input_dtype=a_dtype
            )
        else:
            w_ref, marlin_q_w, marlin_s = rand_marlin_weight_mxfp4_like(
                b_weight.T, group_size, input_dtype=a_dtype
            )
            marlin_s2 = None

        g_idx = None
        sort_indices = None
        marlin_zp = None
    elif b_type == scalar_types.float8_e4m3fn:
        w_ref, marlin_q_w, marlin_s = marlin_quant_fp8_torch(
            b_weight.T, group_size, input_dtype=a_dtype
        )
        g_idx = None
        sort_indices = None
        marlin_zp = None
        marlin_s2 = None
    elif has_zp:
        w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
            b_weight, b_type, group_size, input_dtype=a_dtype
        )
        g_idx = None
        sort_indices = None
        marlin_s2 = None
    else:
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            b_weight, b_type, group_size, act_order, input_dtype=a_dtype
        )

        marlin_zp = None
        marlin_s2 = None

    workspace = marlin_make_workspace_new(w_ref.device)

    if a_type == scalar_types.int8:
        a_input, a_scales = per_token_quant_int8(a_input)
        a_input_ref = a_input.to(a_scales.dtype) * a_scales.view(-1, 1)
        a_input_ref = a_input_ref.to(dtype)

        if group_size != -1:
            a_scales = a_scales / 4096 * marlin_s.max()
            a_scales = a_scales.float()
            marlin_s = marlin_s / marlin_s.max() * 4096
            marlin_s = marlin_s.round().to(torch.int16).view(dtype)
    elif a_type == scalar_types.float8_e4m3fn:
        a_input, a_scales = ops.scaled_fp8_quant(a_input, use_per_token_if_dynamic=True)
        a_input_ref = a_input.to(a_scales.dtype) * a_scales.view(-1, 1)
        a_input_ref = a_input_ref.to(dtype)
    else:
        assert a_type.size_bits == 16
        a_input_ref = a_input
        a_scales = None

    output = torch.empty((size_m, size_n), dtype=dtype, device=a_input.device)

    output = ops.marlin_gemm(
        a_input,
        output,
        marlin_q_w,
        None,
        marlin_s,
        a_scales,
        marlin_s2,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        b_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input_ref, w_ref)

    max_diff = compute_max_diff(output, output_ref)
    assert max_diff < 0.04


SMALL_M_WAVE_SHAPES = [
    # b_type, group_size, size_k, size_n -- decode-shaped GEMMs whose
    # n-slice count exceeds one wave of thread blocks (n / thread_n > sms),
    # traversing the small-M two-blocks-per-SM launch on CC >= 10.0.
    (scalar_types.float8_e4m3fn, -1, 2688, 10304),
    (scalar_types.float4_e2m1f, 16, 2688, 32768),
]


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("b_type_gs_k_n", SMALL_M_WAVE_SHAPES)
@pytest.mark.parametrize("size_m", [1, 3, 4, 8])
@pytest.mark.parametrize("use_fp32_reduce", [True, False])
@pytest.mark.parametrize("workspace_blocks_per_sm", [1, 2])
def test_marlin_gemm_small_m_wave_quantization(
    b_type_gs_k_n, size_m, use_fp32_reduce, workspace_blocks_per_sm
):
    """Small-M correctness on shapes eligible for the two-wave boost.

    workspace_blocks_per_sm=1 pins the pre-existing single-wave launch
    (the boost fails closed on a one-wave workspace), so both grid
    layouts of the same GEMM are checked against the reference.
    """
    b_type, group_size, size_k, size_n = b_type_gs_k_n

    a_input = rand_data((size_m, size_k), dtype=torch.bfloat16)
    b_weight = rand_data((size_k, size_n), dtype=torch.bfloat16)

    if b_type == scalar_types.float4_e2m1f:
        w_ref, marlin_q_w, marlin_s, marlin_s2 = rand_marlin_weight_nvfp4_like(
            b_weight.T, group_size
        )
    else:
        w_ref, marlin_q_w, marlin_s = marlin_quant_fp8_torch(b_weight.T, group_size)
        marlin_s2 = None

    workspace = marlin_make_workspace_new(w_ref.device, workspace_blocks_per_sm)

    output = ops.marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        None,
        marlin_s,
        None,
        marlin_s2,
        None,
        None,
        None,
        workspace,
        b_type,
        size_m,
        size_n,
        size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input, w_ref)

    max_diff = compute_max_diff(output, output_ref)
    assert max_diff < 0.04


def test_marlin_gemm_subset_input():
    quant_type = scalar_types.uint4b8
    group_size = 128

    size_m, size_k, size_n = 32, 1024, 2048
    big_m = size_m * 2
    big_k = size_k * 2

    a_input = rand_data((big_m, big_k))[8 : size_m + 8, 8 : size_k + 8]
    b_weight = rand_data((size_k, size_n))

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    workspace = marlin_make_workspace_new(a_input.device)

    output = ops.marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        None,
        marlin_s,
        None,
        None,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input, w_ref)

    torch.accelerator.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


@pytest.mark.parametrize("size_m", [1, 256])
def test_marlin_gemm_with_bias(size_m):
    quant_type = scalar_types.uint4b8
    group_size = 128

    size_k, size_n = 1024, 2048
    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))
    b_bias = rand_data((size_n,)) * 10

    marlin_bias = marlin_permute_bias(b_bias)

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight, quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    workspace = marlin_make_workspace_new(a_input.device)

    output = ops.marlin_gemm(
        a_input,
        None,
        marlin_q_w,
        marlin_bias,
        marlin_s,
        None,
        None,
        marlin_zp,
        g_idx,
        sort_indices,
        workspace,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    output_ref = torch.matmul(a_input, w_ref) + b_bias.view(1, -1)

    torch.accelerator.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04


# Large-M dequant dispatch (VLLM_MARLIN_LARGE_M_BF16): prefill-sized GEMMs
# leave the Marlin kernel for a dequant + 16-bit GEMM; decode stays Marlin.
LARGE_M_K_N = (512, 1024)


def _make_fp8_layer_factory(k, n, size_k_first, with_bias, dtype):
    w = rand_data((n, k), dtype=dtype) / 10
    scale = w.abs().amax(dim=1, keepdim=True) / 448
    w_fp8 = (w / scale).to(torch.float8_e4m3fn)
    w_ref = w_fp8.to(dtype) * scale
    bias = rand_data((n,), dtype=dtype) * 10 if with_bias else None

    def factory():
        layer = torch.nn.Module()
        weight = w_fp8.T.contiguous() if size_k_first else w_fp8.clone()
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            scale.view(-1).clone(), requires_grad=False
        )
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        layer.orig_dtype = dtype
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        prepare_fp8_layer_for_marlin(layer, size_k_first=size_k_first)
        return layer

    return factory, w_ref, bias


def _apply_fp8(layer, x, n, k):
    return apply_fp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=n,
        size_k=k,
        bias=getattr(layer, "bias", None),
    )


def _nvfp4_dequant_ref(packed, group_scales, dtype):
    """Reference dequant of packed NVFP4 (even elem = low nibble) with
    per-16-group scales already multiplied by the global scale."""
    hi = (packed & 0b10000000) | ((packed & 0b01110000) >> 2)
    lo = packed << 4
    lo = (lo & 0b10000000) | ((lo & 0b01110000) >> 2)
    w = torch.stack(
        [
            lo.view(torch.float8_e4m3fn).to(dtype),
            hi.view(torch.float8_e4m3fn).to(dtype),
        ],
        dim=-1,
    ).view(packed.size(0), -1) * (2**6)
    n, k = w.shape
    w = w.view(n, k // 16, 16) * group_scales.to(dtype).unsqueeze(-1)
    return w.view(n, k)


def _make_nvfp4_layer_factory(k, n, dtype, tiny_scale_rows=0):
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.rand((n, k // 16), device="cuda") * 1.5 + 0.5
    if tiny_scale_rows:
        scales[:tiny_scale_rows] = 2**-20
    scales = scales.to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
    w_ref = _nvfp4_dequant_ref(packed, scales.to(torch.float32) * global_scale, dtype)
    bias = rand_data((n,), dtype=dtype) * 10

    def factory():
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale.clone(), requires_grad=False
        )
        layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        layer.params_dtype = dtype
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        prepare_fp4_layer_for_marlin(layer)
        return layer

    return factory, w_ref, bias


def _apply_fp4(layer, x, n, k):
    return apply_fp4_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        weight_global_scale=layer.weight_global_scale,
        workspace=layer.workspace,
        size_n=n,
        size_k=k,
        bias=getattr(layer, "bias", None),
    )


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("size_k_first", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_large_m_dispatch_fp8(monkeypatch, size_k_first, with_bias, dtype):
    """M >= threshold leaves Marlin for dequant + 16-bit GEMM; both arms
    match the dequantized reference and each other at kernel tolerance."""
    k, n = LARGE_M_K_N
    size_m = 768
    factory, w_ref, _ = _make_fp8_layer_factory(k, n, size_k_first, with_bias, dtype)

    monkeypatch.delenv("VLLM_MARLIN_LARGE_M_BF16", raising=False)
    marlin_layer = factory()
    assert getattr(marlin_layer.weight, "marlin_large_m_ctx", None) is None

    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    dispatch_layer = factory()
    assert getattr(dispatch_layer.weight, "marlin_large_m_ctx", None) is not None

    x = rand_data((size_m, k), dtype=dtype) / 10
    ref = torch.matmul(x, w_ref.T)
    if with_bias:
        ref = ref + dispatch_layer.weight.marlin_large_m_ctx.bias

    marlin_out = _apply_fp8(marlin_layer, x, n, k)
    with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
        dispatch_out = _apply_fp8(dispatch_layer, x, n, k)
        assert spy.call_count == 0
    assert compute_max_diff(dispatch_out, ref) < 0.04
    assert compute_max_diff(marlin_out, ref) < 0.04
    assert compute_max_diff(dispatch_out, marlin_out) < 0.04


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_large_m_dispatch_nvfp4(monkeypatch, dtype):
    k, n = LARGE_M_K_N
    size_m = 768
    factory, w_ref, _ = _make_nvfp4_layer_factory(k, n, dtype)

    monkeypatch.delenv("VLLM_MARLIN_LARGE_M_BF16", raising=False)
    marlin_layer = factory()
    assert getattr(marlin_layer.weight, "marlin_large_m_ctx", None) is None

    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    dispatch_layer = factory()
    ctx = getattr(dispatch_layer.weight, "marlin_large_m_ctx", None)
    assert ctx is not None

    x = rand_data((size_m, k), dtype=dtype) / 10
    ref = torch.matmul(x, w_ref.T) + ctx.bias

    marlin_out = _apply_fp4(marlin_layer, x, n, k)
    with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
        dispatch_out = _apply_fp4(dispatch_layer, x, n, k)
        assert spy.call_count == 0
    assert compute_max_diff(dispatch_out, ref) < 0.04
    assert compute_max_diff(marlin_out, ref) < 0.04
    assert compute_max_diff(dispatch_out, marlin_out) < 0.04


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_large_m_nvfp4_scale_clamp(monkeypatch):
    """Groups whose processed scale clamps to zero in the Marlin format
    must dequantize to zero on the dispatch path too."""
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    factory, _, _ = _make_nvfp4_layer_factory(k, n, dtype, tiny_scale_rows=4)

    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    dispatch_layer = factory()
    ctx = dispatch_layer.weight.marlin_large_m_ctx
    assert (ctx.scales[:4] == 0).all()
    assert (ctx.scales[4:] != 0).any()

    monkeypatch.delenv("VLLM_MARLIN_LARGE_M_BF16")
    marlin_layer = factory()

    x = rand_data((768, k), dtype=dtype) / 10
    marlin_out = _apply_fp4(marlin_layer, x, n, k)
    dispatch_out = _apply_fp4(dispatch_layer, x, n, k)
    assert compute_max_diff(dispatch_out, marlin_out) < 0.04


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("size_m", [1, 4, 8])
def test_marlin_large_m_decode_selects_marlin(monkeypatch, size_m):
    """Decode/verify-sized GEMMs (M <= 8) always stay on the Marlin
    kernel, even with the dispatch enabled."""
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    for factory, apply_fn in (
        (_make_fp8_layer_factory(k, n, True, False, dtype)[0], _apply_fp8),
        (_make_nvfp4_layer_factory(k, n, dtype)[0], _apply_fp4),
    ):
        layer = factory()
        assert layer.weight.marlin_large_m_ctx is not None
        x = rand_data((size_m, k), dtype=dtype) / 10
        with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
            apply_fn(layer, x, n, k)
            assert spy.call_count == 1


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_large_m_fail_closed_gates(monkeypatch):
    """Sub-minimum thresholds and over-cap workspaces disable the
    dispatch entirely (no ctx is attached)."""
    k, n = LARGE_M_K_N
    factory, _, _ = _make_fp8_layer_factory(k, n, True, False, torch.bfloat16)

    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "8")
    assert getattr(factory().weight, "marlin_large_m_ctx", None) is None

    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    monkeypatch.setattr(marlin_utils_fp8, "_LARGE_M_MAX_WORKSPACE_BYTES", 2 * k * n - 1)
    assert getattr(factory().weight, "marlin_large_m_ctx", None) is None

    monkeypatch.setattr(marlin_utils_fp8, "_LARGE_M_MAX_WORKSPACE_BYTES", 2 * k * n)
    assert getattr(factory().weight, "marlin_large_m_ctx", None) is not None


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_large_m_custom_threshold(monkeypatch):
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "64")
    factory, _, _ = _make_fp8_layer_factory(k, n, True, False, dtype)
    layer = factory()
    assert layer.weight.marlin_large_m_ctx.threshold == 64
    with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
        _apply_fp8(layer, rand_data((63, k), dtype=dtype), n, k)
        assert spy.call_count == 1
        _apply_fp8(layer, rand_data((64, k), dtype=dtype), n, k)
        assert spy.call_count == 1


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_large_m_threshold_sweep_values(monkeypatch):
    """Both round-3 sweep points (512, 4096) are reachable via the env
    knob and dispatch exactly at their boundary."""
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    factory, _, _ = _make_fp8_layer_factory(k, n, True, False, dtype)
    for threshold in (512, 4096):
        monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", str(threshold))
        layer = factory()
        assert layer.weight.marlin_large_m_ctx.threshold == threshold
        with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
            _apply_fp8(layer, rand_data((threshold - 1, k), dtype=dtype), n, k)
            assert spy.call_count == 1
            _apply_fp8(layer, rand_data((threshold, k), dtype=dtype), n, k)
            assert spy.call_count == 1


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("size_m", [1, 2, 4])
def test_marlin_large_m_capture_vs_eager(monkeypatch, size_m):
    """FULL-cudagraph legality at decode shapes: capturing the custom op
    freezes the Marlin branch and replay matches eager bitwise."""
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    # Atomic-add reduce is accumulation-order nondeterministic; force it
    # off so capture-vs-eager can assert bitwise equality.
    monkeypatch.setattr(
        marlin_utils_fp8, "should_use_atomic_add_reduce", lambda **kwargs: False
    )
    for factory, apply_fn in (
        (_make_fp8_layer_factory(k, n, True, True, dtype)[0], _apply_fp8),
        (_make_nvfp4_layer_factory(k, n, dtype)[0], _apply_fp4),
    ):
        layer = factory()
        assert layer.weight.marlin_large_m_ctx is not None
        x = rand_data((size_m, k), dtype=dtype) / 10
        eager_out = apply_fn(layer, x, n, k)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                apply_fn(layer, x, n, k)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = apply_fn(layer, x, n, k)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(graph_out, eager_out)


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
def test_marlin_large_m_compile_branch_not_baked(monkeypatch):
    """torch.compile of a region containing the op must not bake the
    M-vs-threshold branch: one dynamic artifact, then both branches stay
    reachable per-call. (The config-lane failure mode: dynamo traced the
    Python >= threshold comparison once and every inductor range artifact
    reused that branch.)"""
    import torch._dynamo

    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "64")
    factory, w_ref, _ = _make_fp8_layer_factory(k, n, True, False, dtype)
    layer = factory()
    assert layer.weight.marlin_large_m_ctx.threshold == 64

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda x: _apply_fp8(layer, x, n, k), fullgraph=True, dynamic=True
    )
    x_large = rand_data((128, k), dtype=dtype) / 10
    x4 = rand_data((4, k), dtype=dtype) / 10
    x2 = rand_data((2, k), dtype=dtype) / 10
    try:
        with mock.patch.object(ops, "marlin_gemm", wraps=ops.marlin_gemm) as spy:
            out_large = compiled(x_large)
            assert spy.call_count == 0  # dequant branch taken
            # A baked branch surfaces as a shape guard: any M on the other
            # side of the threshold would have to recompile.
            with torch._dynamo.config.patch(error_on_recompile=True):
                out4 = compiled(x4)
                out2 = compiled(x2)
            assert spy.call_count == 2  # Marlin branch, same artifact
        assert compute_max_diff(out_large, x_large @ w_ref.T) < 0.04
        assert compute_max_diff(out4, x4 @ w_ref.T) < 0.04
        assert compute_max_diff(out2, x2 @ w_ref.T) < 0.04
    finally:
        torch._dynamo.reset()


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("size_m", [4, 768])
def test_marlin_large_m_op_fake(monkeypatch, size_m):
    """The fake impl's single meta (fresh contiguous [M, N] in the
    activation dtype) must hold for BOTH runtime branches."""
    k, n = LARGE_M_K_N
    dtype = torch.bfloat16
    monkeypatch.setenv("VLLM_MARLIN_LARGE_M_BF16", "1")
    factory, _, _ = _make_fp8_layer_factory(k, n, True, True, dtype)
    layer = factory()
    ctx = layer.weight.marlin_large_m_ctx
    x = rand_data((size_m, k), dtype=dtype) / 10
    opcheck(
        torch.ops.vllm.marlin_large_m_gemm,
        (
            x,
            layer.weight.data,
            layer.weight_scale.data,
            None,
            getattr(layer, "bias", None),
            layer.workspace,
            ctx.weight,
            ctx.scales,
            ctx.bias,
            ctx.workspace.buf,
            ctx.threshold,
            k,
            n,
            ctx.wtype,
            ctx.k_first,
            True,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )
