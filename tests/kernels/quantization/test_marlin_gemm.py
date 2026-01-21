# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the marlin kernel.

Run `pytest tests/kernels/quantization/test_marlin_gemm.py`.
"""

import itertools

import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQ_MARLIN_24_MAX_PARALLEL,
    GPTQ_MARLIN_24_MIN_THREAD_N,
    GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES,
)
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
    rand_marlin_weight_mxfp4_like,
    rand_marlin_weight_nvfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    marlin_quant_fp8_torch,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    awq_marlin_quantize,
    get_weight_perm,
    marlin_quantize,
    marlin_weights,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_24 import (
    marlin_24_quantize,
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
        "These tests require gptq_marlin_repack,"
        "marlin_int4_fp8_preprocess, gptq_marlin_24_gemm,"
        "or gptq_marlin_gemm which are not supported on ROCm.",
        allow_module_level=True,
    )

ACT_ORDER_OPTS = [False, True]
K_FULL_OPTS = [False, True]
USE_ATOMIC_ADD_OPTS = [False, True]
USE_FP32_REDUCE_OPTS = [True]

MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MARLIN_24_K_CHUNKS = [128]
MARLIN_24_N_CHUNKS = [512]

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
    torch.cuda.synchronize()

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
    torch.cuda.synchronize()

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
                and current_platform.get_device_capability() not in [89, 120]
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
def test_gptq_marlin_gemm(
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

    output = ops.gptq_marlin_gemm(
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


# TODO: find better way to test this?
@torch.compile(fullgraph=True)
def marlin_24_gemm_tester(
    a_input,
    marlin_24_q_w_comp,
    marlin_24_meta,
    marlin_24_s,
    scratch,
    quant_type,
    size_m,
    size_n,
    size_k,
):
    return ops.gptq_marlin_24_gemm(
        a_input,
        marlin_24_q_w_comp,
        marlin_24_meta,
        marlin_24_s,
        scratch,
        quant_type,
        size_m,
        size_n,
        size_k,
    )


@pytest.mark.skipif(
    not is_quant_method_supported("gptq_marlin"),
    reason="Marlin is not supported on this GPU type.",
)
@pytest.mark.parametrize("k_chunk", MARLIN_24_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_24_N_CHUNKS)
@pytest.mark.parametrize("quant_type", GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES)
@pytest.mark.parametrize("group_size", GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_24_gemm(k_chunk, n_chunk, quant_type, group_size, mnk_factors):
    m_factor, n_factor, k_factor = mnk_factors

    size_m = m_factor
    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    a_input = rand_data((size_m, size_k))
    b_weight = rand_data((size_k, size_n))

    (w_24_ref, marlin_24_q_w_comp, marlin_24_meta, marlin_24_s) = marlin_24_quantize(
        b_weight, quant_type, group_size
    )

    workspace_24 = MarlinWorkspace(
        size_n, GPTQ_MARLIN_24_MIN_THREAD_N, GPTQ_MARLIN_24_MAX_PARALLEL
    )

    output_ref = torch.matmul(a_input, w_24_ref)

    opcheck(
        torch.ops._C.gptq_marlin_24_gemm,
        (
            a_input,
            marlin_24_q_w_comp,
            marlin_24_meta,
            marlin_24_s,
            workspace_24.scratch,
            quant_type.id,
            a_input.shape[0],
            b_weight.shape[1],
            a_input.shape[1],
        ),
        test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    output = marlin_24_gemm_tester(
        a_input,
        marlin_24_q_w_comp,
        marlin_24_meta,
        marlin_24_s,
        workspace_24.scratch,
        quant_type,
        a_input.shape[0],
        b_weight.shape[1],
        a_input.shape[1],
    )

    torch.cuda.synchronize()

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

    output = ops.gptq_marlin_gemm(
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

    torch.cuda.synchronize()

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

    output = ops.gptq_marlin_gemm(
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

    torch.cuda.synchronize()

    max_diff = compute_max_diff(output, output_ref)

    assert max_diff < 0.04
