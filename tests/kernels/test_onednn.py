# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for FlexAttention backend vs default backend"""

from typing import Optional

import pytest
import torch

from tests.kernels.utils import to_int8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

NK_FACTORS = [
    (256, 128),
    (4096, 4096),
    (16384, 4096),
    (1023, 491),
    (1001, 15),
]
M_FACTORS = [
    (16, 1, 32, 128, 64),
    (1, 17, 1, 31, 17),
]
CACHE_SIZES = [2]
DTYPE = [torch.bfloat16]


def rand_int8(shape: tuple, device: str = "cpu"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)


def ref_int8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    azp: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    output_type: torch.dtype,
):
    if azp is not None:
        a = a.to(dtype=torch.float32) - azp.to(dtype=torch.float32)
    output = torch.mm((scale_a * a.to(dtype=torch.float32)),
                      (scale_b * b.to(dtype=torch.float32)))
    if bias is not None:
        output += bias.float()

    return output.to(dtype=output_type)


def onednn_int8_gemm_test_helper(primitive_cache_size: int,
                                 m: int,
                                 n: int,
                                 k: int,
                                 per_tensor_a_quant: bool,
                                 per_tensor_b_quant: bool,
                                 use_azp: bool,
                                 use_bias: bool,
                                 out_dtype: torch.dtype = torch.bfloat16,
                                 device: str = "cpu"):
    # Test for a oneDNN kernel with per-tensor / per-token activation
    # quantization and per-tensor / per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    a_scales_shape = (1, 1) if per_tensor_a_quant else (m, 1)
    b_scales_shape = (1, 1) if per_tensor_b_quant else (1, n)

    scale_a = (torch.randn(a_scales_shape, device=device, dtype=torch.float32))
    scale_b = (torch.randn(b_scales_shape, device=device, dtype=torch.float32))

    if use_azp:
        azp = torch.rand(a_scales_shape, dtype=torch.float32) * 10 + 1.5
        azp = (azp / scale_a).round().to(dtype=torch.int32)
        azp_adj = scale_b * b.sum(dim=0, keepdim=True, dtype=torch.float32)
    else:
        azp = None
        azp_adj = None

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    handler = ops.create_onednn_scaled_mm(
        b,
        scale_b,
        out_dtype,
        not per_tensor_a_quant,
        use_azp,
        primitive_cache_size,
    )

    out = torch.zeros((m, n), dtype=out_dtype)
    ops.onednn_scaled_mm(handler, a, out, scale_a, azp, azp_adj, bias)
    baseline = ref_int8_scaled_mm(a, b, scale_a, scale_b, azp, bias, out_dtype)

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)

    if use_bias:
        # To test runtime bias setting
        out = torch.zeros((m, n), dtype=out_dtype)
        ops.onednn_scaled_mm(handler, a, out, scale_a, azp, azp_adj, None)
        baseline = ref_int8_scaled_mm(a, b, scale_a, scale_b, azp, None,
                                      out_dtype)

        torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)


def onednn_gemm_test_helper(primitive_cache_size: int,
                            m: int,
                            n: int,
                            k: int,
                            use_bias: bool,
                            use_stride: bool,
                            dtype: torch.dtype = torch.bfloat16,
                            device: str = "cpu"):
    if use_stride:
        a = torch.rand((m, 2 * k), dtype=dtype, device=device) * 1.5
        a = a[:, :k]
    else:
        a = torch.rand((m, k), dtype=dtype, device=device) * 1.5

    b = torch.rand((n, k), dtype=dtype, device=device) * 1.5

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=dtype) * 5
        bias_f32 = bias.float()
    else:
        bias = None
        bias_f32 = None

    handler = ops.create_onednn_mm(
        b.t(),
        primitive_cache_size,
    )

    out = ops.onednn_mm(handler, a, bias)
    baseline = torch.nn.functional.linear(a.float(), b.float(),
                                          bias_f32).to(dtype=a.dtype)

    torch.testing.assert_close(out, baseline)

    if use_bias:
        # To test runtime bias setting
        out = ops.onednn_mm(handler, a, None)
        baseline = torch.nn.functional.linear(a.float(), b.float(),
                                              None).to(dtype=a.dtype)

        torch.testing.assert_close(out, baseline)


@pytest.mark.parametrize("n,k", NK_FACTORS)
@pytest.mark.parametrize("m_list", M_FACTORS)
@pytest.mark.parametrize("per_tensor_a_scale", [True, False])
@pytest.mark.parametrize("per_tensor_b_scale", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("use_azp", [True, False])
@pytest.mark.parametrize("output_type", DTYPE)
@pytest.mark.parametrize("primitive_cache_size", CACHE_SIZES)
def test_onednn_int8_scaled_gemm(
    n: int,
    k: int,
    m_list: tuple[int],
    per_tensor_a_scale: bool,
    per_tensor_b_scale: bool,
    use_bias: bool,
    use_azp: bool,
    output_type: torch.dtype,
    primitive_cache_size: int,
):
    for m in m_list:
        onednn_int8_gemm_test_helper(
            primitive_cache_size=primitive_cache_size,
            m=m,
            n=n,
            k=k,
            per_tensor_a_quant=per_tensor_a_scale,
            per_tensor_b_quant=per_tensor_b_scale,
            use_bias=use_bias,
            use_azp=use_azp,
            out_dtype=output_type,
        )


@pytest.mark.parametrize("n,k", NK_FACTORS)
@pytest.mark.parametrize("m_list", M_FACTORS)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("use_stride", [True, False])
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("primitive_cache_size", CACHE_SIZES)
def test_onednn_gemm(
    n: int,
    k: int,
    m_list: tuple[int],
    use_bias: bool,
    use_stride: bool,
    dtype: torch.dtype,
    primitive_cache_size: int,
):
    for m in m_list:
        onednn_gemm_test_helper(
            primitive_cache_size=primitive_cache_size,
            m=m,
            n=n,
            k=k,
            use_bias=use_bias,
            use_stride=use_stride,
            dtype=dtype,
        )
