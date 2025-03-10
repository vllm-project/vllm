# SPDX-License-Identifier: Apache-2.0
"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import cdiv

from .utils import baseline_scaled_mm, to_fp8, to_int8

MNK_FACTORS = [
    (1, 256, 128),
    (1, 16384, 1024),
    (1, 24576, 496),
    (16, 256, 496),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 2048, 496),
    (64, 16384, 1024),
    (100, 8192, 496),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

# -1 means full extent in that dimension
TENSORWISE_GROUP_SHAPE = (-1, -1)
PER_TOKEN_GROUP_SHAPE = (1, -1)
PER_OUT_CH_GROUP_SHAPE = (-1, 1)

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]


def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)


def group_scale_helper(shape, group_shape):
    return [shape[i] if s < 0 else s for i, s in enumerate(group_shape)]


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    group_shape = group_scale_helper(shape, group_shape)
    return tuple(
        cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def cutlass_fp8_gemm_helper(m: int,
                            n: int,
                            k: int,
                            a_scale_group_shape: tuple,
                            b_scale_group_shape: tuple,
                            use_bias: bool,
                            out_dtype: type[torch.dtype] = torch.bfloat16,
                            device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    a_scales_shape = scale_shape(a.shape, a_scale_group_shape)
    b_scales_shape = scale_shape(b.shape, b_scale_group_shape)

    scale_a = (torch.randn(a_scales_shape, device=device, dtype=torch.float32))
    scale_b = (torch.randn(b_scales_shape, device=device, dtype=torch.float32))

    # make scales M-major for blockwise quant, doesn't affect 1D scales
    scale_a = scale_a.t().contiguous().t()
    # make scales K-major for blockwise quant, doesn't affect 1D scales
    scale_b = scale_b.t().contiguous().t()

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(out, baseline, rtol=1e-2, atol=5e-2)

    opcheck(torch.ops._C.cutlass_scaled_mm,
            (out, a, b, scale_a, scale_b, bias))


def cutlass_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             a_scale_group_shape: tuple,
                             b_scale_group_shape: tuple,
                             use_bias: bool,
                             out_dtype: type[torch.dtype] = torch.bfloat16,
                             device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    a_scales_shape = scale_shape(a.shape, a_scale_group_shape)
    b_scales_shape = scale_shape(b.shape, b_scale_group_shape)

    scale_a = (torch.randn(a_scales_shape, device=device, dtype=torch.float32))
    scale_b = (torch.randn(b_scales_shape, device=device, dtype=torch.float32))

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)

    opcheck(torch.ops._C.cutlass_scaled_mm,
            (out, a, b, scale_a, scale_b, bias))


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not current_platform.has_device_capability(89),
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm(m: int, n: int, k: int, a_scale_group_shape,
                          b_scale_group_shape, use_bias: bool):
    cutlass_fp8_gemm_helper(m, n, k, a_scale_group_shape, b_scale_group_shape,
                            use_bias)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("a_scale_group_shape,b_scale_group_shape",
                         [((1, 128), (128, 128))])
@pytest.mark.parametrize("use_bias", [False])
@pytest.mark.skipif(not current_platform.has_device_capability(90),
                    reason="FP8 blockwise is not supported on this GPU type.")
def test_cutlass_fp8_blockwise_scale_gemm(m: int, n: int, k: int,
                                          a_scale_group_shape,
                                          b_scale_group_shape, use_bias: bool):
    if k % b_scale_group_shape[0] != 0 or n % b_scale_group_shape[1] != 0:
        return
    if m % a_scale_group_shape[0] != 0 or k % a_scale_group_shape[1] != 0:
        return
    cutlass_fp8_gemm_helper(m, n, k, a_scale_group_shape, b_scale_group_shape,
                            use_bias)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm(m: int, n: int, k: int, a_scale_group_shape,
                           b_scale_group_shape, use_bias: bool):
    cutlass_int8_gemm_helper(m, n, k, a_scale_group_shape, b_scale_group_shape,
                             use_bias)


@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm_output_dtype(a_scale_group_shape,
                                        b_scale_group_shape,
                                        out_dtype: type[torch.dtype],
                                        use_bias: bool):
    cutlass_int8_gemm_helper(512,
                             512,
                             512,
                             a_scale_group_shape,
                             b_scale_group_shape,
                             use_bias,
                             out_dtype=out_dtype)


@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not current_platform.has_device_capability(89),
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_output_dtype(a_scale_group_shape,
                                       b_scale_group_shape,
                                       out_dtype: type[torch.dtype],
                                       use_bias: bool):
    cutlass_fp8_gemm_helper(512,
                            512,
                            512,
                            a_scale_group_shape,
                            b_scale_group_shape,
                            use_bias,
                            out_dtype=out_dtype)


@pytest.mark.parametrize("a_scale_group_shape,b_scale_group_shape",
                         [((1, 128), (128, 128))])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False])
@pytest.mark.skipif(not current_platform.has_device_capability(90),
                    reason="FP8 blockwise is not supported on this GPU type.")
def test_cutlass_fp8_blockwise_scale_gemm_dtype(a_scale_group_shape,
                                                b_scale_group_shape,
                                                out_dtype: type[torch.dtype],
                                                use_bias: bool):
    cutlass_fp8_gemm_helper(512,
                            512,
                            512,
                            a_scale_group_shape,
                            b_scale_group_shape,
                            use_bias,
                            out_dtype=out_dtype)


@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(not current_platform.has_device_capability(89),
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_devices(a_scale_group_shape, b_scale_group_shape,
                                  use_bias: bool, device: str):
    cutlass_fp8_gemm_helper(512, 512, 512, a_scale_group_shape,
                            b_scale_group_shape, use_bias, torch.bfloat16,
                            device)


@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cutlass_int8_gemm_devices(a_scale_group_shape, b_scale_group_shape,
                                   use_bias: bool, device: str):
    cutlass_int8_gemm_helper(512,
                             512,
                             512,
                             a_scale_group_shape,
                             b_scale_group_shape,
                             use_bias,
                             out_dtype=torch.bfloat16,
                             device=device)


# For the following two tests:
# N and K correspond to the size of the weight matrix and likely to be multiples
# of a large power of two. In any case, the kernel will have a naive fallback
# when N and K are not divisible by 16. But M is the number of tokens and the
# kernel must handle any M thrown at it.
@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not current_platform.has_device_capability(89),
                    reason="FP8 is not supported on this GPU type.")
def test_cutlass_fp8_gemm_m_sweep(a_scale_group_shape, b_scale_group_shape,
                                  use_bias: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_fp8_gemm_helper(m, nk, nk, a_scale_group_shape,
                                    b_scale_group_shape, use_bias)


@pytest.mark.parametrize("a_scale_group_shape",
                         [PER_TOKEN_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("b_scale_group_shape",
                         [PER_OUT_CH_GROUP_SHAPE, TENSORWISE_GROUP_SHAPE])
@pytest.mark.parametrize("use_bias", [True, False])
def test_cutlass_int8_gemm_m_sweep(a_scale_group_shape, b_scale_group_shape,
                                   use_bias: bool):
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutlass_int8_gemm_helper(m, nk, nk, a_scale_group_shape,
                                     b_scale_group_shape, use_bias)


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("k", [64, 128, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skip
def test_cutlass_int8_azp_bias_fold(m: int, n: int, k: int,
                                    out_dtype: torch.dtype):
    # Currently, the test is failing because folding azp into
    # 16-bit bias loses too much precision
    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, n), device="cuda", dtype=torch.float32) / 10

    aq_i8 = rand_int8((m, k))
    bq_i8 = rand_int8((n, k)).t()

    aq_i32 = aq_i8.to(dtype=torch.int32)
    bq_i32 = bq_i8.to(dtype=torch.int32)

    aq_f32 = aq_i8.to(dtype=torch.float32)
    bq_f32 = bq_i8.to(dtype=torch.float32)

    b_dq = scale_b * bq_f32

    azp_a = torch.rand((1, ), device="cuda", dtype=torch.float32) * 10 + 1.5
    azp_aq_i8 = (azp_a / scale_a).to(dtype=torch.int8)
    azp_a = azp_aq_i8.to(dtype=torch.float32) * scale_a  # correct for rounding

    a_dq = scale_a * (aq_i32 + azp_aq_i8).to(dtype=torch.float32)
    torch.testing.assert_close(a_dq, scale_a * aq_f32 + azp_a)

    baseline_dq = torch.mm(a_dq, b_dq).to(out_dtype)

    J = torch.ones((1, k), device="cuda", dtype=torch.float32)
    azp_bias = (azp_a * scale_b * (J @ bq_f32)).to(out_dtype)
    assert azp_bias.shape == (1, n)
    assert azp_bias[0, :].shape == (n, )

    baseline_q = (scale_a.to(device='cpu') * scale_b.to(device='cpu') * (
        (aq_i32 + azp_aq_i8).to(device='cpu') @ bq_i32.to(device='cpu'))).to(
            dtype=out_dtype, device='cuda')

    out = ops.cutlass_scaled_mm(aq_i8,
                                bq_i8,
                                scale_a,
                                scale_b,
                                out_dtype=out_dtype,
                                bias=azp_bias[0, :])
    torch.testing.assert_close(out, baseline_dq, rtol=1e-2, atol=1e0)
    torch.testing.assert_close(out, baseline_q, rtol=1e-2, atol=1e0)


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("k", [64, 128, 256])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("azp_per_token", [True, False])
def test_cutlass_int8_azp(m: int, n: int, k: int, out_dtype: torch.dtype,
                          use_bias: bool, azp_per_token: bool):
    m_azp = m if azp_per_token else 1
    scale_a = torch.randn((m_azp, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, n), device="cuda", dtype=torch.float32) / 10

    aq_i8 = rand_int8((m, k))
    aq_i32 = aq_i8.to(dtype=torch.int32)
    aq_f32 = aq_i8.to(dtype=torch.float32)

    bq_i8 = rand_int8((n, k)).t()
    bq_i32 = bq_i8.to(dtype=torch.int32)
    bq_f32 = bq_i8.to(dtype=torch.float32)
    b_dq = scale_b * bq_f32

    azp_a = torch.rand(
        (m_azp, 1), device="cuda", dtype=torch.float32) * 10 + 1.5
    azp_aq_i8 = (azp_a / scale_a).to(dtype=torch.int8)
    azp_a = azp_aq_i8.to(dtype=torch.float32) * scale_a  # correct for rounding

    a_dq = scale_a * (aq_i32 - azp_aq_i8).to(dtype=torch.float32)
    torch.testing.assert_close(a_dq,
                               scale_a * aq_f32 - azp_a,
                               rtol=1e-4,
                               atol=1e-3)

    if use_bias:
        bias = torch.rand((1, n), device="cuda", dtype=out_dtype) * 10 + 2.5
    else:
        bias = torch.zeros((1, n), device="cuda", dtype=out_dtype)

    baseline_dq = (torch.mm(a_dq, b_dq) + bias).to(out_dtype)

    # int32 mm not supported on CUDA
    a_noazp_i32_cpu = (aq_i32 - azp_aq_i8).to(device='cpu')
    cq = (a_noazp_i32_cpu @ bq_i32.to(device='cpu')).to(device='cuda')
    baseline_q = (scale_a * scale_b * cq + bias).to(dtype=out_dtype)

    # Hadamard is just the sum of the cols
    azp_adj_i32 = bq_i32.sum(dim=0, keepdim=True, dtype=torch.int32)
    azp_i32 = azp_aq_i8.to(dtype=torch.int32)
    func_bias = bias if use_bias else None

    if azp_per_token:
        out = ops.cutlass_scaled_mm_azp(aq_i8, bq_i8, scale_a, scale_b,
                                        out_dtype, azp_adj_i32, azp_i32,
                                        func_bias)
    else:
        azp_with_adj_i32 = azp_i32 * azp_adj_i32
        out = ops.cutlass_scaled_mm_azp(aq_i8, bq_i8, scale_a, scale_b,
                                        out_dtype, azp_with_adj_i32, None,
                                        func_bias)

    # bfloat16 precision is 7-bit mantissa -> 2^-8 ~ 0.4%
    # float16 precision is 10-bit mantissa -> 2^-11 ~ 0.05%
    rtol = 1e-2 if out_dtype == torch.bfloat16 else 1e-3
    atol = 1e-3
    torch.testing.assert_close(out, baseline_dq, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, baseline_q, rtol=rtol, atol=atol)

    if azp_per_token:
        opcheck(torch.ops._C.cutlass_scaled_mm_azp,
                (out, aq_i8, bq_i8, scale_a, scale_b, azp_adj_i32, azp_i32,
                 func_bias))
    else:
        opcheck(torch.ops._C.cutlass_scaled_mm_azp,
                (out, aq_i8, bq_i8, scale_a, scale_b, azp_with_adj_i32, None,
                 func_bias))


# Test working with a subset of A and B
def test_cutlass_subset():
    big_m, big_n, big_k = 1024, 1024, 1024
    m, n, k = 512, 512, 512

    whole_a = to_int8(torch.randn((big_m, big_k), device="cuda") * 5)
    whole_b = to_int8(torch.randn((big_n, big_k), device="cuda").t() * 5)
    a = whole_a[0:m, 0:k]
    b = whole_b[0:k, 0:n]

    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10

    out = ops.cutlass_scaled_mm(a,
                                b,
                                scale_a,
                                scale_b,
                                out_dtype=torch.bfloat16)
    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=torch.bfloat16)

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)


# Test to make sure cuda graphs work
class CutlassLayer(torch.nn.Module):

    def __init__(self, b, scale_a, scale_b, out_dtype):
        super().__init__()
        self.b = b
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.out_dtype = out_dtype

    def forward(self, a):
        return ops.cutlass_scaled_mm(a, self.b, self.scale_a, self.scale_b,
                                     self.out_dtype)


@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_cuda_graph(per_act_token: bool, per_out_ch: bool):
    m, n, k = 512, 512, 512

    a = to_int8(torch.randn((m, k), device="cuda"))
    b = to_int8(torch.randn((n, k), device="cuda").t())

    m_a_scales = m if per_act_token else 1
    n_b_scales = n if per_out_ch else 1

    scale_a = (torch.randn(
        (m_a_scales, 1), device="cuda", dtype=torch.float32) / 10)
    scale_b = (torch.randn(
        (1, n_b_scales), device="cuda", dtype=torch.float32) / 10)

    # Construct a trivial model with a single layer that calls a CUTLASS kernel
    model = CutlassLayer(b, scale_a, scale_b, torch.bfloat16)

    # Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(a)
    out.zero_()
    g.replay()

    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b * b.to(dtype=torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)


def test_cutlass_support_opcheck():
    opcheck(torch.ops._C.cutlass_scaled_mm_supports_fp8, (capability, ))
