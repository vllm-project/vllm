# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_scaled_mm kernel.

Run `pytest tests/kernels/quantization/test_triton_scaled_mm.py`.
"""

import importlib

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

device = current_platform.device_type

triton_scaled_mm_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
)
triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


def get_8bit_types():
    types = [torch.int8]
    if current_platform.supports_fp8():
        types.append(current_platform.fp8_dtype())
    return types


# This test is to check regressions for int8 support on ROCm.
@pytest.mark.parametrize(
    "model_path",
    [
        "neuralmagic/Llama-3.2-1B-quantized.w8a8",
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [10])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Should only run on ROCm")
def test_rocm_compressed_tensors_w8a8(
    vllm_runner, example_prompts, model_path, max_tokens, num_logprobs
):
    dtype = "bfloat16"

    with vllm_runner(model_path, dtype=dtype) as vllm_model:
        vllm_model.generate_greedy_logprobs(example_prompts, max_tokens, num_logprobs)


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    set_random_seed(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    c_check = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    c_actual = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_default_config", [True, False])
def test_scaled_mm_adds_bias_before_rounding(in_dtype, use_default_config):
    # The accumulator is 1 + 3 * 2**-10 and the bias 2**-8. Rounding to bf16
    # first gives 1.0 (1 + 2**-8 is a tie, to even); adding first, 1 + 2**-7.
    a = torch.ones((1, 32), device=device).to(in_dtype)
    col = torch.tensor([64.0] * 16 + [2.0, 1.0] + [0.0] * 14, device=device)
    b = col[:, None].expand(32, 16).to(in_dtype).contiguous()
    scale_a = torch.ones((1, 1), device=device)
    scale_b = torch.full((1, 1), 2**-10, device=device)
    bias = torch.full((16,), 2**-8, device=device, dtype=torch.bfloat16)
    args = (a, b, scale_a, scale_b, torch.bfloat16, bias)

    out = triton_scaled_mm(*args, use_default_config=use_default_config)
    expected = torch_scaled_mm(*args)
    assert (expected == 1 + 2**-7).all()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# TD operand loads must be bit-exact vs the plain masked-load path.
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton scaled_mm runs on CUDA-alike or XPU.",
)
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]
)
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm_td_matches_plain(M, N, K, in_dtype, use_scalar_scale_a, use_bias):
    dev = current_platform.device_type
    out_dtype = torch.bfloat16
    set_random_seed(0)

    is_fp = torch.tensor([1, 1], dtype=in_dtype).is_floating_point()
    if is_fp:
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=dev)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=dev)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=dev)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=dev)

    scale_a = (
        torch.rand((1, 1), device=dev)
        if use_scalar_scale_a
        else 0.25 * torch.rand((M, 1), device=dev)
    )
    scale_b = 0.25 * torch.rand((N, 1), device=dev)
    bias = torch.rand((N,), device=dev, dtype=out_dtype) if use_bias else None

    out_plain = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=False)
    out_td = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=True)
    torch.testing.assert_close(out_td, out_plain, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_fp8(),
    reason="Requires ROCm FP8",
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 256, 8192),
        (17, 256, 8192),
        (33, 256, 496),
        (15, 2048, 512),
        (129, 4096, 512),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_triton_fp8_per_token_matches_torch(m, n, k, out_dtype, use_bias):
    """Compare identical FP8 operands against the Torch linear backends."""
    set_random_seed(0)
    dtype = current_platform.fp8_dtype()
    a = (0.2 * torch.randn(m, k, device=device)).to(dtype)
    b = (0.2 * torch.randn(n, k, device=device)).to(dtype).t()
    sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)
    bias = torch.randn(n, device=device, dtype=out_dtype) if use_bias else None
    actual = torch.ops.vllm.triton_fp8_per_token_scaled_mm(
        a, b, sa, sb, out_dtype, bias, False, True
    )
    reference_cls = (
        RowWiseTorchFP8ScaledMMLinearKernel
        if out_dtype == torch.bfloat16
        else ChannelWiseTorchFP8ScaledMMLinearKernel
    )
    with set_current_vllm_config(VllmConfig()):
        reference = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            weight_shape=(n, k),
            input_dtype=out_dtype,
            out_dtype=out_dtype,
            force_kernel=reference_cls,
        )
    assert isinstance(reference, reference_cls)
    expected = reference.apply_scaled_mm(
        A=a, B=b, As=sa, Bs=sb, out_dtype=out_dtype, bias=bias, output_shape=[m, n]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("scale_ndim", [1, 2])
def test_scaled_mm_noncontiguous_scales_preserve_values(in_dtype, scale_ndim):
    """Strided scale views must use their logical values, including in the fallback."""
    set_random_seed(0)
    m, n, k = 33, 97, 128
    a = torch.randint(-4, 4, (m, k), device=device).to(in_dtype)
    b = torch.randint(-4, 4, (n, k), device=device).to(in_dtype).t()
    sa = torch.rand(2 * m, device=device)[1::2]
    sb = torch.rand(2 * n, device=device)[1::2]
    if scale_ndim == 2:
        sa, sb = sa[:, None], sb[:, None]
    assert not sa.is_contiguous() and not sb.is_contiguous()
    expected = torch_scaled_mm(
        a, b, sa.reshape(-1, 1), sb.reshape(-1, 1), torch.bfloat16
    )
    actual = triton_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Requires FP8")
@pytest.mark.parametrize(
    "m,n,k,group_m,n_stripe,padding,offset",
    [
        pytest.param(129, 160, 128, 3, 0, 0, 0, id="group_tail"),
        pytest.param(129, 160, 128, 1, 4, 0, 0, id="stripe_tail"),
        pytest.param(64, 160, 128, 1, 0, 0, 0, id="aligned"),
        pytest.param(65, 160, 128, 1, 0, 0, 0, id="m_tail"),
        pytest.param(64, 161, 128, 1, 0, 0, 0, id="n_tail"),
        pytest.param(64, 160, 144, 1, 0, 0, 0, id="k_tail"),
        pytest.param(64, 160, 128, 1, 0, 16, 0, id="padded"),
        pytest.param(64, 160, 128, 1, 0, 0, 1, id="unaligned"),
    ],
)
def test_scaled_mm_tile_and_layout_boundaries(
    m, n, k, group_m, n_stripe, padding, offset
):
    """Cover partial groups/stripes and mask incomplete operand tiles."""
    set_random_seed(0)
    dtype = current_platform.fp8_dtype()
    a, b = (
        torch.randint(-4, 4, (size * (k + padding) + offset,), device=device)
        .to(dtype)[offset:]
        .view(size, k + padding)[:, :k]
        for size in (m, n)
    )
    b = b.t()
    sa, sb = torch.rand(m, 1, device=device), torch.rand(n, 1, device=device)
    actual = triton_scaled_mm(
        a,
        b,
        sa,
        sb,
        torch.bfloat16,
        use_default_config=False,
        use_td=False,
        block_size_m=32,
        block_size_n=32,
        block_size_k=64,
        group_m=group_m,
        n_stripe=n_stripe,
        unmasked_loads=True,
    )
    expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
