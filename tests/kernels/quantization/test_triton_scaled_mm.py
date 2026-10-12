# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_scaled_mm kernel.

Run `pytest tests/kernels/quantization/test_triton_scaled_mm.py`.
"""

import importlib

import pytest
import torch

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
@pytest.mark.parametrize("use_heuristic", [True, False])
def test_scaled_mm_adds_bias_before_rounding(in_dtype, use_heuristic):
    # The accumulator is 1 + 3 * 2**-10 and the bias 2**-8. Rounding to bf16
    # first gives 1.0 (1 + 2**-8 is a tie, to even); adding first, 1 + 2**-7.
    a = torch.ones((1, 32), device=device).to(in_dtype)
    col = torch.tensor([64.0] * 16 + [2.0, 1.0] + [0.0] * 14, device=device)
    b = col[:, None].expand(32, 16).to(in_dtype).contiguous()
    scale_a = torch.ones((1, 1), device=device)
    scale_b = torch.full((1, 1), 2**-10, device=device)
    bias = torch.full((16,), 2**-8, device=device, dtype=torch.bfloat16)
    args = (a, b, scale_a, scale_b, torch.bfloat16, bias)

    out = triton_scaled_mm(*args, use_heuristic=use_heuristic)
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


# The launch must stay opaque to torch.compile (inline tracing breaks split_graph).
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton scaled_mm runs on CUDA-alike or XPU.",
)
@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("scalar_scales", [True, False])
def test_scaled_mm_custom_op_under_compile(M, use_bias, scalar_scales):
    import vllm.model_executor.kernels.linear.scaled_mm.triton  # noqa: F401

    N, K = 2048, 4096
    out_dtype = torch.bfloat16
    set_random_seed(0)
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=device)
    # Checkpoint layout: a transposed [N, K] view.
    b = torch.randint(-32, 32, (N, K), dtype=torch.int8, device=device).t()
    scale_a = 0.25 * torch.rand((1 if scalar_scales else M, 1), device=device)
    scale_b = 0.25 * torch.rand((1 if scalar_scales else N, 1), device=device)
    bias = torch.rand((N,), device=device, dtype=out_dtype) if use_bias else None

    def fn(a, b, scale_a, scale_b, bias):
        return torch.ops.vllm.triton_scaled_mm_func(
            a, b, scale_a, scale_b, out_dtype, bias
        )

    expected = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    compiled = torch.compile(fn, fullgraph=True)(a, b, scale_a, scale_b, bias)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)


# The fake impl must match the real output without touching a device.
@pytest.mark.parametrize("M,N,K", [(1, 32, 64), (7, 128, 256)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_scaled_mm_fake_impl_meta(M, N, K, out_dtype):
    import vllm.model_executor.kernels.linear.scaled_mm.triton  # noqa: F401

    a = torch.empty((M, K), dtype=torch.int8, device="meta")
    b = torch.empty((N, K), dtype=torch.int8, device="meta").t()
    scale_a = torch.empty((M, 1), device="meta")
    scale_b = torch.empty((N, 1), device="meta")
    out = torch.ops.vllm.triton_scaled_mm_func(a, b, scale_a, scale_b, out_dtype, None)
    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    assert out.device.type == "meta"


# The regression itself: apply_weights must reach the opaque op under compile.
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton scaled_mm runs on CUDA-alike or XPU.",
)
@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("is_static", [True, False])
def test_int8_kernel_apply_weights_opaque_under_compile(M, use_bias, is_static):
    from vllm.model_executor.kernels.linear.scaled_mm import (
        Int8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )

    N, K = 2048, 4096
    dtype = torch.bfloat16
    set_random_seed(0)
    config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=is_static, is_channelwise=True, input_symmetric=True
    )
    kernel = TritonInt8ScaledMMLinearKernel(
        config, ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
    )
    layer = torch.nn.Module()
    layer.logical_widths = [N]
    layer.weight = torch.nn.Parameter(
        torch.randint(-32, 32, (N, K), dtype=torch.int8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        0.01 * torch.rand((N, 1), device=device), requires_grad=False
    )
    layer.input_scale = (
        torch.nn.Parameter(torch.full((1,), 0.02, device=device), requires_grad=False)
        if is_static
        else None
    )
    layer.input_zero_point = None
    kernel.process_weights_after_loading(layer)

    x = torch.randn((M, K), device=device, dtype=dtype)
    bias = torch.rand((N,), device=device, dtype=dtype) if use_bias else None

    expected = kernel.apply_weights(layer, x, bias)

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(
        lambda x, bias: kernel.apply_weights(layer, x, bias),
        backend=backend,
        fullgraph=True,
    )(x, bias)

    targets = [str(n.target) for g in graphs for n in g.graph.nodes]
    assert any("triton_scaled_mm_func" in t for t in targets), targets
    if is_static:
        torch.testing.assert_close(compiled, expected, rtol=0, atol=0)
    else:
        # Compiled dynamic quant keeps the per-token scale in fp32; eager rounds
        # it to the activation dtype. Quantized values match, outputs differ ~1e-3.
        atol = 2e-2 * expected.abs().max().item()
        torch.testing.assert_close(compiled, expected, rtol=2e-2, atol=atol)
