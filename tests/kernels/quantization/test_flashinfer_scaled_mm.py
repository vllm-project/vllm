# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Flashinfer FP8 gemms requires compute capability of 10.0 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
) -> None:
    set_random_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((n, k), dtype=dtype, device=device) / k

    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)

    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=dtype)

    if use_bias:
        bias = torch.randn((n,), dtype=dtype, device=device)
        expected_out = expected_out + bias
    else:
        bias = None

    import flashinfer

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp8_mm(
            a_fp8,
            b_fp8.t(),
            a_scale,
            b_scale,
            dtype,
            bias=bias,
        )

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


def _block_fp8_layer(weight, scales, scale_name):
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight.clone(), False))
    layer.register_parameter(scale_name, torch.nn.Parameter(scales.clone(), False))
    layer.weight_block_size = [128, 128]
    return layer


@pytest.mark.skipif(
    not current_platform.is_device_capability(107),
    reason="The small-M MXFP8/DeepGEMM linear is specific to SM107",
)
@pytest.mark.parametrize("num_tokens", [1, 16, 17, 24])
@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
@pytest.mark.parametrize("zero_input", [False, True])
@torch.inference_mode()
def test_sm107_mxfp8_block_linear_matches_quantized_reference(
    num_tokens, scale_name, zero_input, default_vllm_config
):
    """Exercise post-load conversion and both sides of the M=16 dispatch limit.

    Large batches must retain the existing DeepGEMM result exactly. Small
    batches are checked against FP32 accumulation of independently MXFP8-
    quantized activations and the actual requantized weights, using the same
    1e-2 tolerances as the other FlashInfer FP8 GEMM test in this module.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
        DeepGemmFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferMxfp8DeepGemmBlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    supported, reason = FlashInferMxfp8DeepGemmBlockScaledMMKernel.is_supported()
    assert supported, reason
    set_random_seed(42)
    n, k = 256, 512
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    # Non-power-of-two checkpoint scales must be requantized at load, not
    # silently truncated to exponent bytes on the MXFP8 path.
    scales = torch.tensor(
        [[0.011, 0.023, 0.037, 0.051], [0.062, 0.079, 0.093, 0.109]],
        device="cuda",
    )
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(n, k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    kernel = FlashInferMxfp8DeepGemmBlockScaledMMKernel(config)
    baseline = DeepGemmFp8BlockScaledMMKernel(config)
    assert kernel.use_deep_gemm_e8m0 and baseline.use_deep_gemm_e8m0
    layer = _block_fp8_layer(weight, scales, scale_name)
    baseline_layer = _block_fp8_layer(weight, scales, scale_name)
    # DeepGEMM requantizes this FP32 buffer in place before replacing its
    # parameter with packed scales; keep the original buffer for the reference.
    requantized_scales = getattr(baseline_layer, scale_name)
    baseline.process_weights_after_loading(baseline_layer)
    kernel.process_weights_after_loading(layer)
    torch.testing.assert_close(
        layer.weight.view(torch.uint8),
        baseline_layer.weight.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        getattr(layer, scale_name), getattr(baseline_layer, scale_name), atol=0, rtol=0
    )
    assert layer.weight_scale_mxfp8.dtype == torch.uint8
    # Undo the F8_128x4 permutation independently of the conversion helper.
    exponents = (
        layer.weight_scale_mxfp8.view(n // 128, k // 128, 32, 4, 4)
        .transpose(1, 3)
        .reshape(n, k // 32)
    )
    torch.testing.assert_close(
        torch.exp2(exponents.float() - 127),
        requantized_scales.repeat_interleave(128, 0).repeat_interleave(4, 1),
        atol=0,
        rtol=0,
    )
    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    if zero_input:
        x.zero_()
    bias = (
        torch.linspace(-0.125, 0.125, n, device="cuda", dtype=torch.bfloat16)
        if num_tokens in (16, 17)
        else None
    )
    actual = kernel.apply_weights(layer, x, bias)
    if num_tokens > 16:
        expected = baseline.apply_weights(baseline_layer, x, bias)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif zero_input:
        expected = torch.zeros_like(actual)
        if bias is not None:
            expected += bias
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        xq, xs = _mxfp8_e4m3_quantize_torch(x)
        x_dequant = xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, 1)
        w_dequant = layer.weight.float() * requantized_scales.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        expected = (x_dequant @ w_dequant.t()).to(torch.bfloat16)
        if bias is not None:
            expected += bias
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(actual).all()

    if num_tokens == 16 and scale_name == "weight_scale_inv" and not zero_input:
        # The optimized branch must observe updated graph inputs, including
        # its quantization, instead of replaying a stale intermediate buffer.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kernel.apply_weights(layer, x, bias)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = kernel.apply_weights(layer, x, bias)
        x.mul_(0.5)
        expected = kernel.apply_weights(layer, x, bias)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
