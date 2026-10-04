# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)

from vllm import _custom_ops as ops
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    expose_input_quant_key,
)
from vllm.model_executor.layers.fusion.relu2_nvfp4_quant import relu_squared_nvfp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [
    (128, 128, 64),
    (128, 128, 128),
    (256, 128, 64),
    (128, 256, 128),
    (1, 128, 128),
]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96), (2, 128, 64), (3, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("shape", [(1, 1344), (17, 5376), (129, 48)])
@pytest.mark.parametrize("producer", ["relu2", "generic", "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_cutedsl_relu2_prequantized_bypasses_quantizer(
    monkeypatch, shape, producer, with_bias
):
    """The CuTe-DSL consumer preserves output and padding without requantizing."""
    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as consumer_module

    supported, reason = FlashInferCuteDslNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    m, k = shape
    if producer == "silu" and k % 64:
        pytest.skip("SiLU producer does not initialize padded K scales")
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    n = 130  # Exercise output slicing as well as activation K padding.
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.input_global_scale_inv = torch.tensor(64.0, device="cuda")
    weight_scale_inv = torch.tensor(512.0, device="cuda")
    weight, layer.weight_scale = ops.scaled_fp4_quant(w, weight_scale_inv)
    layer.weight, padding = pad_nvfp4_weight_for_cutlass(weight)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.alpha = 1.0 / (layer.input_global_scale_inv * weight_scale_inv)
    kernel = FlashInferCuteDslNvFp4LinearKernel(NvFp4LinearLayerConfig())
    activation = torch.compile(lambda value: torch.square(torch.relu(value)))
    bias = torch.randn(n, device="cuda", dtype=x.dtype) if with_bias else None
    if producer == "relu2":
        expected = kernel.apply_weights(layer, activation(x), bias)
        qa = relu_squared_nvfp4_quant(x, layer)
    elif producer == "generic":
        expected = kernel.apply_weights(layer, x, bias)
        data, scale = ops.scaled_fp4_quant(
            x, layer.input_global_scale_inv, backend="flashinfer-cutedsl"
        )
        qa = QuantizedActivation(data, scale, x.dtype, x.shape, kNvfp4Dynamic)
    else:
        from vllm.model_executor.layers.fusion.fused_act_quant import (
            _silu_and_mul_nvfp4_dynamic,
        )

        qa = _silu_and_mul_nvfp4_dynamic(torch.cat((x, x), dim=-1), layer)
        # Consumer/layout test, not a change to the existing SiLU producer's
        # numerical contract: compare the exact same valid QAct with direct GEMM.
        expected = flashinfer_scaled_fp4_mm(
            pad_nvfp4_activation_for_cutlass(qa.data, padding),
            layer.weight,
            qa.scale,
            layer.weight_scale,
            layer.alpha,
            qa.orig_dtype,
            backend="cute-dsl",
        )[:, :n].contiguous()
        if bias is not None:
            expected = expected + bias

    def reject_requant(*args, **kwargs):
        raise AssertionError("CuTe-DSL requantized an existing QuantizedActivation")

    monkeypatch.setattr(consumer_module, "scaled_fp4_quant", reject_requant)
    actual = kernel.apply_weights(layer, qa, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", [48, 128])
@pytest.mark.parametrize(
    "layout,mode",
    [
        ("contiguous", "native"),
        ("contiguous", "cuda"),
        ("rank3", "native"),
        ("rank3", "cuda"),
        ("strided", "native"),
    ],
)
@torch.inference_mode()
def test_cutedsl_silu_public_dispatch_preserves_tensor_path(
    dtype, k, layout, mode, record_property
):
    """CuTe keeps the exact preexisting SiLU Tensor path for every layout."""
    supported, reason = FlashInferCuteDslNvFp4LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    set_random_seed(42)
    m, n = 17, 64
    x = torch.randn((m, 2 * k), device="cuda", dtype=dtype)
    x[:, :7] = torch.tensor([-16, -1, -1 / 256, 0, 1 / 256, 1, 16], device=x.device)
    if layout == "rank3":
        x = x.reshape(1, m, 2 * k)
    elif layout == "strided":
        storage = torch.full((m, 4 * k), 100, device=x.device, dtype=dtype)
        storage[:, ::2] = x
        x = storage[:, ::2]
    layer = torch.nn.Module()
    layer.input_global_scale_inv = torch.tensor(
        64.0, device=x.device, dtype=torch.float32
    )
    weight_scale_inv = torch.tensor(512.0, device=x.device, dtype=torch.float32)
    weight, layer.weight_scale = ops.scaled_fp4_quant(
        torch.randn((n, k), device=x.device, dtype=dtype), weight_scale_inv
    )
    layer.weight, _ = pad_nvfp4_weight_for_cutlass(weight)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.alpha = 1.0 / (layer.input_global_scale_inv * weight_scale_inv)
    kernel = FlashInferCuteDslNvFp4LinearKernel(NvFp4LinearLayerConfig())
    expose_input_quant_key(layer, kernel)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["all" if mode == "cuda" else "none"]
        )
    )
    with set_current_vllm_config(config):
        act = SiluAndMul(compile_native=False)
        activated = act(x)
        expected = kernel.apply_weights(layer, activated)
        dispatched = maybe_fused_act_quant(act, x, layer)
        assert isinstance(dispatched, torch.Tensor)
        torch.testing.assert_close(dispatched, activated, rtol=0, atol=0)
        actual = kernel.apply_weights(layer, dispatched)
    error = actual.float() - expected.float()
    record_property("max_abs_output_error", error.abs().max().item())
    record_property(
        "relative_l2_output_error",
        (error.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
    is_sf_128x4_layout,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_sf,
        a_global_scale,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_128x4_layout=is_sf_128x4_layout,
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", ["cute-dsl", "cutlass", "cudnn", "trtllm", "b12x"])
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
    backend: str,
    autotune: bool,
) -> None:
    if "trtllm" in backend and dtype == torch.float16:
        pytest.skip("Only torch.bfloat16 is supported for TRTLLM FP4 GEMM operations")
    if backend == "cute-dsl" and not current_platform.is_device_capability_family(100):
        pytest.skip("FlashInfer cutedsl backend is only supported on SM10x")
    if backend == "b12x" and not current_platform.has_device_capability(120):
        pytest.skip("b12x FP4 GEMM requires SM120+ (CC 12.0+)")
    if backend == "b12x" and not has_flashinfer_b12x_gemm():
        pytest.skip("b12x FP4 GEMM backend not available in installed FlashInfer")

    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    # cutlass and b12x use swizzled scales directly; trtllm needs them unswizzled.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(
        a_dtype, a_global_scale, is_sf_swizzled_layout=True, backend=backend
    )
    is_sf_128x4_layout = not (backend == "trtllm" and m <= 32)

    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(
        b_dtype, b_global_scale, is_sf_swizzled_layout=True
    )

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
        is_sf_128x4_layout,
    )

    import flashinfer

    if "trtllm" in backend:
        epilogue_tile_m = 128
        b_fp4 = flashinfer.shuffle_matrix_a(b_fp4.view(torch.uint8), epilogue_tile_m)
        b_scale_interleaved = convert_swizzled_to_linear(
            b_scale_interleaved, n, k, block_size
        )
        b_scale_interleaved = (
            flashinfer.shuffle_matrix_sf_a(
                b_scale_interleaved.view(torch.uint8), epilogue_tile_m
            )
            .reshape(b_scale_interleaved.shape)
            .view(torch.float8_e4m3fn)
        )

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            a_scale_interleaved,
            b_scale_interleaved,
            alpha,
            dtype,
            backend=backend,
        )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_flashinfer_cutedsl_nvfp4_w4a16_linear(
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    supported, reason = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    weight_quant_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight.abs().max()).to(
        torch.float32
    )
    weight_global_scale = weight_quant_scale.reciprocal()
    weight_fp4, weight_scale = ops.scaled_fp4_quant(
        weight,
        weight_quant_scale,
        is_sf_swizzled_layout=False,
    )
    weight_ref = dequantize_to_dtype(
        weight_fp4,
        weight_scale,
        weight_global_scale,
        dtype=torch.bfloat16,
        swizzle=False,
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(weight_fp4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        weight_global_scale, requires_grad=False
    )
    kernel = FlashInferCuteDslNvFp4W4A16LinearKernel(NvFp4LinearLayerConfig())

    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(layer, x)

    expected = F.linear(x, weight_ref)
    assert output.shape == (m, n)
    torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-1)
