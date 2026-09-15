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
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
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


@pytest.mark.parametrize("m", [1, 4, 5, 8, 16, 17, 129])
@pytest.mark.parametrize("fuse_silu", [False, True])
@pytest.mark.parametrize("silu_cutoff", [None, 16])
@torch.inference_mode()
def test_cutedsl_dynamic_precision_preserves_canonical_weights(
    m, fuse_silu, silu_cutoff, monkeypatch
):
    """Both sides of the cutoff reuse weights and keep their own scale semantics."""
    if current_platform.get_device_capability().to_int() not in (120, 121):
        pytest.skip("Dynamic CuTe NVFP4 requires SM120/121")
    from types import SimpleNamespace

    import flashinfer
    from flashinfer.quantization.fp4_quantization import silu_and_mul_nvfp4_quantize

    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.kernels.linear.nvfp4 import dynamic_cutedsl

    config = KernelConfig(
        nvfp4_dynamic_max_tokens=4, nvfp4_dynamic_silu_max_tokens=silu_cutoff
    )
    monkeypatch.setattr(
        dynamic_cutedsl,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            kernel_config=config,
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
    )
    kernel = dynamic_cutedsl.FlashInferCuTeDynamicNvFp4LinearKernel(
        NvFp4LinearLayerConfig()
    )
    cutoff = silu_cutoff if fuse_silu and silu_cutoff is not None else 4

    if not flashinfer.mm_bf16_fp4.is_backend_supported("cute-dsl-native", 121):
        pytest.skip("Native-layout CuTe W4A16 backend is unavailable")
    torch.manual_seed(42)
    n, k = 256, 256
    x = torch.randn(m, k * (2 if fuse_silu else 1), device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a_inv = torch.tensor([8.0], device="cuda")
    w_global = torch.tensor([1 / 16.0], device="cuda")
    alpha = w_global / a_inv
    w_fp4, w_scale = flashinfer.nvfp4_quantize(
        w, w_global.reciprocal(), backend="cute-dsl"
    )
    w_before, sf_before = w_fp4.clone(), w_scale.view(torch.uint8).clone()

    def reference():
        if m <= cutoff:
            activation = F.silu(x[:, :k]) * x[:, k:] if fuse_silu else x
        else:
            if fuse_silu:
                a_fp4, a_scale = silu_and_mul_nvfp4_quantize(x, a_inv)
            else:
                a_fp4, a_scale = flashinfer.nvfp4_quantize(x, a_inv, backend="cute-dsl")
            activation = dequantize_nvfp4_to_dtype(
                a_fp4,
                a_scale,
                a_inv,
                dtype=torch.bfloat16,
                device="cuda",
                block_size=16,
                is_sf_128x4_layout=True,
            )
        weight = dequantize_nvfp4_to_dtype(
            w_fp4,
            w_scale,
            w_global.reciprocal(),
            dtype=torch.bfloat16,
            device="cuda",
            block_size=16,
            is_sf_128x4_layout=True,
        )
        return (activation.float() @ weight.float().T).bfloat16()

    layer = SimpleNamespace(
        weight=w_fp4,
        weight_scale=w_scale,
        weight_global_scale=w_global,
        input_global_scale_inv=a_inv,
        alpha=alpha,
    )
    actual = kernel.apply_silu_or_linear(layer, x, fuse_silu)
    torch.testing.assert_close(actual, reference(), atol=0.03, rtol=0.01)
    assert torch.equal(w_fp4, w_before)
    assert torch.equal(w_scale.view(torch.uint8), sf_before)
    if m in (4, 5, 16, 17):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = kernel.apply_silu_or_linear(layer, x, fuse_silu)
        x.mul_(0.5)
        w_scale.view(torch.uint8).fill_(0x38)
        w_global.mul_(2)
        alpha.mul_(2)
        graph.replay()
        torch.testing.assert_close(replay, reference(), atol=0.03, rtol=0.01)


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
