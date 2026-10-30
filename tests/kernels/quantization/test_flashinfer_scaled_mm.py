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


@pytest.mark.parametrize(
    "m,n,k,threshold,backend,override,uses_cublaslt",
    [
        (127, 256, 256, 128, "auto", None, False),
        (128, 256, 256, 128, "auto", None, True),
        (129, 256, 256, 256, "auto", None, False),
        (256, 256, 256, 256, "auto", None, True),
        (128, 256, 256, None, "auto", None, False),
        (128, 136, 256, 128, "auto", None, False),
        (128, 144, 160, 128, "auto", None, True),
        (128, 256, 256, 128, "flashinfer_cutedsl", None, False),
        (128, 256, 256, 128, "auto", "flashinfer_cutedsl", False),
        (128, 256, 256, 128, "flashinfer_cutedsl", "auto", True),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_mxfp8_cublaslt_threshold_preserves_linear_output(
    m, n, k, threshold, backend, override, uses_cublaslt, dtype
):
    """The public config selects the GEMM while preserving quantized inputs."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    set_random_seed(42)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    weight, scales = mxfp8_e4m3_quantize(torch.randn(n, k, dtype=dtype, device="cuda"))
    layer = torch.nn.Module()
    layer.params_dtype = dtype
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    bias = torch.randn(n, dtype=dtype, device="cuda")
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend=backend,
            linear_backend_per_quant={"mxfp8": override} if override else None,
            mxfp8_cublaslt_min_m=threshold,
            enable_flashinfer_autotune=True,
        )
    )
    with set_current_vllm_config(config):
        candidate = init_mxfp8_linear_kernel(weight_shape=(n, k))
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="flashinfer_cutedsl",
            enable_flashinfer_autotune=True,
        )
    )
    with set_current_vllm_config(config):
        reference = init_mxfp8_linear_kernel(weight_shape=(n, k))
    candidate.process_weights_after_loading(layer)
    expected = reference.apply_weights(layer, x, bias)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        actual = candidate.apply_weights(layer, x, bias)
    assert any(e.key == "aten::_scaled_mm" for e in p.key_averages()) == uses_cublaslt
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate.apply_weights(layer, x, bias)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(
        captured, reference.apply_weights(layer, x, bias), atol=0, rtol=0
    )


@torch.inference_mode()
def test_mxfp8_cublaslt_keeps_flashinfer_warmup():
    """A large-M dispatch still tunes FI before returning to cuBLASLt at runtime."""
    import flashinfer

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.utils.flashinfer import mm_mxfp8

    a, sa = mxfp8_e4m3_quantize(
        torch.randn(128, 128, dtype=torch.bfloat16, device="cuda"),
        is_sf_swizzled_layout=True,
    )
    b, sb = mxfp8_e4m3_quantize(
        torch.randn(256, 128, dtype=torch.bfloat16, device="cuda"),
        is_sf_swizzled_layout=True,
    )

    def call():
        return mm_mxfp8(a, b.t(), sa, sb, torch.bfloat16, backend="cublaslt")

    with (
        flashinfer.autotune(tuning_buckets=(128,)),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p,
    ):
        tuned = call()
    assert not any(e.key == "aten::_scaled_mm" for e in p.key_averages())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        actual = call()
    assert any(e.key == "aten::_scaled_mm" for e in p.key_averages())
    torch.testing.assert_close(actual, tuned, atol=0, rtol=0)
