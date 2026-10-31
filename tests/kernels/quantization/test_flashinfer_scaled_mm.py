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


def _dsv41_mxfp8_linear(m: int, n: int, k: int):
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.nvidia.ops.mxfp8_linear import (
        cublaslt_mxfp8_apply,
    )

    set_random_seed(42)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight, scale = mxfp8_e4m3_quantize(
        torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    )
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    kernel = FlashInferCutedslMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    def apply():
        return cublaslt_mxfp8_apply(kernel, layer, x, None, min_m=128)

    return x, layer, kernel, apply


def _uses_scaled_mm(fn):
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities) as prof:
        output = fn()
    return output, any(e.key == "aten::_scaled_mm" for e in prof.key_averages())


@pytest.mark.parametrize(
    "m,n,k,expect_cublaslt",
    [
        (127, 256, 256, False),
        (128, 256, 256, True),
        (128, 136, 256, False),
        (128, 144, 160, True),
    ],
)
@torch.inference_mode()
def test_dsv41_mxfp8_cublaslt_matches_cutedsl(m, n, k, expect_cublaslt):
    """Large DeepSeek-V4.1 MXFP8 GEMMs move to cuBLASLt with bitwise-equal
    output, eagerly and under CUDA graph replay."""
    x, layer, kernel, apply = _dsv41_mxfp8_linear(m, n, k)
    actual, used_cublaslt = _uses_scaled_mm(apply)
    assert used_cublaslt == expect_cublaslt
    assert torch.equal(actual, kernel.apply_weights(layer, x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    x.copy_(torch.randn_like(x))
    graph.replay()
    assert torch.equal(captured, kernel.apply_weights(layer, x))


@torch.inference_mode()
def test_dsv41_mxfp8_cublaslt_keeps_flashinfer_autotune():
    """FlashInfer autotuning still profiles CuTe-DSL for the small-M runtime."""
    import flashinfer

    _, _, _, apply = _dsv41_mxfp8_linear(128, 256, 128)
    with flashinfer.autotune(tuning_buckets=(128,)):
        tuned, used_cublaslt = _uses_scaled_mm(apply)
    assert not used_cublaslt
    actual, used_cublaslt = _uses_scaled_mm(apply)
    assert used_cublaslt and torch.equal(actual, tuned)
