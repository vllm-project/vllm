# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_scaled_mm kernel.

Run `pytest tests/kernels/quantization/test_triton_scaled_mm.py`.
"""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

device = current_platform.device_type

triton_scaled_mm_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
)
triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm
fp8_utils_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.utils.fp8_utils"
)


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
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


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


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Requires FP8")
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_triton_fp8_per_token_backend_preserves_linear_shape(out_dtype, use_bias):
    """Select the backend and validate quantization, weight layout and 3D output."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonFp8PerTokenScaledMMKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    set_random_seed(0)
    n, k = 256, 128
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend="triton"))
    ):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            input_dtype=out_dtype,
            out_dtype=out_dtype,
            weight_shape=(n, k),
        )
    assert isinstance(kernel, TritonFp8PerTokenScaledMMKernel)
    layer = torch.nn.Module()
    layer.weight = (
        (0.2 * torch.randn(n, k, device=device)).to(current_platform.fp8_dtype()).t()
    )
    layer.weight_scale = torch.rand(n, 1, device=device) + 0.5
    x = torch.randn(3, 11, k, device=device, dtype=out_dtype)
    bias = torch.randn(n, device=device, dtype=out_dtype) if use_bias else None
    x_q, x_s = kernel.quant_fp8(x.reshape(-1, k))
    expected = torch_scaled_mm(
        x_q, layer.weight, x_s, layer.weight_scale, out_dtype, bias
    ).view(3, 11, n)
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    compiled = torch.compile(kernel.apply_weights, fullgraph=True)
    # Eager and compiled native quantization can round FP8 differently.
    compiled_q, compiled_s = torch.compile(kernel.quant_fp8, fullgraph=True)(
        x.reshape(-1, k)
    )
    compiled_expected = torch_scaled_mm(
        compiled_q, layer.weight, compiled_s, layer.weight_scale, out_dtype, bias
    ).view(3, 11, n)
    torch.testing.assert_close(
        compiled(layer, x, bias), compiled_expected, rtol=1e-2, atol=1e-2
    )
    qa = QuantizedActivation(x_q, x_s, x.dtype, x.shape, kFp8DynamicTokenSym)
    torch.testing.assert_close(
        compiled(layer, qa, bias), expected, rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="Requires FP8")
@pytest.mark.parametrize("n", [256, 8192])
@pytest.mark.parametrize("tuned", [False, True])
def test_triton_fp8_per_token_dynamic_shapes_keep_runtime_tiles(
    monkeypatch, tmp_path, n, tuned
):
    """Prefill compilation must preserve decode tiles and GPU graph replay."""
    importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.triton")
    monkeypatch.setattr(fp8_utils_module, "_W8A8_PER_TOKEN_FP8_CONFIG_DIR", tmp_path)
    fp8_utils_module.get_w8a8_per_token_fp8_configs.cache_clear()
    tuned_config = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "num_warps": 4,
        "num_stages": 1,
    }
    if tuned:
        filename = fp8_utils_module.get_w8a8_per_token_fp8_config_filename(
            n,
            128,
            fp8_utils_module.get_device_name_as_file_name(),
            current_platform.fp8_dtype(),
            torch.bfloat16,
        )
        (tmp_path / filename).write_text(json.dumps({1: tuned_config, 8: tuned_config}))
    set_random_seed(0)
    launches = []
    real_kernel = triton_scaled_mm_module.scaled_mm_kernel

    class RecordingKernel:
        def __getitem__(self, grid):
            launch = real_kernel[grid]

            def record(*args, **kwargs):
                launches.append(
                    tuple(kwargs[f"BLOCK_SIZE_{dim}"] for dim in ("M", "N", "K"))
                )
                return launch(*args, **kwargs)

            return record

    monkeypatch.setattr(triton_scaled_mm_module, "scaled_mm_kernel", RecordingKernel())
    k = 128
    b = (0.2 * torch.randn(n, k, device=device)).to(current_platform.fp8_dtype()).t()
    sb = torch.rand(n, 1, device=device) + 0.5

    def run(a, sa):
        return torch.ops.vllm.w8a8_triton_per_token_scaled_mm_func(
            a, b, sa, sb, torch.bfloat16, None
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for m in (256, 1, 8, 32, 33, 64, 65, 128, 129):
        a = (0.2 * torch.randn(m, k, device=device)).to(current_platform.fp8_dtype())
        sa = torch.rand(m, 1, device=device) + 0.5
        expected_tile = (
            (64, 64 if n < 8192 else 128, 256)
            if m <= 32
            else (64, 64, 256)
            if m <= 64
            else (64, 128, 128)
            if m <= 128
            else (128, 128, 128)
        )
        if tuned and m <= 8:
            expected_tile = (32, 64, 128)
        expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
        actual = compiled(a, sa)
        assert launches[-1] == expected_tile
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(a, sa)
        assert launches[-1] == expected_tile
        count = len(launches)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=1e-2)
        assert len(launches) == count
    fp8_utils_module.get_w8a8_per_token_fp8_configs.cache_clear()


def test_per_token_fp8_config_respects_device_dtype_and_measured_m(
    monkeypatch, tmp_path
):
    """Partial decode tuning must not override prefill or other devices/dtypes."""
    module = fp8_utils_module
    monkeypatch.setattr(module, "_W8A8_PER_TOKEN_FP8_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(module, "get_device_name_as_file_name", lambda: "test_device")
    module.get_w8a8_per_token_fp8_configs.cache_clear()
    dtype = torch.float8_e4m3fn
    filename = module.get_w8a8_per_token_fp8_config_filename(
        256, 128, "test_device", dtype, torch.bfloat16
    )
    configs = {
        1: {"BLOCK_SIZE_M": 16},
        2: {"BLOCK_SIZE_M": 32},
        8: {"BLOCK_SIZE_M": 64},
    }
    (tmp_path / filename).write_text(json.dumps(configs))
    get = lambda m, out=torch.bfloat16: module.get_w8a8_per_token_fp8_config(
        m, 256, 128, dtype, out
    )
    assert get(1) == configs[1]
    assert get(3) == configs[2]
    assert get(8) == configs[8]
    assert get(0) is None
    assert get(16) is None
    assert get(1, torch.float16) is None
    monkeypatch.setattr(module, "get_device_name_as_file_name", lambda: "other_device")
    assert get(1) is None
    module.get_w8a8_per_token_fp8_configs.cache_clear()


@pytest.mark.parametrize("platform", ["cuda", "rocm", "xpu", "cpu"])
def test_triton_fp8_per_token_support_matches_block_scaled(monkeypatch, platform):
    from vllm.model_executor.kernels.linear.scaled_mm import triton as module

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(
            is_cuda_alike=lambda: platform in ("cuda", "rocm"),
            is_xpu=lambda: platform == "xpu",
        ),
    )
    assert module.TritonFp8PerTokenScaledMMKernel.is_supported() == (
        module.TritonFp8BlockScaledMMKernel.is_supported()
    )


@pytest.mark.parametrize("backend", ["auto", "triton", "torch"])
def test_triton_fp8_per_token_default_precedes_rowwise(monkeypatch, backend):
    """Auto prefers Triton; an explicit torch backend must still select RowWise."""
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels import linear
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )
    from vllm.platforms.interface import PlatformEnum

    monkeypatch.setattr(linear.current_platform, "_enum", PlatformEnum.ROCM)
    monkeypatch.setattr(linear.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "")
    eligible = (
        linear.TritonFp8PerTokenScaledMMKernel,
        linear.RowWiseTorchFP8ScaledMMLinearKernel,
    )
    for kernel in linear._POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM]:
        monkeypatch.setattr(
            kernel,
            "is_supported",
            classmethod(lambda cls, cc=None: (cls in eligible, None)),
        )
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=(256, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    with set_current_vllm_config(
        VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    ):
        selected = linear.choose_scaled_mm_linear_kernel(
            config,
            linear._POSSIBLE_FP8_KERNELS,
            compute_capability=90,
            quantization="fp8_w8a8",
        )
    expected = eligible[1] if backend == "torch" else eligible[0]
    assert selected is expected


@pytest.mark.parametrize("in_dtype", get_8bit_types())
def test_scaled_mm_explicit_tiles_without_heuristic(in_dtype):
    """Manual tuning overrides must work for both shared INT8 and FP8 paths."""
    set_random_seed(0)
    a = torch.randint(-4, 4, (3, 128), device=device).to(in_dtype)
    b = torch.randint(-4, 4, (96, 128), device=device).to(in_dtype).t()
    sa = torch.rand(3, 1, device=device)
    sb = torch.rand(96, 1, device=device)
    actual = triton_scaled_mm(
        a,
        b,
        sa,
        sb,
        torch.bfloat16,
        use_heuristic=False,
        block_size_m=32,
        block_size_n=64,
        block_size_k=64,
        num_warps=4,
        num_stages=1,
    )
    expected = torch_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
