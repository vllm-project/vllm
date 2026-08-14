# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    init_fp8_linear_kernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8.b12x import (
    B12xMxfp8LinearKernel,
    _b12x_mxfp8_expected_m,
    warmup_b12x_mxfp8_linear,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_block import (
    B12xFp8BlockScaledMMKernel,
    _run_b12x_fp8_block_scaled_mm,
    warmup_b12x_block_fp8_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor import (
    B12xTensorFP8ScaledMMLinearKernel,
    warmup_b12x_tensor_fp8_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm.utils.b12x import b12x_warmup_token_counts


def test_b12x_backend_maps_mxfp8_kernel() -> None:
    assert B12xMxfp8LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    assert B12xMxfp8LinearKernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]


def test_b12x_backend_maps_tensor_fp8_kernel() -> None:
    assert B12xTensorFP8ScaledMMLinearKernel in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    assert B12xTensorFP8ScaledMMLinearKernel in _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA]


@pytest.mark.parametrize(
    ("kernels", "before", "b12x", "after"),
    [
        (
            _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA],
            "CutlassFP8ScaledMMLinearKernel",
            "B12xTensorFP8ScaledMMLinearKernel",
            "PerTensorTorchFP8ScaledMMLinearKernel",
        ),
        (
            _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA],
            "CutlassFp8BlockScaledMMKernel",
            "B12xFp8BlockScaledMMKernel",
            "MarlinFP8ScaledMMLinearKernel",
        ),
        (
            _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA],
            "MarlinMxfp8LinearKernel",
            "B12xMxfp8LinearKernel",
            "EmulationMxfp8LinearKernel",
        ),
    ],
)
def test_b12x_fp8_fallback_priority(
    kernels: list[type],
    before: str,
    b12x: str,
    after: str,
) -> None:
    names = [kernel.__name__ for kernel in kernels]

    assert names.index(before) < names.index(b12x) < names.index(after)


@torch.inference_mode()
def test_b12x_backend_does_not_intercept_unquantized_bf16(
    default_vllm_config,
    dist_init,
) -> None:
    default_vllm_config.kernel_config.linear_backend = "b12x"
    device = current_platform.device_type
    layer = ReplicatedLinear(
        128,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=None,
        prefix="bf16_linear",
    ).to(device)
    layer.weight.data.normal_()
    x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)

    output, output_bias = layer(x)
    expected = torch.nn.functional.linear(x, layer.weight)

    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert output_bias is None
    torch.testing.assert_close(output, expected)


def test_b12x_explicit_backend_selects_per_tensor_fp8(
    monkeypatch,
    default_vllm_config,
) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        B12xTensorFP8ScaledMMLinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        B12xTensorFP8ScaledMMLinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8StaticTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(2048, 2048),
    )

    assert isinstance(kernel, B12xTensorFP8ScaledMMLinearKernel)


def test_b12x_tensor_fp8_can_implement_supported_config() -> None:
    config = FP8ScaledMMLinearLayerConfig(
        activation_quant_key=kFp8StaticTensorSym,
        weight_quant_key=kFp8StaticTensorSym,
        weight_shape=(64, 128),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )

    can_implement, reason = B12xTensorFP8ScaledMMLinearKernel.can_implement(config)

    assert can_implement
    assert reason is None


def test_b12x_explicit_backend_selects_block_fp8(
    monkeypatch,
    default_vllm_config,
) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        B12xFp8BlockScaledMMKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_fp8_linear_kernel(
        activation_quant_key=kFp8Dynamic128Sym,
        weight_quant_key=kFp8Static128BlockSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(2048, 2048),
    )

    assert isinstance(kernel, B12xFp8BlockScaledMMKernel)


def test_b12x_block_fp8_checks_runtime_support(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_block as b12x_mod

    platform = types.SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "current_platform", platform)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(is_supported=lambda: False),
    )

    supported, reason = B12xFp8BlockScaledMMKernel.is_supported()

    assert not supported
    assert reason == "B12X regular block-FP8 GEMM is not supported"


def test_b12x_block_fp8_requires_matching_supported_dtypes() -> None:
    def config(input_dtype: torch.dtype, out_dtype: torch.dtype):
        return FP8ScaledMMLinearLayerConfig(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            weight_shape=(256, 128),
            input_dtype=input_dtype,
            out_dtype=out_dtype,
        )

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.float32, torch.float32)
    )
    assert not can_implement
    assert reason == "Supports only bf16/fp16 input dtype"

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.bfloat16, torch.float16)
    )
    assert not can_implement
    assert reason == "Input and output dtype must match"

    can_implement, reason = B12xFp8BlockScaledMMKernel.can_implement(
        config(torch.float16, torch.float16)
    )
    assert can_implement
    assert reason is None


def test_b12x_block_fp8_requires_aligned_features() -> None:
    def can_implement(weight_shape: tuple[int, int]):
        config = FP8ScaledMMLinearLayerConfig(
            activation_quant_key=kFp8Dynamic128Sym,
            weight_quant_key=kFp8Static128BlockSym,
            weight_shape=weight_shape,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        return B12xFp8BlockScaledMMKernel.can_implement(config)

    assert can_implement((256, 192)) == (
        False,
        "Input features must be a positive multiple of 128",
    )
    assert can_implement((192, 256)) == (
        False,
        "Output features must be a positive multiple of 128",
    )


def test_b12x_warmup_token_counts_cover_serving_regimes() -> None:
    assert b12x_warmup_token_counts(
        max_tokens=2048,
        cudagraph_capture_sizes=[1, 2, 8, 128],
    ) == (1, 2, 8, 128, 2048)


def test_warmup_b12x_block_fp8_dedupes_weight_signatures(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_block as b12x_mod

    calls = []

    def run(a, weight, a_scale, weight_scale, out_dtype):
        calls.append((a.shape, weight, a_scale.shape, weight_scale, out_dtype))
        return torch.empty((a.shape[0], weight.shape[0]), dtype=out_dtype)

    platform = types.SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "current_platform", platform)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(),
    )
    monkeypatch.setattr(b12x_mod, "_run_b12x_fp8_block_scaled_mm", run)

    def layer(in_features: int, out_features: int):
        return types.SimpleNamespace(
            b12x_block_fp8_linear=True,
            weight=torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.empty(
                (out_features // 128, in_features // 128), dtype=torch.float32
            ),
        )

    layer_a = layer(128, 256)
    layer_b = layer(256, 128)
    modules = [
        layer_a,
        layer_a,
        layer_b,
        types.SimpleNamespace(),
    ]
    model = types.SimpleNamespace(modules=lambda: iter(modules))

    warmed = warmup_b12x_block_fp8_linear(
        model,
        max_tokens=32,
        cudagraph_capture_sizes=[2, 8],
        output_dtype=torch.bfloat16,
    )

    assert warmed == 8
    assert [call[0][0] for call in calls] == [1, 2, 8, 32] * 2
    assert [call[2][0] for call in calls] == [1, 2, 8, 32] * 2
    assert calls[0][1] is layer_a.weight
    assert calls[4][1] is layer_b.weight
    assert calls[0][3] is layer_a.weight_scale_inv
    assert calls[4][3] is layer_b.weight_scale_inv
    assert all(call[4] == torch.bfloat16 for call in calls)


def test_b12x_tensor_fp8_process_weights_packs_modelopt_layout(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor as b12x_mod

    calls = []
    packed = types.SimpleNamespace(out_features=64)

    def pack(weight: torch.Tensor, output_scale: torch.Tensor):
        calls.append((weight, output_scale))
        return packed

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.qkv_proj"
    original_weight = (
        torch.randn((128, 64), dtype=torch.float32).clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    layer.weight = torch.nn.Parameter(original_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(weight_shape=(64, 128))
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_scale_ub",
    )

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_tensor_fp8_packed_weight is packed
    assert len(calls) == 1
    weight, output_scale = calls[0]
    torch.testing.assert_close(weight, original_weight.T.contiguous())
    torch.testing.assert_close(output_scale, torch.tensor([0.125]))
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.5))


def test_warmup_b12x_tensor_fp8_dedupes_weight_signatures(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor as b12x_mod

    calls = []

    def prewarm(packed_weight, token_counts, *, out_dtype, stream):
        del stream
        calls.append((packed_weight, tuple(token_counts), out_dtype))
        return len(tuple(token_counts))

    platform = types.SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "current_platform", platform)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: types.SimpleNamespace(prewarm=prewarm),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=object()),
    )

    def packed(in_features: int, padded_in_features: int, out_features: int):
        return types.SimpleNamespace(
            in_features=in_features,
            padded_in_features=padded_in_features,
            out_features=out_features,
            values=torch.empty(1),
        )

    packed_a = packed(128, 128, 256)
    packed_b = packed(160, 256, 512)
    modules = [
        types.SimpleNamespace(b12x_tensor_fp8_packed_weight=packed_a),
        types.SimpleNamespace(b12x_tensor_fp8_packed_weight=packed_a),
        types.SimpleNamespace(b12x_tensor_fp8_packed_weight=packed_b),
        types.SimpleNamespace(),
    ]
    model = types.SimpleNamespace(modules=lambda: iter(modules))

    warmed = warmup_b12x_tensor_fp8_linear(
        model,
        max_tokens=2048,
        cudagraph_capture_sizes=[1, 2],
    )

    assert warmed == 6
    assert calls == [
        (packed_a, (1, 2, 2048), torch.bfloat16),
        (packed_b, (1, 2, 2048), torch.bfloat16),
    ]


def test_b12x_tensor_fp8_apply_quantizes_and_uses_packed_weight(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor as b12x_mod

    calls = []

    def mm(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype,
        expected_m: int,
        stream: object = None,
    ) -> torch.Tensor:
        del stream
        calls.append((source, packed_weight, bias, out_dtype, expected_m))
        return torch.full(
            (source.shape[0], packed_weight.out_features),
            3.0,
            dtype=out_dtype,
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_tensor_fp8",
        lambda: types.SimpleNamespace(mm=mm),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=object()),
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    layer = torch.nn.Module()
    packed = types.SimpleNamespace(out_features=48)
    layer.b12x_tensor_fp8_packed_weight = packed
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 48), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    layer.input_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    x_q = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    bias = torch.empty((48,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xTensorFP8ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.layer_param_names = (
        "weight",
        "weight_scale",
        "input_scale",
        "input_scale_ub",
    )
    kernel.quant_fp8 = lambda source, scale, scale_ub: (x_q, scale)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    assert output.dtype == torch.bfloat16
    assert len(calls) == 1
    source, called_packed, called_bias, out_dtype, expected_m = calls[0]
    assert source.data_ptr() == x_q.data_ptr()
    assert called_packed is packed
    assert called_bias is bias
    assert out_dtype == torch.bfloat16
    assert expected_m == 6


def test_b12x_mxfp8_explicit_backend_selects_kernel(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "can_implement",
        classmethod(lambda cls, c: (True, None)),
    )

    kernel = init_mxfp8_linear_kernel()

    assert isinstance(kernel, B12xMxfp8LinearKernel)


def test_b12x_mxfp8_can_implement_supported_config() -> None:
    can_implement, reason = B12xMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig()
    )

    assert can_implement
    assert reason is None


def test_b12x_mxfp8_expected_m_uses_live_m() -> None:
    assert _b12x_mxfp8_expected_m(0) == 1
    assert _b12x_mxfp8_expected_m(1) == 1
    assert _b12x_mxfp8_expected_m(2) == 2
    assert _b12x_mxfp8_expected_m(8) == 8
    assert _b12x_mxfp8_expected_m(9) == 9
    assert _b12x_mxfp8_expected_m(128) == 128
    assert _b12x_mxfp8_expected_m(129) == 129
    assert _b12x_mxfp8_expected_m(2048) == 2048


def test_warmup_b12x_mxfp8_linear_dedupes_weight_signatures(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []

    def mm(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
        stream: object = None,
    ) -> torch.Tensor:
        del stream
        calls.append((source.shape, packed_weight, bias, expected_m))
        return source.new_empty((source.shape[0], packed_weight.out_features))

    platform = types.SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "current_platform", platform)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(mm=mm),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=object()),
    )

    def packed(in_features: int, padded_in_features: int, out_features: int):
        return types.SimpleNamespace(
            in_features=in_features,
            padded_in_features=padded_in_features,
            out_features=out_features,
            weight=types.SimpleNamespace(values=torch.empty(1)),
        )

    packed_a = packed(128, 128, 256)
    packed_b = packed(128, 128, 512)
    modules = [
        types.SimpleNamespace(b12x_mxfp8_packed_weight=packed_a),
        types.SimpleNamespace(b12x_mxfp8_packed_weight=packed_a),
        types.SimpleNamespace(b12x_mxfp8_packed_weight=packed_b),
        types.SimpleNamespace(),
    ]
    model = types.SimpleNamespace(modules=lambda: iter(modules))

    warmed = warmup_b12x_mxfp8_linear(
        model,
        max_tokens=2048,
        cudagraph_capture_sizes=[1, 2],
    )

    assert warmed == 6
    assert [call[0] for call in calls] == [
        torch.Size([1, 128]),
        torch.Size([2, 128]),
        torch.Size([2048, 128]),
        torch.Size([1, 128]),
        torch.Size([2, 128]),
        torch.Size([2048, 128]),
    ]
    assert [call[3] for call in calls] == [1, 2, 2048, 1, 2, 2048]
    assert calls[0][1] is packed_a
    assert calls[3][1] is packed_b


def test_b12x_mxfp8_support_check_reports_missing_import(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "_import_b12x_mxfp8", lambda: None)

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "Install the B12X backend with `pip install vllm[b12x]`"


def test_b12x_mxfp8_support_respects_runtime_probe(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(is_supported=lambda: False),
    )

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "b12x.gemm.mxfp8_linear is not supported"


def test_b12x_mxfp8_process_weights_packs_modelopt_layout(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []
    packed = types.SimpleNamespace(out_features=48)

    def pack(weight: torch.Tensor, weight_scale: torch.Tensor):
        calls.append((weight, weight_scale))
        return packed

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.qkv_proj"
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((64, 8), dtype=torch.uint8),
        requires_grad=False,
    )
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale.weight_loader = scale_loader
    kernel = object.__new__(B12xMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    assert len(calls) == 1
    weight, weight_scale = calls[0]
    assert weight.shape == (48, 128)
    assert weight_scale.shape == (48, 4)
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.uint8
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader


def test_b12x_mxfp8_reload_reuses_packed_tensor_addresses(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    @dataclass(frozen=True)
    class PackedWeight:
        values: torch.Tensor
        scales: torch.Tensor
        out_features: int

    def pack(weight: torch.Tensor, weight_scale: torch.Tensor) -> PackedWeight:
        return PackedWeight(
            values=weight.clone(),
            scales=weight_scale.clone(),
            out_features=int(weight.shape[0]),
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(pack_weight=pack),
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.down_proj"
    layer.weight = torch.nn.Parameter(
        torch.zeros((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((48, 4), dtype=torch.uint8),
        requires_grad=False,
    )
    kernel = object.__new__(B12xMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)
    packed = layer.b12x_mxfp8_packed_weight
    values_ptr = packed.values.data_ptr()
    scales_ptr = packed.scales.data_ptr()

    layer.weight = torch.nn.Parameter(
        torch.ones((48, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((48, 4), 3, dtype=torch.uint8),
        requires_grad=False,
    )
    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    assert packed.values.data_ptr() == values_ptr
    assert packed.scales.data_ptr() == scales_ptr
    torch.testing.assert_close(
        packed.values,
        torch.ones((48, 128), dtype=torch.float8_e4m3fn),
    )
    torch.testing.assert_close(
        packed.scales,
        torch.full((48, 4), 3, dtype=torch.uint8),
    )
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0


@pytest.fixture
def _mock_b12x_cuda_fp8_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    monkeypatch.setattr(
        fp8_utils,
        "current_platform",
        types.SimpleNamespace(
            is_fp8_fnuz=lambda: False,
            is_rocm=lambda: False,
            fp8_dtype=lambda: torch.float8_e4m3fn,
            is_xpu=lambda: False,
            is_cuda_alike=lambda: True,
        ),
    )


@pytest.mark.usefixtures("_mock_b12x_cuda_fp8_platform")
def test_b12x_block_fp8_process_weights_keeps_native_block_layout() -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.empty((1, 1), dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_block_size = [128, 128]
    weight_loader = object()
    scale_loader = object()
    layer.weight.weight_loader = weight_loader
    layer.weight_scale_inv.weight_loader = scale_loader
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_block_fp8_linear
    assert layer.weight.shape == (128, 128)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.shape == (1, 1)
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale_inv.weight_loader is scale_loader


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.uint8])
@pytest.mark.usefixtures("_mock_b12x_cuda_fp8_platform")
def test_b12x_block_fp8_upcasts_e8m0_weight_scales(scale_dtype) -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    scale_bytes = torch.tensor([[125]], dtype=torch.uint8)
    layer.weight_scale_inv = torch.nn.Parameter(
        scale_bytes.view(scale_dtype),
        requires_grad=False,
    )
    layer.weight_block_size = [128, 128]
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale_inv.dtype == torch.float32
    torch.testing.assert_close(
        layer.weight_scale_inv,
        torch.tensor([[0.25]], dtype=torch.float32),
    )


def test_b12x_mxfp8_apply_uses_packed_weight(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []

    def mxfp8_linear(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
        stream: object = None,
    ) -> torch.Tensor:
        del stream
        calls.append((source, packed_weight, bias, expected_m))
        return source.new_full((source.shape[0], packed_weight.out_features), 3.0)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(mm=mxfp8_linear),
    )

    layer = torch.nn.Module()
    packed = types.SimpleNamespace(out_features=48)
    layer.b12x_mxfp8_packed_weight = packed
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.empty((48,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    assert output.dtype == x.dtype
    assert len(calls) == 1
    source, called_packed, called_bias, expected_m = calls[0]
    assert source.shape == (6, 128)
    assert called_packed is packed
    assert called_bias is bias
    assert expected_m == 6


def test_b12x_block_fp8_apply_uses_b12x_recipe_api(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_block as b12x_mod

    calls = []

    def mm_block_fp8(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full(
            (args[0].shape[0], args[2].shape[0]),
            13.0,
            dtype=kwargs["out_dtype"],
        )

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_block_fp8=mm_block_fp8),
    )

    a = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((6, 1), dtype=torch.float32)
    weight_scale = torch.empty((2, 1), dtype=torch.float32)
    kernel = object.__new__(B12xFp8BlockScaledMMKernel)
    kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)

    output = kernel.apply_block_scaled_mm(a, weight, a_scale, weight_scale)

    assert output.shape == (6, 256)
    assert output.dtype == torch.bfloat16
    assert len(calls) == 1
    assert calls[0] == (
        (a, a_scale, weight, weight_scale),
        {"out_dtype": torch.bfloat16},
    )
    torch.testing.assert_close(output, torch.full_like(output, 13.0))


def test_b12x_block_fp8_helper_uses_regular_compact_scale_api(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x_block as b12x_mod

    calls = []

    def mm_block_fp8(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((6, 256), 17.0, dtype=torch.bfloat16)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_block_fp8=mm_block_fp8),
    )

    a = torch.empty((6, 128), dtype=torch.float8_e4m3fn)
    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    a_scale = torch.empty((6, 1), dtype=torch.float32)
    weight_scale = torch.empty((2, 1), dtype=torch.float32)
    output = _run_b12x_fp8_block_scaled_mm(
        a,
        weight,
        a_scale,
        weight_scale,
        torch.bfloat16,
    )

    assert output.shape == (6, 256)
    args, kwargs = calls[0]
    assert args == (a, a_scale, weight, weight_scale)
    assert kwargs == {
        "out_dtype": torch.bfloat16,
    }
    torch.testing.assert_close(output, torch.full_like(output, 17.0))
