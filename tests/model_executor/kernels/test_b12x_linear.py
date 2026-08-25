# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_MXFP4_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    B12xFp8BlockScaledMMKernel,
    B12xMxFp4LinearKernel,
    B12xMxfp8LinearKernel,
    B12xNvFp4LinearKernel,
    B12xTensorFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Mxfp8LinearLayerConfig,
    init_fp8_linear_kernel,
    init_mxfp4_linear_kernel,
    init_mxfp8_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
)
from vllm.platforms import PlatformEnum


@pytest.mark.parametrize(
    ("kernel_cls", "kernels", "before", "after", "initializer", "kwargs"),
    [
        (
            B12xMxFp4LinearKernel,
            _POSSIBLE_MXFP4_KERNELS[PlatformEnum.CUDA],
            "HummingMxFp4LinearKernel",
            "EmulationMxfp4LinearKernel",
            init_mxfp4_linear_kernel,
            {"activation_quant_key": kMxfp4Dynamic},
        ),
        (
            B12xNvFp4LinearKernel,
            _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA],
            "FbgemmNvFp4LinearKernel",
            "EmulationNvFp4LinearKernel",
            init_nvfp4_linear_kernel,
            {},
        ),
        (
            B12xMxfp8LinearKernel,
            _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA],
            "MarlinMxfp8LinearKernel",
            "EmulationMxfp8LinearKernel",
            init_mxfp8_linear_kernel,
            {},
        ),
        (
            B12xTensorFP8ScaledMMLinearKernel,
            _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA],
            "CutlassFP8ScaledMMLinearKernel",
            "PerTensorTorchFP8ScaledMMLinearKernel",
            init_fp8_linear_kernel,
            {
                "activation_quant_key": kFp8StaticTensorSym,
                "weight_quant_key": kFp8StaticTensorSym,
                "input_dtype": torch.bfloat16,
                "out_dtype": torch.bfloat16,
                "weight_shape": (2048, 2048),
            },
        ),
        (
            B12xFp8BlockScaledMMKernel,
            _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA],
            "CutlassFp8BlockScaledMMKernel",
            "MarlinFP8ScaledMMLinearKernel",
            init_fp8_linear_kernel,
            {
                "activation_quant_key": kFp8Dynamic128Sym,
                "weight_quant_key": kFp8Static128BlockSym,
                "input_dtype": torch.bfloat16,
                "out_dtype": torch.bfloat16,
                "weight_shape": (2048, 2048),
            },
        ),
    ],
)
def test_b12x_backend_registration_priority_and_selection(
    monkeypatch,
    default_vllm_config,
    kernel_cls,
    kernels,
    before: str,
    after: str,
    initializer,
    kwargs: dict,
) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    assert kernel_cls in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    names = [kernel.__name__ for kernel in kernels]
    assert names.index(before) < names.index(kernel_cls.__name__) < names.index(after)

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        kernel_cls,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        kernel_cls,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    assert isinstance(initializer(**kwargs), kernel_cls)


def test_b12x_module_lookup_is_dynamo_safe(monkeypatch) -> None:
    import vllm.utils.b12x as b12x_utils

    module = types.ModuleType("b12x.gemm.blockscaled")
    module.run = lambda x: x + 1  # type: ignore[attr-defined]
    monkeypatch.setitem(
        b12x_utils._B12X_SUBMODULES,
        "b12x.gemm.blockscaled",
        module,
    )

    @torch.compile(backend="eager", fullgraph=True)
    def forward(x: torch.Tensor) -> torch.Tensor:
        blockscaled = b12x_utils.get_b12x_blockscaled()
        assert blockscaled is not None
        return blockscaled.run(x)  # type: ignore[attr-defined]

    x = torch.ones(1)
    torch.testing.assert_close(forward(x), x + 1)


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


def test_b12x_block_fp8_checks_runtime_support(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

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


def test_b12x_tensor_fp8_process_weights_packs_modelopt_layout(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

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
    assert layer.b12x_warmup_provider is kernel
    assert len(calls) == 1
    weight, output_scale = calls[0]
    torch.testing.assert_close(weight, original_weight.T.contiguous())
    torch.testing.assert_close(output_scale, torch.tensor([0.125]))
    assert layer.weight.numel() == 0
    assert layer.weight_scale.numel() == 0
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale.weight_loader is scale_loader
    torch.testing.assert_close(layer.input_scale, torch.tensor(0.5))


def test_b12x_tensor_fp8_apply_quantizes_and_uses_packed_weight(
    monkeypatch,
) -> None:
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

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


def test_b12x_mxfp8_can_implement_supported_config() -> None:
    can_implement, reason = B12xMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig()
    )

    assert can_implement
    assert reason is None


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
    assert layer.b12x_warmup_provider is kernel
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

    assert layer.b12x_warmup_provider is kernel
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
    import vllm.model_executor.kernels.linear.scaled_mm.b12x as b12x_mod

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


def test_b12x_mxfp4_requires_dynamic_activations() -> None:
    config = types.SimpleNamespace(activation_quant_key=kMxfp4Dynamic)
    can_implement, reason = B12xMxFp4LinearKernel.can_implement(config)

    assert can_implement
    assert reason is None

    config.activation_quant_key = None
    can_implement, reason = B12xMxFp4LinearKernel.can_implement(config)

    assert not can_implement
    assert reason == "B12X MXFP4 GEMM requires dynamic MXFP4 activations"


@pytest.mark.parametrize(
    ("kernel_cls", "module_name", "scale_dtype"),
    [
        (
            B12xMxFp4LinearKernel,
            "vllm.model_executor.kernels.linear.mxfp4.b12x",
            torch.uint8,
        ),
        (
            B12xNvFp4LinearKernel,
            "vllm.model_executor.kernels.linear.nvfp4.b12x",
            torch.float8_e4m3fn,
        ),
    ],
)
def test_b12x_fp4_processes_scale_and_preserves_loader(
    monkeypatch,
    kernel_cls,
    module_name: str,
    scale_dtype: torch.dtype,
) -> None:
    scale = torch.empty((48, 8), dtype=scale_dtype)
    swizzled_scale = torch.empty((128, 8), dtype=scale_dtype)
    intrinsics = types.SimpleNamespace(swizzle_block_scale=lambda value: swizzled_scale)
    monkeypatch.setattr(
        importlib.import_module(module_name),
        "_import_b12x_intrinsics",
        lambda: intrinsics,
    )
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.shared_expert.down_proj"
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    weight_loader = object()
    layer.weight_scale.weight_loader = weight_loader
    kernel = object.__new__(kernel_cls)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale.data_ptr() == swizzled_scale.data_ptr()
    assert layer.weight_scale.weight_loader is weight_loader
    assert layer.b12x_warmup_provider is kernel


def test_b12x_mxfp4_apply_calls_native_blockscaled_gemm(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp4.b12x as b12x_mod
    import vllm.utils.flashinfer as flashinfer_utils

    calls: list[tuple] = []
    x_packed = torch.empty((6, 64), dtype=torch.uint8)
    x_scale_storage = torch.empty((128, 4), dtype=torch.uint8)

    def mm_mxfp4(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((6, 48), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(
        flashinfer_utils,
        "flashinfer_mxfp4_quantize",
        lambda *args, **kwargs: (x_packed, x_scale_storage),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_mxfp4=mm_mxfp4),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = 48
    layer.weight = torch.empty((48, 64), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 4), dtype=torch.uint8)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.ones(48, dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxFp4LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        x_packed,
        x_scale_storage,
        layer.weight,
        layer.weight_scale,
    )
    assert kwargs == {"out_dtype": torch.bfloat16}


def test_b12x_nvfp4_can_implement_supported_config() -> None:
    can_implement, reason = B12xNvFp4LinearKernel.can_implement(None)

    assert can_implement
    assert reason is None


def test_b12x_backend_preserves_w4a16_fallback(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        MarlinNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_b12x_nvfp4_apply_calls_native_blockscaled_gemm(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    calls: list[tuple] = []
    quant_calls: list[tuple] = []
    x_packed = torch.empty((6, 64), dtype=torch.uint8)
    x_scale_storage = torch.empty((128, 8), dtype=torch.float8_e4m3fn)

    def quant(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return x_packed, x_scale_storage

    def mm_nvfp4(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full((6, 48), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", quant)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm_nvfp4=mm_nvfp4),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = 48
    layer.weight = torch.empty((48, 64), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 8), dtype=torch.float8_e4m3fn)
    layer.input_global_scale_inv = torch.tensor(2.0)
    layer.alpha = torch.tensor(0.25)
    x = torch.empty((2, 3, 256), dtype=torch.bfloat16)[..., ::2]
    bias = torch.ones(48, dtype=torch.bfloat16)
    kernel = object.__new__(B12xNvFp4LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    assert len(quant_calls) == 1
    quant_args, quant_kwargs = quant_calls[0]
    assert quant_args[0].shape == (6, 128)
    assert quant_args[0].data_ptr() == x.data_ptr()
    assert quant_args[1] is layer.input_global_scale_inv
    assert not quant_args[0].is_contiguous()
    assert quant_kwargs == {"is_sf_swizzled_layout": True}
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (
        x_packed,
        x_scale_storage,
        layer.weight,
        layer.weight_scale,
        layer.alpha,
    )
    assert kwargs == {"out_dtype": torch.bfloat16}
