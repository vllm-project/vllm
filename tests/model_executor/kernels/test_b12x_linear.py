# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass

import pytest
import torch

from vllm.config.quantization import PhaseSpec, QuantSpec
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
    B12xNvFp4W4A16LinearKernel,
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
    kNvfp4Dynamic,
    kNvfp4DynamicTensor,
)
from vllm.platforms import PlatformEnum
from vllm.utils.torch_utils import _encode_layer_name


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
            {"weight_shape": (2048, 2048)},
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
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "b12x"
    )
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
        return blockscaled.run(x)

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
        Mxfp8LinearLayerConfig(weight_shape=(256, 512))
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


def test_b12x_backend_selects_w4a16(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "b12x"
    )
    monkeypatch.setitem(
        linear_mod._POSSIBLE_NVFP4_KERNELS,
        PlatformEnum.CUDA,
        [B12xNvFp4LinearKernel, B12xNvFp4W4A16LinearKernel, MarlinNvFp4LinearKernel],
    )
    monkeypatch.setattr(
        B12xNvFp4W4A16LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert isinstance(kernel, B12xNvFp4W4A16LinearKernel)


def _b12x_phase(max_decode_tokens: int | None, **formats: str) -> PhaseSpec:
    formats.setdefault("prefill", "nvfp4")
    return PhaseSpec(max_decode_tokens=max_decode_tokens, **formats)


def _with_phase(kernel, phase: PhaseSpec):
    """``kernel`` with ``phase`` set as the oracle sets it, without the config
    checks."""
    kernel.phase = phase
    return kernel


def _use_phase(monkeypatch, phase: PhaseSpec) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod, "_user_linear_spec", lambda: QuantSpec(phase=phase))


@pytest.fixture
def _select_b12x_w4a16(monkeypatch, default_vllm_config):
    import vllm.model_executor.kernels.linear as linear_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "b12x"
    )
    monkeypatch.setitem(
        linear_mod._POSSIBLE_NVFP4_KERNELS,
        PlatformEnum.CUDA,
        [B12xNvFp4LinearKernel, B12xNvFp4W4A16LinearKernel, MarlinNvFp4LinearKernel],
    )
    monkeypatch.setattr(
        B12xNvFp4W4A16LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    return default_vllm_config


def test_b12x_w4a16_serves_phase_split(monkeypatch, _select_b12x_w4a16):
    """The oracle returns the b12x W4A16 kernel itself with
    quantization_config.linear.phase set; an lm_head copy keeps no phase."""
    phase = _b12x_phase(8)
    _use_phase(monkeypatch, phase)

    kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert type(kernel) is B12xNvFp4W4A16LinearKernel
    assert kernel.phase == phase
    assert (kernel.phase_mode(8), kernel.phase_mode(9)) == (None, kNvfp4DynamicTensor)
    assert kernel.input_quant_key() is None
    bare = init_nvfp4_linear_kernel(use_a16=True, allow_phase_split=False)
    assert type(bare) is B12xNvFp4W4A16LinearKernel
    assert bare.phase is None


def test_b12x_w4a4_rejects_phase(monkeypatch, _select_b12x_w4a16):
    _use_phase(monkeypatch, _b12x_phase(8))
    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    with pytest.raises(NotImplementedError, match="weight=nvfp4_w4a4"):
        init_nvfp4_linear_kernel(use_a16=False)


_NVFP4_CELLS = [
    ("unquantized", "nvfp4"),
    ("nvfp4", "nvfp4"),
    ("nvfp4", "unquantized"),
    ("unquantized", "unquantized"),
]
_MXFP8_CELLS = [
    ("unquantized", "mxfp8"),
    ("mxfp8", "mxfp8"),
    ("mxfp8", "unquantized"),
    ("unquantized", "unquantized"),
]


@pytest.mark.parametrize("decode,prefill", _NVFP4_CELLS)
def test_b12x_w4a16_serves_every_nvfp4_cell(
    monkeypatch, _select_b12x_w4a16, decode, prefill
):
    phase = _b12x_phase(8, decode=decode, prefill=prefill)
    _use_phase(monkeypatch, phase)
    kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert type(kernel) is B12xNvFp4W4A16LinearKernel
    assert (kernel.phase_mode(8), kernel.phase_mode(9)) == phase.activation_modes()


def test_b12x_w4a16_keeps_activations_for_mxfp8_phase(monkeypatch, _select_b12x_w4a16):
    """A phase in another quantized format leaves NVFP4 linears alone."""
    _use_phase(monkeypatch, _b12x_phase(8, prefill="mxfp8"))
    kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert type(kernel) is B12xNvFp4W4A16LinearKernel
    assert kernel.phase is None


@pytest.fixture
def _select_b12x_mxfp8(monkeypatch, default_vllm_config):
    import vllm.model_executor.kernels.linear as linear_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "b12x"
    )
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    return default_vllm_config


@pytest.mark.parametrize("decode,prefill", _MXFP8_CELLS)
def test_b12x_mxfp8_serves_phase_split(
    monkeypatch, _select_b12x_mxfp8, decode, prefill
):
    phase = _b12x_phase(8, decode=decode, prefill=prefill)
    _use_phase(monkeypatch, phase)
    kernel = init_mxfp8_linear_kernel(weight_shape=(2048, 2048))
    assert type(kernel) is B12xMxfp8LinearKernel
    assert kernel.phase == phase
    assert (kernel.phase_mode(8), kernel.phase_mode(9)) == phase.activation_modes()
    assert not kernel.supports_pre_processed_weights


def test_mxfp8_phase_rejects_other_kernels_and_formats(monkeypatch, _select_b12x_mxfp8):
    """An NVFP4 phase leaves MXFP8 linears alone; an MXFP8 phase on a kernel
    that cannot serve it raises and names the b12x kernel."""
    import vllm.model_executor.kernels.linear as linear_mod
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )

    _use_phase(monkeypatch, _b12x_phase(8))
    kernel = init_mxfp8_linear_kernel(weight_shape=(2048, 2048))
    assert type(kernel) is B12xMxfp8LinearKernel
    assert kernel.phase is None

    _use_phase(monkeypatch, _b12x_phase(8, prefill="mxfp8"))
    monkeypatch.setattr(
        linear_mod, "_get_linear_backend", lambda *, quantization: "emulation"
    )
    monkeypatch.setattr(
        EmulationMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    with pytest.raises(NotImplementedError) as error:
        init_mxfp8_linear_kernel(weight_shape=(2048, 2048))
    message = str(error.value)
    assert "weight=mxfp8" in message
    assert "EmulationMxfp8LinearKernel cannot serve it" in message
    assert "B12xMxfp8LinearKernel" in message.rsplit("b12x):", 1)[1]
    # Batched (BMM) layers keep their own activations.
    kernel = init_mxfp8_linear_kernel(weight_shape=(2048, 2048), bmm_batch_size=4)
    assert type(kernel) is EmulationMxfp8LinearKernel


@pytest.mark.parametrize(
    "setting,match",
    [
        ("capture_sizes", "largest CUDA graph capture size 8"),
        ("decoder_replay_sizes", "largest CUDA graph capture size 16"),
        ("data_parallel", "does not support data parallelism"),
        ("elastic_ep", "does not support data parallelism"),
        ("encoder_cudagraphs", "encoder CUDA graphs"),
    ],
)
def test_b12x_w4a16_phase_rejects_unsafe_rows(
    monkeypatch, _select_b12x_w4a16, setting, match
) -> None:
    """Padded rows are uninitialized and must not reach the shared scale."""
    config = _select_b12x_w4a16
    _use_phase(monkeypatch, _b12x_phase(4 if setting == "capture_sizes" else 8))
    config.parallel_config.data_parallel_size = 2 if setting == "data_parallel" else 1
    config.parallel_config.enable_elastic_ep = setting == "elastic_ep"
    config.compilation_config.cudagraph_mm_encoder = setting == "encoder_cudagraphs"
    config.compilation_config.decoder_replay_cudagraph_capture_sizes = (
        [16] if setting == "decoder_replay_sizes" else []
    )
    with pytest.raises(ValueError, match=match) as error:
        init_nvfp4_linear_kernel(use_a16=True)
    assert "quantization_config.linear.phase" in str(error.value)


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


def test_b12x_nvfp4_w4a16_uses_weight_only_scale_and_a16_dispatch(
    monkeypatch, default_vllm_config
):
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = []
    calls = []
    plans: list[types.SimpleNamespace] = []
    packed_weight = types.SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=48,
        global_scale_kind="multiplier",
    )

    def pack_weight(weight, scale, *, recipe, global_scale):
        assert weight is layer.weight
        assert scale is layer.weight_scale
        assert recipe == "nvfp4"
        assert global_scale is layer.weight_global_scale
        return packed_weight

    def plan_regimes(query, *, exact_m, a16_max_tokens):
        assert a16_max_tokens == 0
        plans.append(types.SimpleNamespace(handle=len(plans), token_counts=exact_m))
        return plans[-1]

    def mm(source, weight, *, plan, required_mode):
        assert weight is packed_weight
        assert plan is plans[-1]
        assert required_mode == "a16"
        assert source.dtype == torch.bfloat16
        assert source.is_contiguous()
        calls.append(source)
        return torch.full((source.shape[0], 48), 3.0, dtype=source.dtype)

    def reject_activation_quantization(*args, **kwargs):
        pytest.fail("W4A16 must not quantize activations")

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", reject_activation_quantization)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(
            pack_weight=pack_weight,
            mm=mm,
            plan_regimes=plan_regimes,
            BlockscaledQuery=types.SimpleNamespace,
        ),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_intrinsics",
        lambda: types.SimpleNamespace(swizzle_block_scale=lambda scale: scale),
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((48, 8), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = torch.tensor([0.125])
    kernel = object.__new__(B12xNvFp4W4A16LinearKernel)
    kernel.process_weights_after_loading(layer)
    assert layer.b12x_warmup_provider is kernel
    # A reload with unchanged shapes keeps the prepared plan.
    kernel.process_weights_after_loading(layer)
    assert len(plans) == 1
    kernel.get_b12x_warmup_unit(layer, (1, 8), torch.bfloat16).compile()
    assert [source.shape[0] for source in calls] == [1, 8]

    x = torch.randn((2, 3, 256), dtype=torch.bfloat16)[..., ::2]
    output = kernel.apply_weights(layer, x, torch.ones(48, dtype=torch.bfloat16))
    torch.testing.assert_close(calls[-1], x.reshape(6, 128))
    torch.testing.assert_close(output, torch.full((2, 3, 48), 4.0, dtype=x.dtype))
    with pytest.raises(ValueError, match="requires BF16 input"):
        kernel.apply_weights(layer, x.float())


def _fake_blockscaled(packed_weight, plans, calls):
    """A b12x blockscaled module that records plans and GEMM calls."""

    def plan_regimes(query, *, exact_m, a16_max_tokens=0):
        counts = {
            *exact_m,
            query.num_tokens,
            *([a16_max_tokens] * bool(a16_max_tokens)),
        }
        plans.append(
            types.SimpleNamespace(
                handle=len(plans),
                token_counts=tuple(sorted(counts)),
                exact_m=exact_m,
                query=query,
                a16_max_tokens=a16_max_tokens,
            )
        )
        return plans[-1]

    def mm(source, weight, *, plan, required_mode, activation_global_scale=None):
        assert weight is packed_weight
        # b12x resolves a call's regime from its row count and rejects a call
        # whose required mode differs, or that exceeds the plan.
        rows = source.shape[0]
        a16 = plan.query.activation_mode == "a16" or rows <= plan.a16_max_tokens
        assert required_mode == ("a16" if a16 else "quantized"), (rows, plan)
        assert rows <= plan.query.num_tokens, (rows, plan)
        calls.append(
            (source.shape[0], plan.handle, required_mode, activation_global_scale)
        )
        return torch.zeros((source.shape[0], 48), dtype=source.dtype)

    return types.SimpleNamespace(
        pack_weight=lambda *args, **kwargs: packed_weight,
        mm=mm,
        plan_regimes=plan_regimes,
        BlockscaledQuery=types.SimpleNamespace,
    )


# (decode, prefill) -> per plan (activation mode, rows, a16_max_tokens, exact
# M), then per call of 8 and 9 rows (plan, required mode, activation scale).
_NVFP4_PLANS: dict[tuple[str, str], tuple[list, list]] = {
    ("unquantized", "nvfp4"): (
        [("quantized", 64, 8, (1, 2, 4, 8))],
        [(0, "a16", "weight"), (0, "quantized", "amax")],
    ),
    ("nvfp4", "nvfp4"): (
        [("quantized", 64, 0, (1, 2, 4, 8))],
        [(0, "quantized", "amax"), (0, "quantized", "amax")],
    ),
    ("nvfp4", "unquantized"): (
        [("a16", 64, 0, ()), ("quantized", 8, 0, (1, 2, 4))],
        [(1, "quantized", "amax"), (0, "a16", "weight")],
    ),
    ("unquantized", "unquantized"): (
        [("a16", 64, 0, (1, 2, 4, 8))],
        [(0, "a16", None), (0, "a16", None)],
    ),
}


@pytest.mark.parametrize("decode,prefill", _NVFP4_CELLS)
def test_b12x_nvfp4_w4a16_plans_and_dispatches_every_cell(
    monkeypatch, default_vllm_config, decode, prefill
):
    """Each phase's calls reach a plan regime with its activations. Quantized
    calls take a per-call scale from their absolute maximum; quantized decode
    with A16 prefill gets its own plan, since b12x forces A16 only below a row
    count. Layers b12x cannot quantize keep BF16 activations."""
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    default_vllm_config.scheduler_config.max_num_batched_tokens = 64
    packed_weight = types.SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=48,
        global_scale_kind="multiplier",
        global_scale=torch.tensor([0.125]),
    )
    plans: list[types.SimpleNamespace] = []
    calls: list[tuple] = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: _fake_blockscaled(packed_weight, plans, calls),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_intrinsics",
        lambda: types.SimpleNamespace(swizzle_block_scale=lambda scale: scale),
    )
    layer = torch.nn.Module()
    layer.prefix = f"proj.{decode}.{prefill}"
    layer.output_size_per_partition = 48
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((48, 8), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = torch.tensor([0.125])
    phase = _b12x_phase(8, decode=decode, prefill=prefill)
    kernel = _with_phase(object.__new__(B12xNvFp4W4A16LinearKernel), phase)
    kernel.process_weights_after_loading(layer)
    expected_plans, expected_calls = _NVFP4_PLANS[(decode, prefill)]
    quantized = phase.quantized_format is not None
    assert [
        (
            plan.query.activation_mode,
            plan.query.num_tokens,
            plan.a16_max_tokens,
            plan.exact_m,
        )
        for plan in plans
    ] == expected_plans
    assert all(plan.query.activation_scale_available == quantized for plan in plans)
    # A reload with unchanged shapes keeps the prepared plans.
    kernel.process_weights_after_loading(layer)
    assert len(plans) == len(expected_plans)

    x = torch.randn((9, 128), dtype=torch.bfloat16)
    kernel.apply_weights(layer, x[:8])
    kernel.apply_weights(layer, x)
    scales = {
        "weight": packed_weight.global_scale,
        "amax": None,
        None: None,
    }
    for (rows, handle, mode, scale), (plan, expected_mode, kind) in zip(
        calls, expected_calls
    ):
        assert (handle, mode) == (plan, expected_mode)
        if kind == "amax":
            torch.testing.assert_close(
                scale, 6.0 * 448.0 / x[:rows].abs().max().float()
            )
        else:
            assert scale is scales[kind]
    with pytest.raises(ValueError, match="does not support activations"):
        kernel.apply_mode(layer, x, mode=kNvfp4Dynamic)

    # The warmup prepares every planned shape in its phase's activations.
    calls.clear()
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 64), torch.bfloat16
    ).compile()
    assert sorted({rows for rows, *_ in calls}) == [1, 2, 4, 8, 64]
    assert all(
        mode == expected_calls[0 if rows <= 8 else 1][1] for rows, _, mode, _ in calls
    )

    packed_weight.padded_in_features = 136
    plans.clear()
    calls.clear()
    kernel.process_weights_after_loading(layer)
    assert [plan.query.activation_mode for plan in plans] == ["a16"]
    kernel.apply_weights(layer, x)
    assert calls == [(9, 0, "a16", None)]


@pytest.mark.parametrize("max_decode_tokens", [64, 100])
def test_b12x_nvfp4_quantized_decode_cutoff_at_capacity(
    monkeypatch, default_vllm_config, max_decode_tokens
):
    """Quantized decode with BF16 prefill and a cutoff at or above the planned
    capacity keeps an A16 plan for prefill calls, which run in capacity-sized
    chunks."""
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    default_vllm_config.scheduler_config.max_num_batched_tokens = 64
    packed_weight = types.SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=48,
        global_scale_kind="multiplier",
        global_scale=torch.tensor([0.125]),
    )
    plans: list[types.SimpleNamespace] = []
    calls: list[tuple] = []
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: _fake_blockscaled(packed_weight, plans, calls),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_intrinsics",
        lambda: types.SimpleNamespace(swizzle_block_scale=lambda scale: scale),
    )
    layer = torch.nn.Module()
    layer.prefix = f"proj.cutoff{max_decode_tokens}"
    layer.output_size_per_partition = 48
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((48, 8), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_global_scale = torch.tensor([0.125])
    phase = _b12x_phase(max_decode_tokens, decode="nvfp4", prefill="unquantized")
    kernel = _with_phase(object.__new__(B12xNvFp4W4A16LinearKernel), phase)
    kernel.process_weights_after_loading(layer)
    assert [(p.query.activation_mode, p.query.num_tokens) for p in plans] == [
        ("a16", 64),
        ("quantized", 64),
    ]
    kernel.apply_weights(layer, torch.randn((8, 128), dtype=torch.bfloat16))
    kernel.apply_weights(
        layer, torch.randn((max_decode_tokens + 6, 128), dtype=torch.bfloat16)
    )
    assert [(rows, handle, mode) for rows, handle, mode, _ in calls] == [
        (8, 1, "quantized"),
        (64, 0, "a16"),
        (max_decode_tokens - 58, 0, "a16"),
    ]

    # The warmup prepares the A16 prefill plan as well, though every planned
    # row count is a decode call.
    calls.clear()
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 64), torch.bfloat16
    ).compile()
    assert {(handle, mode) for _, handle, mode, _ in calls} == {
        (0, "a16"),
        (1, "quantized"),
    }


@torch.inference_mode()
def test_b12x_nvfp4_scale_skips_padding_rows(monkeypatch, default_vllm_config):
    """CUDA graph padding rows may hold uninitialized memory, so the shared
    NVFP4 scale comes from the rows the runner marks as real. Without a mask
    every row counts, except in a CUDA graph, where that raises."""
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod
    from vllm.forward_context import set_forward_context

    x = torch.randn((4, 128), dtype=torch.bfloat16)
    x[2] = float("nan")
    x[3] = 1e4
    real = 6.0 * 448.0 / x[:2].abs().max().float()
    is_padding = torch.tensor([False, False, True, True])
    with set_forward_context(None, default_vllm_config, is_padding=is_padding):
        torch.testing.assert_close(b12x_mod._activation_global_scale(x), real)
        # All-padding dummy runs keep a finite scale.
        is_padding.fill_(True)
        assert torch.isfinite(b12x_mod._activation_global_scale(x))
    x[2] = 0
    torch.testing.assert_close(
        b12x_mod._activation_global_scale(x), 6.0 * 448.0 / x.abs().max().float()
    )
    assert b12x_mod._activation_global_scale(x[:0]) == 1
    monkeypatch.setattr(type(x), "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="padding mask"):
        b12x_mod._activation_global_scale(x)


# (decode, prefill) -> per plan (activation mode, rows, a16_max_tokens, exact
# M), then per call of 8 and 9 rows (plan, required mode).
_MXFP8_PLANS: dict[tuple[str, str], tuple[list, list]] = {
    ("unquantized", "mxfp8"): (
        [("quantized", 64, 8, (1, 2, 4, 8))],
        [(0, "a16"), (0, "quantized")],
    ),
    ("mxfp8", "mxfp8"): (
        [("quantized", 64, 0, (1, 2, 4, 8))],
        [(0, "quantized"), (0, "quantized")],
    ),
    ("mxfp8", "unquantized"): (
        [("a16", 64, 0, ()), ("quantized", 8, 0, (1, 2, 4))],
        [(1, "quantized"), (0, "a16")],
    ),
    ("unquantized", "unquantized"): (
        [("a16", 64, 0, (1, 2, 4, 8))],
        [(0, "a16"), (0, "a16")],
    ),
}


@pytest.mark.parametrize("decode,prefill", _MXFP8_CELLS)
def test_b12x_mxfp8_plans_and_dispatches_every_cell(
    monkeypatch, default_vllm_config, decode, prefill
):
    """With a phase split, b12x MXFP8 plans every serving shape with each
    phase's activations instead of choosing them by row count. MXFP8 scales
    are per row block, so no activation scale is passed."""
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    default_vllm_config.scheduler_config.max_num_batched_tokens = 64
    packed = types.SimpleNamespace(
        in_features=128,
        padded_in_features=128,
        out_features=48,
        weight=types.SimpleNamespace(values=torch.empty(0)),
    )
    plans: list[types.SimpleNamespace] = []
    calls: list[tuple] = []
    blockscaled = _fake_blockscaled(packed, plans, calls)
    monkeypatch.setattr(b12x_mod, "_import_b12x_blockscaled", lambda: blockscaled)
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(
            pack_weight=lambda *args: packed, mm=blockscaled.mm
        ),
    )
    layer = torch.nn.Module()
    layer.prefix = f"mxfp8.{decode}.{prefill}"
    layer.output_size_per_partition = 48
    layer.weight = torch.nn.Parameter(
        torch.empty((48, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((48, 4), dtype=torch.uint8), requires_grad=False
    )
    phase = _b12x_phase(8, decode=decode, prefill=prefill)
    kernel = _with_phase(object.__new__(B12xMxfp8LinearKernel), phase)
    kernel.process_weights_after_loading(layer)
    expected_plans, expected_calls = _MXFP8_PLANS[(decode, prefill)]
    assert [
        (
            plan.query.activation_mode,
            plan.query.num_tokens,
            plan.a16_max_tokens,
            plan.exact_m,
        )
        for plan in plans
    ] == expected_plans
    assert not any(plan.query.activation_scale_available for plan in plans)

    x = torch.randn((9, 128), dtype=torch.bfloat16)
    bias = torch.ones(48, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x[:8], bias)
    torch.testing.assert_close(out, torch.ones((8, 48), dtype=torch.bfloat16))
    kernel.apply_weights(layer, x)
    assert [(handle, mode) for _, handle, mode, _ in calls] == expected_calls
    assert all(scale is None for *_, scale in calls)
    with pytest.raises(ValueError, match="does not support activations"):
        kernel.apply_mode(layer, x, mode=kNvfp4DynamicTensor)
    # b12x picks the activation mode per call for BF16 sources only.
    with pytest.raises(ValueError, match="requires BF16 activations"):
        kernel.apply_weights(layer, x.half())

    calls.clear()
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 64), torch.bfloat16
    ).compile()
    assert sorted({rows for rows, *_ in calls}) == [1, 2, 4, 8, 64]
    assert all(
        mode == expected_calls[0 if rows <= 8 else 1][1] for rows, _, mode, _ in calls
    )


def _nvfp4_w4a16_layer(n, k):
    """A random NVFP4 W4A16 layer and its decoded fp32 weight."""
    torch.manual_seed(17)
    codes = torch.randint(0, 16, (n, k), device="cuda")
    scales = (2.0 ** torch.randint(-2, 2, (n, k // 16), device="cuda")).to(
        torch.float8_e4m3fn
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        (codes[:, ::2] | codes[:, 1::2] << 4).to(torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor([0.125], device="cuda"), requires_grad=False
    )
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    decoded = lut[codes] * scales.float().repeat_interleave(16, dim=1) * 0.125
    layer.prefix = f"proj_{n}x{k}"
    layer.output_size_per_partition = n
    return layer, decoded


@pytest.mark.parametrize("cell", [None, *_NVFP4_CELLS])
@torch.inference_mode()
def test_b12x_nvfp4_w4a16_does_not_prepare_after_warmup(
    monkeypatch, default_vllm_config, cell
) -> None:
    """Every serving token count is planned at load time and prepared by the
    warmup, before CUDA graph capture.

    A b12x preparation evicts programs no prepared plan retains, including
    routed-MoE kernels compiled earlier in the same forward, which then fails
    CUDA graph capture.
    """
    supported, reason = B12xNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    import b12x.preparation.session as session

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    k = 256
    layer, decoded = _nvfp4_w4a16_layer(256, k)
    kernel = B12xNvFp4W4A16LinearKernel(None)
    mode = None
    if cell is not None:
        kernel.phase = _b12x_phase(8, decode=cell[0], prefill=cell[1])
        mode = kNvfp4DynamicTensor
    kernel.process_weights_after_loading(layer)
    capacity = layer.b12x_nvfp4_plans.capacity
    kernel.get_b12x_warmup_unit(layer, (1, capacity), torch.bfloat16).compile()

    prepared = []
    prepare_default = session.prepare_default

    def record_prepare(request):
        prepared.append(request)
        return prepare_default(request)

    def quantized(rows: int) -> bool:
        return cell is not None and cell[0 if rows <= 8 else 1] == "nvfp4"

    monkeypatch.setattr(session, "prepare_default", record_prepare)
    for m in (8, 4, 2, 1, 3, 40, capacity + 5):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        expected = torch.cat(
            [
                _nvfp4_activation_reference(chunk, decoded)
                if quantized(chunk.shape[0])
                else (chunk.double() @ decoded.double().T).bfloat16()
                for chunk in x.split(capacity)
            ]
        )
        actual = kernel.apply_mode(layer, x, mode=mode)
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.01)
    assert not prepared


def _nvfp4_activation_reference(x, decoded):
    """Reference output for NVFP4 activations scaled by amax(x), times ``decoded``."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant,
    )

    global_scale = 6.0 * 448.0 / x.abs().max().float()
    codes, block_scale = ref_nvfp4_quant(x, global_scale, 16)
    dequant = (
        codes.view(*block_scale.shape, 16) * (block_scale / global_scale)[..., None]
    )
    return (dequant.view_as(x).double() @ decoded.double().T).bfloat16()


@pytest.mark.parametrize("magnitude", [1.0, 64.0])
@torch.inference_mode()
def test_b12x_nvfp4_w4a16_quantizes_activations_above_cutoff(
    default_vllm_config, magnitude
) -> None:
    """Passes above ``max_decode_tokens`` rows use NVFP4 activations and smaller
    passes keep BF16 ones, decided at execution time under torch.compile: a
    branch in traced code would be fixed for every size."""
    supported, reason = B12xNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    from torch._dynamo.testing import CompileCounterWithBackend

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    k = 256
    layer, decoded = _nvfp4_w4a16_layer(256, k)
    reference, _ = _nvfp4_w4a16_layer(256, k)
    kernel = _with_phase(B12xNvFp4W4A16LinearKernel(None), _b12x_phase(8))
    kernel.process_weights_after_loading(layer)
    a16_kernel = B12xNvFp4W4A16LinearKernel(None)
    a16_kernel.process_weights_after_loading(reference)
    capacity = layer.b12x_nvfp4_plans.capacity
    source = torch.zeros(capacity, k, device="cuda", dtype=torch.bfloat16)
    kernel.apply_mode(layer, source, mode=kNvfp4DynamicTensor)
    a16_kernel.apply_weights(reference, source)

    counter = CompileCounterWithBackend("inductor")
    run = torch.compile(
        lambda x: kernel.apply_weights(layer, x),
        backend=counter,
        fullgraph=True,
        dynamic=True,
    )
    for m in (4, 40, 8, 9):
        x = magnitude * torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        actual = run(x)
        a16 = a16_kernel.apply_weights(reference, x)
        if m <= 8:
            torch.testing.assert_close(actual, a16, atol=0, rtol=0)
        else:
            expected = _nvfp4_activation_reference(x, decoded)
            torch.testing.assert_close(
                actual, expected, atol=0.005 * magnitude, rtol=0.01
            )
            assert not torch.allclose(actual, a16, atol=0.005 * magnitude, rtol=0.01)
    assert counter.frame_count == 1
    zeros = torch.zeros(40, k, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(run(zeros), torch.zeros(40, 256, device="cuda").bfloat16())

    # SchemaCheckMode compares inputs with allclose, which CUDA does not
    # implement for float8. The numerical tests above guard input mutation.
    torch.library.opcheck(
        torch.ops.vllm.phase_split_linear.default,
        (
            torch.randn(40, k, device="cuda", dtype=torch.bfloat16),
            None,
            _encode_layer_name(layer.phase_split_layer_name),
            256,
        ),
        test_utils="test_faketensor",
    )


@pytest.mark.parametrize("max_decode_tokens", [None, 32])
@pytest.mark.parametrize("m,n,k", [(2, 136, 96), (17, 256, 256)])
@torch.inference_mode()
def test_b12x_nvfp4_w4a16_preserves_bf16_activations_under_graph_replay(
    monkeypatch, default_vllm_config, m, n, k, max_decode_tokens
) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    supported, reason = B12xNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 32]

    def reject_activation_quantization(*args, **kwargs):
        pytest.fail("W4A16 must not quantize activations")

    monkeypatch.setattr(b12x_mod, "scaled_fp4_quant", reject_activation_quantization)
    layer, decoded = _nvfp4_w4a16_layer(n, k)
    x_storage = torch.randn(1, m, k * 2, device="cuda", dtype=torch.bfloat16)
    x = x_storage[..., ::2]
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    kernel = B12xNvFp4W4A16LinearKernel(None)
    if max_decode_tokens is not None:
        # Capture sizes up to max_decode_tokens replay the decode phase.
        kernel = _with_phase(kernel, _b12x_phase(max_decode_tokens))
    kernel.process_weights_after_loading(layer)
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 2, 4, 32), x.dtype
    ).compile()
    run = torch.compile(
        lambda inputs: kernel.apply_weights(layer, inputs, bias),
        fullgraph=True,
        dynamic=True,
    )
    run(x)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(x)
    for _ in range(2):
        x_storage.normal_()
        graph.replay()
        expected = (x.double() @ decoded.double().T).bfloat16() + bias
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.01)


def _mxfp8_layer(n, k):
    """A random MXFP8 layer and its decoded fp32 weight."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
        dequant_mxfp8_to_bf16,
    )

    torch.manual_seed(17)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 16
    values, scales = _mxfp8_e4m3_quantize_torch(weight)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(values, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.prefix = f"mxfp8_proj_{n}x{k}"
    layer.output_size_per_partition = n
    return layer, dequant_mxfp8_to_bf16(values, scales).float()


def _mxfp8_activation_reference(x, decoded):
    """Reference output for MXFP8 activations, times ``decoded``."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
        dequant_mxfp8_to_bf16,
    )

    values, scales = _mxfp8_e4m3_quantize_torch(x)
    return (
        dequant_mxfp8_to_bf16(values, scales).double() @ decoded.double().T
    ).bfloat16()


def _phase_split_layer(weight, decode, prefill):
    """A b12x phase-split layer, its decoded weight, and the reference for its
    quantized activations; skips without b12x on this device."""
    kernel_cls = (
        B12xNvFp4W4A16LinearKernel if weight == "nvfp4" else B12xMxfp8LinearKernel
    )
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)
    if weight == "nvfp4":
        layer, decoded = _nvfp4_w4a16_layer(256, 256)
        kernel = B12xNvFp4W4A16LinearKernel(None)
        reference = _nvfp4_activation_reference
    else:
        layer, decoded = _mxfp8_layer(256, 256)
        kernel = B12xMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(256, 256)))
        reference = _mxfp8_activation_reference
    _with_phase(kernel, _b12x_phase(8, decode=decode, prefill=prefill))
    kernel.process_weights_after_loading(layer)
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 2, 4, 8), torch.bfloat16
    ).compile()
    return kernel, layer, decoded, reference


def _assert_phase_output(actual, x, decoded, reference, quantized):
    """A16 calls match the BF16 product; quantized ones are much closer to the
    quantized-activation reference than to it."""
    a16 = (x.double() @ decoded.double().T).bfloat16()
    if not quantized:
        torch.testing.assert_close(actual, a16, atol=0.005, rtol=0.01)
        return
    assert torch.isfinite(actual).all()
    error = (actual.float() - reference(x, decoded).float()).abs().max()
    a16_error = (actual.float() - a16.float()).abs().max()
    assert error < 0.25 * a16_error, (error, a16_error)


_GPU_CELLS = [("nvfp4", *cell) for cell in _NVFP4_CELLS] + [
    ("mxfp8", *cell) for cell in _MXFP8_CELLS
]


@pytest.mark.parametrize("weight,decode,prefill", _GPU_CELLS)
@torch.inference_mode()
def test_b12x_phase_split_matches_reference(
    monkeypatch, default_vllm_config, weight, decode, prefill
) -> None:
    """Each served (decode, prefill) cell computes calls of up to
    max_decode_tokens rows with the decode activations and larger ones with
    the prefill activations, without preparing anything after the warmup."""
    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    kernel, layer, decoded, reference = _phase_split_layer(weight, decode, prefill)
    import b12x.preparation.session as session

    prepared = []
    prepare_default = session.prepare_default

    def record_prepare(request):
        prepared.append(request)
        return prepare_default(request)

    monkeypatch.setattr(session, "prepare_default", record_prepare)
    for m in (1, 8, 9, 40):
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
        phase_format = decode if m <= 8 else prefill
        _assert_phase_output(
            kernel.apply_weights(layer, x),
            x,
            decoded,
            reference,
            quantized=phase_format != "unquantized",
        )
    assert not prepared


@pytest.mark.parametrize("weight", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("prefill", ["quantized", "unquantized"])
@torch.inference_mode()
def test_b12x_quantized_decode_ignores_graph_padding_rows(
    default_vllm_config, weight, prefill
) -> None:
    """Quantized decode calls replay in CUDA graphs whose padding rows may hold
    NaN. NVFP4 takes its shared scale from the rows the runner's padding mask
    marks as real; MXFP8 scales are per row block."""
    from vllm.forward_context import set_forward_context

    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
    kernel, layer, decoded, reference = _phase_split_layer(
        weight, weight, weight if prefill == "quantized" else "unquantized"
    )
    x = torch.zeros(8, 256, device="cuda", dtype=torch.bfloat16)
    is_padding = torch.ones(8, dtype=torch.bool, device="cuda")
    run = torch.compile(
        lambda inputs: kernel.apply_weights(layer, inputs), fullgraph=True, dynamic=True
    )
    with set_forward_context(None, default_vllm_config, is_padding=is_padding):
        run(x)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(x)
    for real in (3, 8):
        x.normal_()
        x[real:] = float("nan")
        is_padding.copy_(torch.arange(8, device="cuda") >= real)
        graph.replay()
        _assert_phase_output(
            actual[:real], x[:real], decoded, reference, quantized=True
        )


def _block_weights(n, k, codec):
    block_size, block_bytes = (
        (32, 34) if codec == "q8_0" else (256, 66 if codec == "iq2_xxs" else 74)
    )
    raw = torch.randint(0, 256, (n, k // block_size, block_bytes), dtype=torch.uint8)
    raw[..., :2] = torch.full(
        (n, k // block_size, 1), 1 / 128, dtype=torch.float16
    ).view(torch.uint8)
    return raw


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
@pytest.mark.parametrize("profile_first", [False, True])
@torch.inference_mode()
def test_b12x_block_quant_dense_compiled_graph(
    default_vllm_config, codec, profile_first
):
    supported, reason = B12xNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    from b12x.testing.iq2_xs_reference import dequantize_blocks

    from vllm.model_executor.layers.quantization.modelopt_block_quant import (
        ModelOptBlockQuantLinearMethod,
    )

    default_vllm_config.scheduler_config.max_num_batched_tokens = 32
    default_vllm_config.scheduler_config.max_num_scheduled_tokens = 32
    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 4]
    method = ModelOptBlockQuantLinearMethod(codec)
    raw = _block_weights(128, 256, codec)
    decoded = dequantize_blocks(raw).bfloat16().cuda()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(raw.cuda(), requires_grad=False)
    method.process_weights_after_loading(layer)
    assert layer.b12x_block_plan.prepared is None
    if profile_first:
        method.apply(layer, torch.zeros(32, 256, device="cuda", dtype=torch.bfloat16))
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 4, 32), torch.bfloat16
    ).compile()
    assert layer.b12x_block_plan.prepared is not None
    run = torch.compile(lambda x: method.apply(layer, x), fullgraph=True, dynamic=True)
    for rows in (1, 4, 17):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        run(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(x)
        for _ in range(2):
            x.normal_()
            allocated = torch.accelerator.memory_allocated()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() == allocated
            expected = (x.float() @ decoded.float().T).bfloat16()
            torch.testing.assert_close(output, expected, rtol=0.015, atol=0.03125)
        graph.reset()


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("profile_first", [False, True])
@torch.inference_mode()
def test_b12x_q8_embedding_graph(default_vllm_config, id_dtype, profile_first):
    supported, reason = B12xNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
    from b12x.testing.q8_0_reference import dequantize_blocks

    from vllm.model_executor.layers.quantization.modelopt_block_quant import (
        ModelOptBlockQuantLinearMethod,
    )

    default_vllm_config.scheduler_config.max_num_batched_tokens = 32
    default_vllm_config.scheduler_config.max_num_scheduled_tokens = 32
    default_vllm_config.compilation_config.cudagraph_capture_sizes = [1, 4]
    method = ModelOptBlockQuantLinearMethod("q8_0")
    method.is_embedding = True
    raw = _block_weights(128, 256, "q8_0")
    decoded = dequantize_blocks(raw).bfloat16().cuda()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(raw.cuda(), requires_grad=False)
    method.process_weights_after_loading(layer)
    assert all(plan.prepared is None for plan in layer.b12x_embedding_plans.values())
    if profile_first:
        method.embedding(layer, torch.zeros(32, device="cuda", dtype=id_dtype))
    layer.b12x_warmup_provider.get_b12x_warmup_unit(
        layer, (1, 4, 32), torch.bfloat16
    ).compile()
    assert all(
        plan.prepared is not None for plan in layer.b12x_embedding_plans.values()
    )
    run = torch.compile(
        lambda x: method.embedding(layer, x), fullgraph=True, dynamic=True
    )
    for count in (7, 9, 70):
        ids = torch.arange(count, device="cuda", dtype=id_dtype)
        torch._dynamo.mark_dynamic(ids, 0)
        torch.testing.assert_close(run(ids), decoded[ids.long()], rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(ids)
    ids.add_(20)
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
    torch.testing.assert_close(output, decoded[ids.long()], rtol=0, atol=0)
    graph.reset()
