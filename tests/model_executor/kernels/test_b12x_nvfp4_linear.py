# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types

import torch

from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_NVFP4_KERNELS,
    _resolve_backend_kernels,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    B12xNvFp4LinearKernel,
    warmup_b12x_nvfp4_linear,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.platforms import PlatformEnum


def test_b12x_backend_maps_nvfp4_kernel() -> None:
    assert B12xNvFp4LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    assert B12xNvFp4LinearKernel in _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]


def test_b12x_nvfp4_fallback_priority() -> None:
    kernels = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
    names = [kernel.__name__ for kernel in kernels]

    assert (
        names.index("FbgemmNvFp4LinearKernel")
        < names.index("B12xNvFp4LinearKernel")
        < names.index("EmulationNvFp4LinearKernel")
    )


def test_b12x_nvfp4_explicit_backend_selects_native_kernel(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "can_implement",
        classmethod(lambda cls, config: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel()

    assert isinstance(kernel, B12xNvFp4LinearKernel)


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


def test_backend_without_w4a16_kernel_preserves_fallback(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    kernels = [MarlinNvFp4LinearKernel]
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    assert _resolve_backend_kernels(kernels, "NVFP4") == kernels


def test_b12x_nvfp4_processes_scale_and_preserves_loader(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    scale = torch.empty((48, 8), dtype=torch.float8_e4m3fn)
    swizzled_scale = torch.empty((128, 8), dtype=torch.float8_e4m3fn)
    intrinsics = types.SimpleNamespace(swizzle_block_scale=lambda value: swizzled_scale)
    monkeypatch.setattr(b12x_mod, "_import_b12x_intrinsics", lambda: intrinsics)

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.shared_expert.down_proj"
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    weight_loader = object()
    layer.weight_scale.weight_loader = weight_loader
    kernel = object.__new__(B12xNvFp4LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale.data_ptr() == swizzled_scale.data_ptr()
    assert layer.weight_scale.weight_loader is weight_loader
    assert layer.b12x_nvfp4_linear


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

    monkeypatch.setattr(
        b12x_mod,
        "scaled_fp4_quant",
        quant,
    )
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


def test_warmup_b12x_nvfp4_dedupes_weight_signatures(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    calls = []

    def apply(
        source,
        weight,
        weight_scale,
        input_global_scale_inv,
        alpha,
        bias,
    ):
        calls.append(
            (
                source.shape,
                weight,
                weight_scale,
                input_global_scale_inv,
                alpha,
                bias,
            )
        )
        return source.new_empty((source.shape[0], weight.shape[0]))

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
    monkeypatch.setattr(b12x_mod, "_apply_b12x_nvfp4_linear", apply)

    def layer(n: int):
        return types.SimpleNamespace(
            b12x_nvfp4_linear=True,
            weight=torch.empty((n, 64), dtype=torch.uint8),
            weight_scale=torch.empty((128, 8), dtype=torch.float8_e4m3fn),
            input_global_scale_inv=torch.tensor(2.0),
            alpha=torch.tensor(0.25),
        )

    layer_a = layer(48)
    layer_b = layer(48)
    layer_c = layer(96)
    model = types.SimpleNamespace(
        modules=lambda: iter([layer_a, layer_b, layer_c, types.SimpleNamespace()])
    )

    warmed = warmup_b12x_nvfp4_linear(
        model,
        max_tokens=8,
        cudagraph_capture_sizes=[1, 2],
    )

    assert warmed == 6
    assert [call[0] for call in calls] == [
        torch.Size([1, 128]),
        torch.Size([2, 128]),
        torch.Size([8, 128]),
        torch.Size([1, 128]),
        torch.Size([2, 128]),
        torch.Size([8, 128]),
    ]
    assert calls[0][1] is layer_a.weight
    assert calls[3][1] is layer_c.weight
    assert all(call[5] is None for call in calls)
