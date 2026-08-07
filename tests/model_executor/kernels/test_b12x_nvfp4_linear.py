# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types

import torch

from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_NVFP4_KERNELS,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    B12xNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    MarlinNvFp4LinearKernel,
)
from vllm.platforms import PlatformEnum


def test_b12x_backend_maps_nvfp4_kernel() -> None:
    assert B12xNvFp4LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    assert B12xNvFp4LinearKernel in _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]


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


def test_b12x_fp4_env_selects_nvfp4_with_auto_backend(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(linear_mod.envs, "VLLM_USE_B12X_FP4_GEMM", True)
    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel()

    assert isinstance(kernel, B12xNvFp4LinearKernel)


def test_b12x_nvfp4_env_enables_auto_backend(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod, "_current_linear_backend", lambda: "auto")
    monkeypatch.setattr(b12x_mod.envs, "VLLM_USE_B12X_FP4_GEMM", False)

    can_implement, reason = B12xNvFp4LinearKernel.can_implement(None)

    assert not can_implement
    assert reason == "B12X NVFP4 GEMM is not enabled"

    monkeypatch.setattr(b12x_mod.envs, "VLLM_USE_B12X_FP4_GEMM", True)
    can_implement, reason = B12xNvFp4LinearKernel.can_implement(None)

    assert can_implement
    assert reason is None


def test_b12x_fp4_env_preserves_w4a16_auto_fallback(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(linear_mod.envs, "VLLM_USE_B12X_FP4_GEMM", True)
    monkeypatch.setattr(
        MarlinNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )

    kernel = init_nvfp4_linear_kernel(use_a16=True)

    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_b12x_nvfp4_processes_scale_and_registers_layer(monkeypatch) -> None:
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
    vllm_config = VllmConfig()
    monkeypatch.setattr(
        b12x_mod, "get_current_vllm_config_or_none", lambda: vllm_config
    )
    kernel = object.__new__(B12xNvFp4LinearKernel)

    kernel.process_weights_after_loading(layer)

    assert layer.weight_scale.data_ptr() == swizzled_scale.data_ptr()
    assert layer.weight_scale.weight_loader is weight_loader
    assert vllm_config.compilation_config.static_forward_context[layer.prefix] is layer


def test_b12x_nvfp4_apply_calls_native_blockscaled_gemm(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.nvfp4.b12x as b12x_mod

    calls: list[tuple] = []
    x_packed = torch.empty((6, 64), dtype=torch.uint8)
    x_scale_storage = torch.empty((128, 8), dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((32, 4, 1, 4, 2, 1), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((32, 4, 1, 4, 2, 1), dtype=torch.float8_e4m3fn)

    def as_grouped_scale_view(storage, rows: int, cols: int):
        return x_scale if rows == 6 else weight_scale

    def mm(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        return torch.full((6, 48, 1), 3.0, dtype=torch.bfloat16)

    monkeypatch.setattr(
        b12x_mod,
        "scaled_fp4_quant",
        lambda *args, **kwargs: (x_packed, x_scale_storage),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_blockscaled",
        lambda: types.SimpleNamespace(mm=mm),
    )
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_intrinsics",
        lambda: types.SimpleNamespace(as_grouped_scale_view=as_grouped_scale_view),
    )
    monkeypatch.setattr(
        b12x_mod,
        "current_stream",
        lambda: types.SimpleNamespace(cuda_stream=123),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = 48
    layer.weight = torch.empty((48, 64), dtype=torch.uint8)
    layer.weight_scale = torch.empty((128, 8), dtype=torch.float8_e4m3fn)
    layer.input_global_scale_inv = torch.tensor(2.0)
    layer.alpha = torch.tensor(0.25)
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.ones(48, dtype=torch.bfloat16)
    kernel = object.__new__(B12xNvFp4LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    assert len(calls) == 1
    lhs, rhs, kwargs = calls[0]
    assert lhs[0].data_ptr() == x_packed.data_ptr()
    assert lhs[0].shape == (6, 64, 1)
    assert lhs[1] is x_scale
    assert rhs[0].data_ptr() == layer.weight.data_ptr()
    assert rhs[0].shape == (48, 64, 1)
    assert rhs[1] is weight_scale
    assert kwargs["ab_dtype"] == "float4_e2m1fn"
    assert kwargs["sf_dtype"] == "float8_e4m3fn"
    assert kwargs["sf_vec_size"] == 16
    assert kwargs["expected_m"] == 6
    assert kwargs["stream"] == 123
