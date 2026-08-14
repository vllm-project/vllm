# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from collections.abc import Generator
from unittest.mock import patch

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_MXFP8_KERNELS,
    FlashInferTrtllmMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    MXFP8_TRTLLM_LAYOUT_ENV,
    MXFP8_TRTLLM_SWITCH_M_ENV,
    _mxfp8_trtllm_layout_config,
    _mxfp8_trtllm_linear_fixed_impl,
    mxfp8_trtllm_linear,
    mxfp8_trtllm_use_8x4_sf_layout,
)
from vllm.platforms import PlatformEnum

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _kernel() -> FlashInferTrtllmMxfp8LinearKernel:
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    kernel.config = Mxfp8LinearLayerConfig()
    return kernel


@pytest.fixture(autouse=True)
def clear_mxfp8_trtllm_layout_config() -> Generator[None, None, None]:
    _mxfp8_trtllm_layout_config.cache_clear()
    yield
    _mxfp8_trtllm_layout_config.cache_clear()


@pytest.mark.parametrize(
    ("layout", "m", "expected"),
    [
        ("8x4", 1, True),
        ("8x4", 8192, True),
        ("128x4", 1, False),
        ("128x4", 8192, False),
        ("adaptive", 256, True),
        ("adaptive", 257, False),
    ],
)
def test_mxfp8_trtllm_layout_policy(
    monkeypatch, layout: str, m: int, expected: bool
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, layout)

    assert mxfp8_trtllm_use_8x4_sf_layout(m) is expected


def test_mxfp8_trtllm_adaptive_switch_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "32")

    assert mxfp8_trtllm_use_8x4_sf_layout(32)
    assert not mxfp8_trtllm_use_8x4_sf_layout(33)


def test_mxfp8_trtllm_layout_policy_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "bad-layout")

    with pytest.raises(ValueError, match="Valid options"):
        mxfp8_trtllm_use_8x4_sf_layout(1)


def test_mxfp8_trtllm_environment_variables_are_registered() -> None:
    assert MXFP8_TRTLLM_LAYOUT_ENV in envs.environment_variables
    assert MXFP8_TRTLLM_SWITCH_M_ENV in envs.environment_variables


@pytest.mark.parametrize("switch_m", ["0", "-1"])
def test_mxfp8_trtllm_layout_policy_rejects_nonpositive_switch(
    monkeypatch, switch_m: str
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, switch_m)

    with pytest.raises(ValueError, match="must be positive"):
        mxfp8_trtllm_use_8x4_sf_layout(1)


def test_mxfp8_trtllm_layout_policy_rejects_noninteger_switch(monkeypatch) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "not-an-integer")

    with pytest.raises(ValueError, match="invalid literal"):
        mxfp8_trtllm_use_8x4_sf_layout(1)


@pytest.mark.parametrize("use_8x4", [True, False])
def test_mxfp8_trtllm_fixed_impl_uses_matching_scale_layout(
    monkeypatch, use_8x4: bool
) -> None:
    calls: dict[str, object] = {}

    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["quantizer"] = "8x4" if use_8x4 else "128x4"
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((128,), dtype=torch.uint8),
        )

    def wrong_quantizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("selected the wrong MXFP8 scale layout")

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.vllm_flashinfer.flashinfer_mxfp8_quantize_8x4",
        quantize if use_8x4 else wrong_quantizer,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.vllm_flashinfer.flashinfer_mxfp8_quantize_128x4",
        wrong_quantizer if use_8x4 else quantize,
        raising=False,
    )

    def mm_mxfp8(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype,
        backend: str,
        use_8x4_sf_layout: bool,
    ) -> torch.Tensor:
        calls["mm_layout"] = use_8x4_sf_layout
        return torch.ones((a.shape[0], b.shape[1]), dtype=out_dtype)

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.vllm_flashinfer.mm_mxfp8",
        mm_mxfp8,
        raising=False,
    )

    output = _mxfp8_trtllm_linear_fixed_impl(
        torch.zeros((3, 512), dtype=torch.bfloat16),
        torch.zeros((256, 512), dtype=torch.float8_e4m3fn),
        torch.zeros((4096,), dtype=torch.uint8),
        130,
        use_8x4_sf_layout=use_8x4,
    )

    assert calls == {
        "quantizer": "8x4" if use_8x4 else "128x4",
        "mm_layout": use_8x4,
    }
    assert output.shape == (3, 130)
    assert output.is_contiguous()


def test_mxfp8_trtllm_adaptive_op_uses_runtime_shape(monkeypatch) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "2")
    calls: list[bool] = []

    def fixed_impl(*args, use_8x4_sf_layout: bool, **kwargs) -> torch.Tensor:
        calls.append(use_8x4_sf_layout)
        return torch.empty((args[0].shape[0], args[3]), dtype=args[0].dtype)

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer._mxfp8_trtllm_linear_fixed_impl",
        fixed_impl,
    )

    weight = torch.empty((256, 512), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((4096,), dtype=torch.uint8)
    mxfp8_trtllm_linear(
        torch.empty((2, 512), dtype=torch.bfloat16), weight, weight_scale, 130
    )
    mxfp8_trtllm_linear(
        torch.empty((3, 512), dtype=torch.bfloat16), weight, weight_scale, 130
    )

    assert calls == [True, False]


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [(100, True), (103, True), (107, True), (101, False), (120, False)],
)
def test_mxfp8_trtllm_supports_exact_blackwell_capabilities(
    compute_capability: int, expected: bool
) -> None:
    with (
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.kernels.linear.mxfp8.flashinfer.has_flashinfer",
            return_value=True,
        ),
    ):
        supported, _ = FlashInferTrtllmMxfp8LinearKernel.is_supported(
            compute_capability
        )

    assert supported is expected


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_mxfp8_trtllm_backend_selects_trtllm_kernel(platform_mock) -> None:
    platform_mock._enum = PlatformEnum.CUDA

    with (
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value="flashinfer_trtllm",
        ),
        patch.object(
            FlashInferTrtllmMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
    ):
        kernel = init_mxfp8_linear_kernel()

    assert isinstance(kernel, FlashInferTrtllmMxfp8LinearKernel)


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_mxfp8_trtllm_backend_is_not_considered_by_auto(platform_mock) -> None:
    platform_mock._enum = PlatformEnum.CUDA

    with (
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value="auto",
        ),
        patch.dict(
            _POSSIBLE_MXFP8_KERNELS,
            {PlatformEnum.CUDA: [FlashInferTrtllmMxfp8LinearKernel]},
        ),
        patch.object(
            FlashInferTrtllmMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
        pytest.raises(
            ValueError, match="Failed to find a kernel that can implement the MXFP8"
        ),
    ):
        init_mxfp8_linear_kernel()


def test_mxfp8_trtllm_prepares_padded_weight_and_scale(monkeypatch) -> None:
    calls: dict[str, tuple[tuple[int, ...], int, int | None]] = {}

    def shuffle_matrix_a(weight: torch.Tensor, tile_m: int) -> torch.Tensor:
        calls["weight"] = (tuple(weight.shape), tile_m, None)
        return weight

    def shuffle_matrix_sf_a(
        scale: torch.Tensor, tile_m: int, *, num_elts_per_sf: int
    ) -> torch.Tensor:
        calls["scale"] = (tuple(scale.shape), tile_m, num_elts_per_sf)
        return scale

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__dict__["shuffle_matrix_a"] = shuffle_matrix_a
    flashinfer.__dict__["shuffle_matrix_sf_a"] = shuffle_matrix_sf_a
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((130, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((130, 16), dtype=torch.uint8), requires_grad=False
    )

    _kernel().process_weights_after_loading(layer)

    assert calls == {
        "weight": ((256, 512), 128, None),
        "scale": ((256, 16), 128, 32),
    }
    assert layer.weight.shape == (256, 512)
    assert layer.weight_scale.shape == (4096,)
    assert layer._mxfp8_trtllm_output_size == 130

    _kernel().process_weights_after_loading(layer)
    assert layer.weight.shape == (256, 512)
    assert layer._mxfp8_trtllm_output_size == 130

    # Layerwise reload restores checkpoint-format tensors but keeps ordinary
    # Python attributes on the module. The backend must prepare them again.
    layer.weight = torch.nn.Parameter(
        torch.zeros((130, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((130, 16), dtype=torch.uint8), requires_grad=False
    )

    _kernel().process_weights_after_loading(layer)
    assert layer.weight.shape == (256, 512)
    assert layer.weight_scale.shape == (4096,)
    assert layer._mxfp8_trtllm_output_size == 130


def test_mxfp8_trtllm_rejects_unsupported_k(monkeypatch) -> None:
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__dict__["shuffle_matrix_a"] = lambda weight, tile_m: weight
    flashinfer.__dict__["shuffle_matrix_sf_a"] = lambda scale, tile_m, **kwargs: scale
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((128, 288), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((128, 9), dtype=torch.uint8), requires_grad=False
    )

    with pytest.raises(ValueError, match="K to be divisible by 256"):
        _kernel().process_weights_after_loading(layer)


def test_mxfp8_trtllm_uses_8x4_quantization_and_slices_output(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def quantize_8x4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls["quant_shape"] = tuple(x.shape)
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((128,), dtype=torch.uint8),
        )

    def mm_mxfp8(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype,
        backend: str,
        use_8x4_sf_layout: bool,
    ) -> torch.Tensor:
        calls["mm"] = {
            "a": tuple(a.shape),
            "b": tuple(b.shape),
            "a_scale": tuple(a_scale.shape),
            "b_scale": tuple(b_scale.shape),
            "out_dtype": out_dtype,
            "backend": backend,
            "use_8x4_sf_layout": use_8x4_sf_layout,
        }
        return torch.ones((a.shape[0], b.shape[1]), dtype=out_dtype)

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.vllm_flashinfer.flashinfer_mxfp8_quantize_8x4",
        quantize_8x4,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.flashinfer.vllm_flashinfer.mm_mxfp8",
        mm_mxfp8,
        raising=False,
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((256, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((4096,), dtype=torch.uint8), requires_grad=False
    )
    layer._mxfp8_trtllm_output_size = 130
    bias = torch.arange(130, dtype=torch.bfloat16)

    x = torch.zeros((2, 3, 512), dtype=torch.bfloat16)
    output = _kernel().apply_weights(layer, x)
    output_with_bias = _kernel().apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 130)
    assert output.is_contiguous()
    assert calls["quant_shape"] == (6, 512)
    assert calls["mm"] == {
        "a": (6, 512),
        "b": (512, 256),
        "a_scale": (128,),
        "b_scale": (4096,),
        "out_dtype": torch.bfloat16,
        "backend": "trtllm",
        "use_8x4_sf_layout": True,
    }
    torch.testing.assert_close(output_with_bias[0, 0], bias + 1)


def test_mxfp8_trtllm_rejects_float16_input() -> None:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((256, 512), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.zeros((4096,), dtype=torch.uint8), requires_grad=False
    )
    layer._mxfp8_trtllm_output_size = 256

    with pytest.raises(ValueError, match="requires bfloat16 output"):
        _kernel().apply_weights(layer, torch.zeros((1, 512), dtype=torch.float16))
