# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScaledMM kernel selection logic (CPU-only).

Run `pytest tests/kernels/quantization/test_scaled_mm_kernel_selection.py`.
"""

import inspect
import sys
from abc import ABC
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    AiterInt8ScaledMMLinearKernel,
    CPUInt8ScaledMMLinearKernel,
    HummingFP8ScaledMMLinearKernel,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    TritonFp8PerTokenScaledMMKernel,
    _get_linear_backend,
    _resolve_backend_kernels,
    init_fp8_linear_kernel,
    init_int8_linear_kernel,
    register_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.humming import prioritize_humming
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def test_auto_linear_backend_is_unchanged_without_overrides():
    kernels: list[type] = [object]
    quantizations = (
        "fp8_block_w8a8",
        "fp8_w8a8",
        "int8_w8a8",
        "mixed_precision",
        "mxfp8",
        "mxfp4",
        "mxfp6",
        "w8a16_fp8",
        "nvfp4_w4a4",
        "nvfp4_w4a16",
    )
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="auto"))

    with set_current_vllm_config(config):
        for quantization in quantizations:
            assert _get_linear_backend(quantization=quantization) == "auto"
            assert (
                _resolve_backend_kernels(kernels, "test", quantization=quantization)
                is kernels
            )


def test_linear_backend_override_is_quantization_specific():
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"nvfp4_w4a16": "humming"},
        )
    )

    with set_current_vllm_config(config):
        assert _get_linear_backend(quantization="nvfp4_w4a16") == "humming"
        assert _get_linear_backend(quantization="nvfp4_w4a4") == "cutlass"
        assert _get_linear_backend(quantization="fp8_w8a8") == "cutlass"


@patch.object(HummingFP8ScaledMMLinearKernel, "is_supported", return_value=(True, None))
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_fp8_linear_backend_override(platform_mock, _):
    platform_mock._enum = PlatformEnum.CUDA
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            linear_backend_per_quant={"fp8_w8a8": "humming"},
        )
    )

    with set_current_vllm_config(config):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(128, 128),
        )

    assert type(kernel) is HummingFP8ScaledMMLinearKernel


@pytest.mark.parametrize("supports_fp8", [False, True])
@patch("vllm.model_executor.kernels.linear.scaled_mm.triton.current_platform")
@patch("vllm.model_executor.kernels.linear.current_platform")
def test_triton_fp8_backend_requires_fp8(
    platform_mock, triton_platform_mock, supports_fp8, monkeypatch
):
    """The registered ROCm Triton backend requires FP8-capable hardware."""
    platform_mock._enum = PlatformEnum.ROCM
    platform_mock.get_device_capability.return_value = (12, 0)
    triton_platform_mock.supports_fp8.return_value = supports_fp8
    monkeypatch.setattr(
        "vllm.utils.platform_utils.num_compute_units", lambda *_args: 32
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic.get_device_cache_size",
        lambda *_args: 64 * 1024**2,
    )
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "")
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="triton"))
    kwargs = dict(
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticChannelSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=(256, 128),
    )
    with set_current_vllm_config(config):
        if supports_fp8:
            assert isinstance(
                init_fp8_linear_kernel(**kwargs), TritonFp8PerTokenScaledMMKernel
            )
        else:
            with pytest.raises(ValueError, match="requires FP8 support"):
                init_fp8_linear_kernel(**kwargs)


def test_fp8_heuristic_scales_occupancy_and_split_k():
    """Fewer processors can fill the GPU without the same split-K expansion."""
    from vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic_rules import (  # noqa: E501
        get_heuristic_config,
    )

    assert get_heuristic_config(128, 4096, 4096, 32)["engine"] == "contiguous"
    assert get_heuristic_config(128, 4096, 4096, 16)["engine"] == "shared"
    assert get_heuristic_config(8, 2048, 32768, 32)["split_k"] == 4
    assert get_heuristic_config(8, 2048, 32768, 16)["split_k"] == 1


def test_fp8_heuristic_scales_persistent_grid_without_mutating_cached_choices():
    """Selecting a smaller GPU must preserve cached choices for a larger GPU."""
    from vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic_rules import (  # noqa: E501
        get_heuristic_config,
    )

    large = get_heuristic_config(8192, 28672, 4096, 32)
    small = get_heuristic_config(8192, 28672, 4096, 16)
    assert large["variant"] == small["variant"] == "persistent"
    assert large["programs"] == 64
    assert small["programs"] == 32
    assert get_heuristic_config(8192, 28672, 4096, 32) == large


@pytest.mark.parametrize("num_processors", [0, -1])
def test_fp8_heuristic_requires_positive_processor_count(num_processors):
    from vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic_rules import (  # noqa: E501
        get_heuristic_config,
    )

    with pytest.raises(ValueError, match="num_processors must be positive"):
        get_heuristic_config(128, 4096, 4096, num_processors)


def test_fp8_heuristic_scales_cache_boundaries():
    from vllm.model_executor.layers.quantization.utils.fp8_per_token_heuristic_rules import (  # noqa: E501
        get_heuristic_config,
    )

    small = get_heuristic_config(65, 18944, 3584, 32, 32 * 1024**2)
    reference = get_heuristic_config(65, 18944, 3584, 32, 64 * 1024**2)
    large = get_heuristic_config(65, 18944, 3584, 32, 128 * 1024**2)
    assert small["block_size_n"] == 64
    assert reference["block_size_k"] == 64
    assert large["block_size_n"] == large["block_size_k"] == 128
    assert get_heuristic_config(65, 18944, 3584, 32, 0) == large
    with pytest.raises(ValueError, match="cache_size_bytes must be nonnegative"):
        get_heuristic_config(65, 18944, 3584, 32, -1)


def test_fp8_heuristic_device_cache_discovery_and_fallback(monkeypatch):
    """Use the visible GPU's last data-cache level, then L2 if SMI is unavailable."""
    from vllm.model_executor.layers.quantization.utils import (
        fp8_per_token_heuristic as heuristic,
    )

    handles = [object(), object()]
    observed = []

    def query(handle):
        observed.append(handle)
        return {
            "cache": [
                dict(
                    cache_properties=properties,
                    cache_level=level,
                    cache_size=size,
                    num_cache_instance=instances,
                )
                for properties, level, size, instances in [
                    (["DATA_CACHE"], 2, 8192, 1),
                    (["DATA_CACHE"], 3, 16384, 2),
                    (["INST_CACHE"], 4, 65536, 1),
                    (["DATA_CACHE", "CPU_CACHE"], 4, 65536, 1),
                    (["DATA_CACHE"], 2**32 - 1, 2**32 - 1, 2**32 - 1),
                    (["DATA_CACHE"], 4, 0, 1),
                ]
            ]
        }

    smi = SimpleNamespace(
        amdsmi_get_processor_handles=lambda: handles, amdsmi_get_gpu_cache_info=query
    )
    monkeypatch.setitem(sys.modules, "amdsmi", smi)
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(with_amdsmi_context=lambda fn: fn),
    )
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(visible_device_id_to_physical_device_id=lambda device: 1),
    )
    monkeypatch.setattr(
        "vllm.utils.platform_utils.cuda_get_device_properties",
        lambda *_args, **_kwargs: (8 * 1024**2,),
    )
    heuristic.get_device_cache_size.cache_clear()
    try:
        assert heuristic.get_device_cache_size(0) == 32 * 1024**2
        assert heuristic.get_device_cache_size(0) == 32 * 1024**2
        assert observed == [handles[1]]

        def unavailable(_handle):
            raise RuntimeError("SMI unavailable")

        monkeypatch.setattr(smi, "amdsmi_get_gpu_cache_info", unavailable)
        heuristic.get_device_cache_size.cache_clear()
        assert heuristic.get_device_cache_size(0) == 8 * 1024**2
    finally:
        heuristic.get_device_cache_size.cache_clear()


def test_is_supported_is_abstract():
    """Test that is_supported() is properly defined as abstract."""
    assert issubclass(ScaledMMLinearKernel, ABC)
    assert hasattr(ScaledMMLinearKernel, "is_supported")


def test_cpu_kernel_implements_is_supported():
    """Test that CPUInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(CPUInt8ScaledMMLinearKernel, "is_supported"), (
        "CPUInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        CPUInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(CPUInt8ScaledMMLinearKernel.is_supported), (
        "CPUInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    result, reason = CPUInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"


def test_aiter_kernel_implements_is_supported():
    """Test that AiterInt8ScaledMMLinearKernel implements is_supported() method."""
    assert hasattr(AiterInt8ScaledMMLinearKernel, "is_supported"), (
        "AiterInt8ScaledMMLinearKernel missing is_supported() method"
    )
    # Verify it's a classmethod by checking if it can be called with the class
    # and by checking the method type
    assert inspect.ismethod(
        AiterInt8ScaledMMLinearKernel.is_supported
    ) or inspect.isfunction(AiterInt8ScaledMMLinearKernel.is_supported), (
        "AiterInt8ScaledMMLinearKernel.is_supported() should be a classmethod"
    )
    # Verify it can be called as a classmethod
    # (will return False on CPU, which is expected)
    result, reason = AiterInt8ScaledMMLinearKernel.is_supported()
    assert isinstance(result, bool), "is_supported() should return a bool"
    assert reason is None or isinstance(reason, str), "reason should be str or None"
    # On CPU, it should return False with a reason about requiring ROCm
    # This validates the method works correctly even on non-ROCm platforms


def test_cpu_kernel_accepts_all_configs():
    """Test that CPUInt8ScaledMMLinearKernel accepts all config combinations."""
    configs = [
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=False,
            is_static_input_scheme=True,
            input_symmetric=True,
        ),
        Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=False,
        ),
    ]

    for config in configs:
        can_impl, reason = CPUInt8ScaledMMLinearKernel.can_implement(config)
        assert can_impl, (
            f"CPUInt8ScaledMMLinearKernel should accept config {config}: {reason}"
        )


class OOTInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_register_oot_linear_kernel(platform_mock):
    """Test that the linear kernel registration works correctly."""
    platform_mock._enum = PlatformEnum.OOT
    register_linear_kernel(OOTInt8ScaledMMLinearKernel, PlatformEnum.OOT, "int8")

    kernel = init_int8_linear_kernel(True, True, True, "module")

    assert isinstance(kernel, OOTInt8ScaledMMLinearKernel), (
        "init_int8_linear_kernel should return an instance of the registered kernel"
    )


@pytest.mark.parametrize(
    "possible_kernels", [_POSSIBLE_KERNELS, _POSSIBLE_NVFP4_KERNELS]
)
def test_humming_priority_only_reorders_humming_and_marlin(possible_kernels):
    """Preferring Humming must not demote Marlin below unrelated kernels."""
    kernels = possible_kernels[PlatformEnum.CUDA]
    original = list(kernels)

    assert prioritize_humming(kernels, compute_capability=80) == original

    names = [k.__name__ for k in prioritize_humming(kernels, compute_capability=90)]
    humming = next(i for i, name in enumerate(names) if "Humming" in name)
    assert "Marlin" in names[humming + 1]
    del names[humming]
    assert names == [k.__name__ for k in original if "Humming" not in k.__name__]
    assert kernels == original
